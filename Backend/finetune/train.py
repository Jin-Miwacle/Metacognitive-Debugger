"""QLoRA fine-tuning for the Socratic reading tutor (SEA-LION Gemma-2 9B IT).

Usage (from the Backend folder, venv active, HF_HUB_OFFLINE=1 if the model is cached):
    python finetune/train.py --data data/socratic.jsonl --out checkpoints/socratic-lora

Writes the best adapter (lowest validation loss) to <out>/best and the final one to <out>/last.
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from common import build_example, eval_loss, lean_loss, load_dataset, load_model, to_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/philiri_dialogues",
                    help="folder containing chat_train/chat_val/chat_test .jsonl, or a single jsonl with a 'split' field")
    ap.add_argument("--model", default=os.getenv("SEALION_MODEL", "aisingapore/Gemma-SEA-LION-v3-9B-IT"))
    ap.add_argument("--out", default="checkpoints/socratic-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5, help="optimizer steps")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - this needs the CUDA build of torch.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    by_split = load_dataset(args.data)
    print({k: len(v) for k, v in by_split.items()})
    if not by_split["train"] or not by_split["val"]:
        raise SystemExit("Need train and validation data (files named like chat_train.jsonl / "
                         "chat_val.jsonl, or rows with split = train / val).")

    # Same passage+question must not appear in two splits (personas/repeats are near-duplicates).
    seen = defaultdict(set)
    for split_name, split_rows in by_split.items():
        for r in split_rows:
            seen[(r.get("passage_id"), r.get("question_number"))].add(split_name)
    leaky = [k for k, v in seen.items() if len(v) > 1]
    if leaky:
        print(f"WARNING: {len(leaky)} passage/question pairs appear in more than one split "
              f"(e.g. {leaky[:3]}). Validation loss will look better than it should.")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    def build(split):
        exs = [build_example(tok, r["messages"], args.max_len) for r in by_split[split]]
        kept = [e for e in exs if e]
        print(f"{split}: kept {len(kept)}/{len(exs)} (dropped = longer than {args.max_len} tokens)")
        return kept

    train, val = build("train"), build("val")
    lens = [len(e["input_ids"]) for e in train]
    print(f"train tokens/example: mean {sum(lens)/len(lens):.0f}, max {max(lens)}")

    # Sanity check on masking: this should print ONLY tutor replies.
    ex = train[0]
    sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print("\n--- supervised text of first example (must be tutor turns only) ---")
    print(tok.decode(sup))
    print("--------------------------------------------------------------------\n")

    # ---------------- model ----------------
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb

    model = load_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # NOTE: prepare_model_for_kbit_training is deliberately NOT used: it upcasts the
    # 256k-vocab embedding matrix to fp32 (+2 GB), which will not fit on an 8 GB GPU.
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    print(f"{len(train)} examples/epoch -> {steps_per_epoch} optimizer steps/epoch, {total_steps} total")

    # ---------------- train ----------------
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    best = float("inf")
    step, t0 = 0, time.time()
    try:
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(train)))
            random.shuffle(order)
            running, micro = 0.0, 0
            for i, idx in enumerate(order):
                ids, am, lb = to_tensors(train[idx])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = lean_loss(model, ids, am, lb)
                (loss / args.grad_accum).backward()
                running += loss.item()
                micro += 1
                if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0 or step == 1:
                        rec = {"step": step, "epoch": epoch + 1, "train_loss": running / micro,
                               "lr": sched.get_last_lr()[0],
                               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                               "elapsed_min": round((time.time() - t0) / 60, 1)}
                        print(rec)
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        running, micro = 0.0, 0
            vl = eval_loss(model, val)
            print(f"== epoch {epoch+1}: val_loss {vl:.4f} (ppl {math.exp(vl):.2f})")
            log.write(json.dumps({"epoch": epoch + 1, "val_loss": vl}) + "\n")
            log.flush()
            if vl < best:
                best = vl
                model.save_pretrained(out / "best")
                tok.save_pretrained(out / "best")
                print("   saved new best adapter")
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "\nCUDA out of memory. Try, in order: --max_len 768, --lora_r 8, close other GPU apps "
            "(browser hardware acceleration), or train on a bigger GPU (Colab/Kaggle/cloud)."
        )

    model.save_pretrained(out / "last")
    print(f"Done. Best val loss {best:.4f}. Adapter: {out/'best'}")


if __name__ == "__main__":
    main()