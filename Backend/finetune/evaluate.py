"""Validity checks for the fine-tuned Socratic tutor.

Run the SAME command twice, once without --adapter (baseline) and once with it, then compare:
    python finetune/evaluate.py --data data/socratic.jsonl --tag base
    python finetune/evaluate.py --data data/socratic.jsonl --adapter checkpoints/socratic-lora/best --tag tuned

Reports (on the held-out split, default 'test'):
  * assistant-token loss / perplexity (teacher forced)
  * answer-leak rate           (tutor reveals the correct option before the student does)
  * question rate              (non-final tutor turns that contain a question)
  * single-question rate       (non-final turns with exactly one question mark)
  * average reply length vs reference
and writes every generated reply to eval_<tag>.jsonl so a teacher/human can read them.
The leak check is a heuristic: it is also run on the REFERENCE replies as a sanity check
(should be ~0%), and flagged replies still need a human look.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import (build_example, eval_loss, leaks_answer, load_dataset, load_model,
                    parse_correct, prepare_messages)


def words(s):
    return len(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/philiri_dialogues"],
                    help="one or more folders / jsonl files; several are merged split by split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=os.getenv("SEALION_MODEL", "aisingapore/Gemma-SEA-LION-v3-9B-IT"))
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the baseline model")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--max_convs", type=int, default=30, help="conversations to generate for")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--skip_generation", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for d in args.data:
        rows += load_dataset(d).get(args.split, [])
    if not rows:
        raise SystemExit(f"No '{args.split}' data found in {args.data} "
                         f"(expected e.g. chat_{args.split}.jsonl).")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = load_model(args.model, args.adapter)
    model.eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": args.tag, "adapter": args.adapter, "split": args.split, "n_conversations": len(rows)}

    # ---- 1. teacher-forced loss on the whole split ----
    exs = [e for e in (build_example(tok, r["messages"], args.max_len) for r in rows) if e]
    loss = eval_loss(model, exs)
    summary.update(loss=round(loss, 4), perplexity=round(math.exp(loss), 3))
    print(f"[{args.tag}] {args.split} loss {loss:.4f}  ppl {math.exp(loss):.2f}")

    # ---- 2. generation checks ----
    if not args.skip_generation:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_convs]
        records = []
        for n, r in enumerate(rows, 1):
            letter, text = parse_correct(r["messages"][0]["content"])
            msgs = prepare_messages(tok, r["messages"])
            a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            for turn_no, i in enumerate(a_idx):
                prompt = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                reply = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ref, last_user = msgs[i]["content"], msgs[i - 1]["content"]
                records.append({
                    "id": r.get("id"), "turn": turn_no + 1, "final_turn": i == a_idx[-1],
                    "student": last_user[-300:], "reference": ref, "generated": reply,
                    "leak": leaks_answer(reply, letter, text, last_user, turn_no),
                    "ref_leak": leaks_answer(ref, letter, text, last_user, turn_no),
                    "n_questions": reply.count("?"), "words": words(reply), "ref_words": words(ref),
                })
            print(f"  generated {n}/{len(rows)} conversations", end="\r")
        print()

        mid = [x for x in records if not x["final_turn"]]
        fin = [x for x in records if x["final_turn"]]
        rate = lambda xs, f: round(100 * sum(1 for x in xs if f(x)) / max(len(xs), 1), 1)
        summary.update(
            generated_turns=len(records),
            leak_rate_pct=rate(records, lambda x: x["leak"]),
            reference_leak_rate_pct=rate(records, lambda x: x["ref_leak"]),
            question_rate_pct=rate(mid, lambda x: x["n_questions"] >= 1),
            final_turn_question_rate_pct=rate(fin, lambda x: x["n_questions"] >= 1),
            single_question_rate_pct=rate(mid, lambda x: x["n_questions"] == 1),
            avg_words=round(sum(x["words"] for x in records) / len(records), 1),
            avg_reference_words=round(sum(x["ref_words"] for x in records) / len(records), 1),
        )
        with open(out_dir / f"eval_{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for x in records:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(out_dir / f"summary_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nPer-turn replies for human review: {out_dir / ('eval_' + args.tag + '.jsonl')}")


if __name__ == "__main__":
    main()