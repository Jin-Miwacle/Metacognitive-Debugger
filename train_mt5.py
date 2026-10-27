"""
Fine-tune mT5 on two tasks (question generation + question answering) for English and Filipino.

Data: a JSONL file, one object per line:
    {"context": "...", "question": "...", "answer": "...", "lang": "en"}      # lang is "en" or "fil"
The answer must appear in the context (the highlight needs it).

Run:
    pip install transformers datasets accelerate sentencepiece protobuf
    python train_mt5.py

Settings can be changed with environment variables, e.g.
    BASE_MODEL=google/mt5-base EPOCHS=5 python train_mt5.py
"""
import json
import os
import random

import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from qg_format import qg_input, qa_input

# ---- settings ---------------------------------------------------------------
DATA_FILE   = os.getenv("DATA_FILE", "train_data_clean.jsonl")
BASE_MODEL  = os.getenv("BASE_MODEL", "google/mt5-small")   # or Aiken1016/MT5_Final_Model to continue from it
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "MT5_Second_model")
TOKENIZER   = os.getenv("TOKENIZER", "google/mt5-small")   # same vocab as every mT5 size
EPOCHS      = float(os.getenv("EPOCHS", "4"))
LR          = float(os.getenv("LR", "3e-4"))                 # Adafactor + mT5 likes 1e-4 to 3e-4
BATCH       = int(os.getenv("BATCH", "4"))
GRAD_ACCUM  = int(os.getenv("GRAD_ACCUM", "4"))
PATIENCE    = int(os.getenv("PATIENCE", "2"))   # stop after this many epochs with no improvement
SRC_LEN     = 768   # MUST match max_length in main.py
TGT_LEN     = 64
SEED        = 42
# -----------------------------------------------------------------------------

random.seed(SEED)


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ctx, q, a = r.get("context", "").strip(), r.get("question", "").strip(), r.get("answer", "").strip()
            lang = r.get("lang", "en")
            if not (ctx and q and a):
                print(f"skip line {i}: missing field")
                continue
            if a.lower() not in ctx.lower():
                print(f"skip line {i}: answer not found in context")
                continue
            rows.append({"context": ctx, "question": q, "answer": a, "lang": lang})
    return rows


def to_examples(row):
    return [
        {"src": qg_input(row["answer"], row["context"], row["lang"]), "tgt": row["question"]},
        {"src": qa_input(row["question"], row["context"], row["lang"]), "tgt": row["answer"]},
    ]


def main():
    rows = load_rows(DATA_FILE)
    if not rows:
        raise SystemExit(f"No usable rows in {DATA_FILE}. Check the file has data and each answer appears in its context.")
    print(f"{len(rows)} usable rows -> {len(rows) * 2} training examples")
    if len(rows) < 50:
        print("WARNING: very little data. Expect weak results; aim for thousands of rows.")

    # Split by ROW (not by example) so a row's QG and QA examples never land in different splits.
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.05))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    def build(rs):
        return Dataset.from_list([e for r in rs for e in to_examples(r)])

    train_ds, val_ds = build(train_rows), build(val_rows)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        x = tok(batch["src"], max_length=SRC_LEN, truncation=True)
        x["labels"] = tok(text_target=batch["tgt"], max_length=TGT_LEN, truncation=True)["input_ids"]
        return x

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["src", "tgt"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["src", "tgt"])

    # mT5 gives NaN losses in fp16. Use bf16 if the GPU supports it, otherwise fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"bf16={use_bf16} (fp16 is never used for mT5)")

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        optim="adafactor",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="linear",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    last = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    print("Resuming from:", last)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    sanity_check(model, tok, val_rows[:5])


def sanity_check(model, tok, rows):
    """Print what the model produces for a few held-out rows, for both tasks."""
    model.eval()
    device = next(model.parameters()).device

    def gen(prompt):
        x = tok(prompt, return_tensors="pt", truncation=True, max_length=SRC_LEN).to(device)
        with torch.inference_mode():
            out = model.generate(**x, max_new_tokens=48, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)

    for r in rows:
        print("\n--- context:", r["context"][:120], "...")
        print("answer given     :", r["answer"])
        print("question (gold)  :", r["question"])
        print("question (model) :", gen(qg_input(r["answer"], r["context"], r["lang"])))
        print("answer (model)   :", gen(qa_input(r["question"], r["context"], r["lang"])))


if __name__ == "__main__":
    main()