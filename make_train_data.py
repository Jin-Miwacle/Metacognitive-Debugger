"""
Build train_data.jsonl for train_mt5.py from:
  - English: SQuAD (Kaggle "stanfordu/stanford-question-answering-dataset")
  - Filipino: FilQuAD train.csv

Run:
    pip install kagglehub pandas
    python make_train_data.py

Settings (environment variables):
    FIL_CSV     path to FilQuAD train.csv           (default: train.csv)
    SQUAD_JSON  path to a SQuAD train json file     (default: download with kagglehub)
    EN_ROWS     how many English rows to keep       (default: 4000)
    OUT         output file                         (default: train_data.jsonl)
"""
import glob
import json
import os
import random

import pandas as pd

FIL_CSV = os.getenv("FIL_CSV", "train.csv")
SQUAD_JSON = os.getenv("SQUAD_JSON", "")
EN_ROWS = int(os.getenv("EN_ROWS", "4000"))
OUT = os.getenv("OUT", "train_data.jsonl")
MAX_ANSWER_WORDS = 8   # quiz answers are short phrases

random.seed(42)


def find_squad_json():
    if SQUAD_JSON:
        return SQUAD_JSON
    import kagglehub
    path = kagglehub.dataset_download("stanfordu/stanford-question-answering-dataset")
    hits = sorted(glob.glob(os.path.join(path, "**", "train*.json"), recursive=True))
    if not hits:
        raise SystemExit(f"No train*.json found under {path}. Files there: {os.listdir(path)}")
    print("using", hits[0])
    return hits[0]


def load_english():
    with open(find_squad_json(), encoding="utf-8") as f:
        data = json.load(f)["data"]
    rows = []
    for article in data:
        for para in article["paragraphs"]:
            ctx = para["context"].strip()
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                ans = qa["answers"][0]["text"].strip()
                if not ans or len(ans.split()) > MAX_ANSWER_WORDS:
                    continue
                rows.append({"context": ctx, "question": qa["question"].strip(),
                             "answer": ans, "lang": "en"})
    print(f"English: {len(rows)} usable rows in SQuAD")
    random.shuffle(rows)
    return rows[:EN_ROWS]


def load_filipino():
    df = pd.read_csv(FIL_CSV).dropna(subset=["Passage", "Question", "Answer"])
    n0 = len(df)
    df = df[df["Relevant"].astype(str) == "True"]
    df = df[df["Correct"].astype(str) == "True"]
    df = df[~df["Naturalness"].isin([1, 2])]          # -1 means unrated, so keep those
    rows = []
    for r in df.itertuples():
        ctx, q, a = str(r.Passage).strip(), str(r.Question).strip(), str(r.Answer).strip()
        if a and a.lower() in ctx.lower() and len(a.split()) <= MAX_ANSWER_WORDS:
            rows.append({"context": ctx, "question": q, "answer": a, "lang": "fil"})
    print(f"Filipino: kept {len(rows)} of {n0} FilQuAD rows")
    return rows


def main():
    rows = load_english() + load_filipino()
    random.shuffle(rows)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    en = sum(r["lang"] == "en" for r in rows)
    print(f"wrote {len(rows)} rows to {OUT}  (en={en}, fil={len(rows) - en})")


if __name__ == "__main__":
    main()