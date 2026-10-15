"""Count the fine-tuning data per split (train / val / test).

Usage (from the project root, venv active):
    python Backend\\finetune\\count_splits.py --data Backend\\data\\philiri_split_all
    python Backend\\finetune\\count_splits.py --data Backend\\data\\philiri_split_all --max_len 768

Reports, per split: conversations, distinct questions and passages, tutor (assistant)
turns, student turns, and the language / question-level breakdown. It also checks that no
question or passage appears in more than one split.

With --max_len N it also loads the tokenizer (no model, no GPU) and reports token lengths
and how many conversations are longer than N tokens; train.py and evaluate.py drop those.
"""
import argparse
import json
import os
from collections import Counter

from common import build_example, load_dataset

SPLITS = ("train", "val", "test")


def question_key(r):
    # Same key train.py uses for its leakage warning.
    return (r.get("passage_id"), r.get("question_number"))


def count_split(rows):
    turns = Counter(m["role"] for r in rows for m in r["messages"])
    return {
        "conversations": len(rows),
        "questions": len({question_key(r) for r in rows}),
        "passages": len({r.get("passage_id") for r in rows}),
        "tutor_turns": turns["assistant"],
        "student_turns": turns["user"],
        "lang": dict(Counter(r.get("lang", "?") for r in rows)),
        "question_level": dict(Counter(r.get("question_level", "?") for r in rows)),
    }


def overlap(by_split, key):
    sets = {s: {key(r) for r in by_split.get(s, [])} for s in SPLITS}
    return {f"{a}&{b}": len(sets[a] & sets[b])
            for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}


def token_stats(tok, rows, max_len):
    lens = []
    for r in rows:
        # A huge limit so nothing is dropped here; we want every length.
        ex = build_example(tok, r["messages"], max_len=10**9)
        if ex:
            lens.append(len(ex["input_ids"]))
    if not lens:
        return {}
    lens.sort()
    return {
        "mean_tokens": round(sum(lens) / len(lens)),
        "median_tokens": lens[len(lens) // 2],
        "max_tokens": lens[-1],
        f"over_{max_len}": sum(1 for n in lens if n > max_len),
        f"kept_at_{max_len}": sum(1 for n in lens if n <= max_len),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Backend/data/philiri_split_all",
                    help="folder with chat_train/chat_valid/chat_test .jsonl, or one jsonl with a 'split' field")
    ap.add_argument("--max_len", type=int, default=None,
                    help="also report token lengths and how many examples exceed this")
    ap.add_argument("--model", default=os.getenv("SEALION_MODEL", "aisingapore/Gemma-SEA-LION-v3-9B-IT"))
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    by_split = load_dataset(args.data)
    result = {s: count_split(by_split.get(s, [])) for s in SPLITS}
    result["total"] = count_split([r for s in SPLITS for r in by_split.get(s, [])])

    if args.max_len:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        for s in SPLITS:
            result[s].update(token_stats(tok, by_split.get(s, []), args.max_len))

    q_overlap = overlap(by_split, question_key)
    p_overlap = overlap(by_split, lambda r: r.get("passage_id"))

    if args.json:
        print(json.dumps({"splits": result, "question_overlap": q_overlap,
                          "passage_overlap": p_overlap}, indent=2, ensure_ascii=False))
        return

    cols = ["conversations", "questions", "passages", "tutor_turns", "student_turns"]
    if args.max_len:
        cols += ["mean_tokens", "max_tokens", f"over_{args.max_len}"]
    names = list(SPLITS) + ["total"]
    print(f"{'':16}" + "".join(f"{n:>10}" for n in names))
    for c in cols:
        print(f"{c:16}" + "".join(f"{str(result[n].get(c, '')):>10}" for n in names))
    print()
    for s in names:
        print(f"{s:6} lang: {result[s]['lang']}  level: {result[s]['question_level']}")
    print()
    print(f"Questions shared between splits: {q_overlap}  (should all be 0)")
    print(f"Passages shared between splits:  {p_overlap}  (non-zero is expected with --group_by question)")


if __name__ == "__main__":
    main()
