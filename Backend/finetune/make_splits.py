"""Split a chat-format jsonl into chat_train / chat_valid / chat_test files.

Rows are grouped before splitting, so the same question (or passage) never appears
in two splits. Your ids show each question repeated across personas/runs, so a plain
row-level split would leak near-duplicates from train into validation/test.

Usage:
    python Backend\\finetune\\make_splits.py ^
        --src "D:\\...\\philiri_dialogues\\chat_train.jsonl" ^
        --out_dir "Backend\\data\\philiri_split"

The originals are never modified; the new files go to --out_dir.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="one or more jsonl files or folders (a folder contributes every chat_*.jsonl in it)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_by", choices=["question", "passage"], default="question",
                    help="'passage' is stricter (test passages are never seen in training) but needs more passages")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, seen_content, dupes = [], set(), 0
    for src in args.src:
        p = Path(src)
        files = sorted(p.rglob("chat_*.jsonl")) if p.is_dir() else [p]  # rglob: also searches subfolders
        if not files:
            raise SystemExit(f"No chat_*.jsonl files found in or under {src}")
        for fp in files:
            n_file = 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if not r.get("messages"):
                        continue
                    key = json.dumps(r["messages"], sort_keys=True, ensure_ascii=False)
                    if key in seen_content:   # same conversation present twice (e.g. in two files)
                        dupes += 1
                        continue
                    seen_content.add(key)
                    rows.append(r)
                    n_file += 1
            print(f"read {n_file:4d} rows from {fp}")
    if dupes:
        print(f"skipped {dupes} duplicate conversations")
    if not rows:
        raise SystemExit("No usable rows (need a 'messages' field).")

    if args.group_by == "question":
        key = lambda r: (r.get("passage_id"), r.get("question_number"))
    else:
        key = lambda r: r.get("passage_id")
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    keys = sorted(groups, key=str)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    print(f"{len(rows)} rows in {n} {args.group_by} groups")

    n_test = max(1, round(n * args.test_frac)) if args.test_frac > 0 and n >= 3 else 0
    n_val = max(1, round(n * args.val_frac)) if args.val_frac > 0 and n >= 2 else 0
    if n_test + n_val >= n:
        raise SystemExit(f"Only {n} groups - not enough to hold out validation/test and still train. "
                         f"Try --group_by question, a smaller --test_frac, or get more data.")

    parts = {
        "test": keys[:n_test],
        "val": keys[n_test:n_test + n_val],
        "train": keys[n_test + n_val:],
    }
    names = {"train": "chat_train.jsonl", "val": "chat_valid.jsonl", "test": "chat_test.jsonl"}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, ks in parts.items():
        count = 0
        with open(out / names[split], "w", encoding="utf-8") as f:
            for k in ks:
                for r in groups[k]:
                    r = dict(r)
                    r["split"] = split
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{split:5s}: {len(ks):3d} groups, {count:3d} rows -> {out / names[split]}")


if __name__ == "__main__":
    main()