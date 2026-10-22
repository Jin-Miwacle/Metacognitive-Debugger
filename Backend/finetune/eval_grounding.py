"""
Evaluate the grounding rules with the real model, rules ON vs OFF (GROUNDING_RULES).

Input: JSONL, one question per line:
    {"passage": "...", "selection": "...", "question": "...", "expected_label": "STATED", "lang": "en"}
expected_label is one of STATED, INFERABLE, NOT_STATED, NOT_IN_TEXT, AMBIGUOUS.

Each question is sent the way /partner handles a question typed in the chat (mode "followup"):
grounding (when on) and then the coach reply, with the same functions main.py uses.

Reports per run: branch accuracy and confusion matrix (rules ON only, since OFF has no labels),
correct refusal rate, false refusal rate, R5 violation rate of the final replies, mean latency.

Run from the project root (loads the models, needs the GPU; the backend must NOT be running):
    $env:HF_HUB_OFFLINE="1"; Backend\\.venv\\Scripts\\python.exe Backend\\finetune\\eval_grounding.py `
        --data Backend\\finetune\\grounding_seed.jsonl --out grounding_eval.json
Expect roughly 5-15 s per question per run on the RTX 3050.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Backend/, for main and grounding

LABELS = ["STATED", "INFERABLE", "NOT_STATED", "NOT_IN_TEXT", "AMBIGUOUS"]

def is_refusal(reply: str) -> bool:
    # Same check the backend uses for NOT_STATED replies (grounding.says_not_stated).
    import grounding
    return grounding.says_not_stated(reply)


def paragraph_of(passage: str, selection: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", passage) if p.strip()]
    for p in paragraphs:
        if selection and selection.lower() in p.lower():
            return p
    return paragraphs[0] if paragraphs else ""


def run(items, rules_on: bool, main, gr):
    os.environ["GROUNDING_RULES"] = "1" if rules_on else "0"
    rows = []
    for n, it in enumerate(items, 1):
        ask = main.Ask(mode="followup", problem="other", passage=it.get("selection", ""),
                       paragraph=paragraph_of(it["passage"], it.get("selection", "")),
                       full_text=it["passage"], message=it["question"],
                       response_language=it.get("lang", "en"), history=[])
        t = time.time()
        g = main.run_grounding(ask)
        if g is not None and g.label == gr.NOT_IN_TEXT:
            reply = gr.fallback_reply(g, main.pick_language(ask))
        else:
            reply = main.generate_hint(ask, g)
        secs = time.time() - t
        violations = gr.check_reply(reply, it["passage"], it["question"], [], it.get("selection", ""))
        rows.append({
            "question": it["question"], "expected": it["expected_label"],
            "predicted": g.label if g is not None else None, "rule": g.rule if g is not None else None,
            "evidence_ids": g.evidence_ids if g is not None else [],
            "reply": reply, "refusal": is_refusal(reply), "r5_violations": violations, "seconds": round(secs, 2),
        })
        print(f"[{'ON' if rules_on else 'OFF'} {n}/{len(items)}] {secs:5.1f}s expected={it['expected_label']} "
              f"predicted={rows[-1]['predicted']} refusal={rows[-1]['refusal']} r5={violations}\n    {reply}")
    return rows


def rate(hits, total):
    return round(hits / total, 3) if total else None


def summarize(rows, rules_on: bool):
    should_refuse = [r for r in rows if r["expected"] in ("NOT_STATED", "NOT_IN_TEXT")]
    stated = [r for r in rows if r["expected"] == "STATED"]
    out = {
        "rules": "ON" if rules_on else "OFF",
        "n": len(rows),
        "correct_refusal_rate": rate(sum(r["refusal"] for r in should_refuse), len(should_refuse)),
        "false_refusal_rate": rate(sum(r["refusal"] for r in stated), len(stated)),
        "r5_violation_rate": rate(sum(bool(r["r5_violations"]) for r in rows), len(rows)),
        "mean_latency_s": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
    }
    if rules_on:
        out["branch_accuracy"] = rate(sum(r["expected"] == r["predicted"] for r in rows), len(rows))
        pairs = Counter((r["expected"], r["predicted"] or "SKIPPED") for r in rows)
        cols = LABELS + ["SKIPPED"]
        out["confusion"] = {e: {p: pairs.get((e, p), 0) for p in cols} for e in LABELS}
    return out


def print_summary(s):
    print(f"\n=== Rules {s['rules']} (n={s['n']}) ===")
    for k in ("branch_accuracy", "correct_refusal_rate", "false_refusal_rate", "r5_violation_rate",
              "mean_latency_s"):
        if k in s:
            print(f"  {k:22} {s[k]}")
    if "confusion" in s:
        cols = LABELS + ["SKIPPED"]
        print("  confusion (rows = expected, columns = predicted):")
        print("    " + " " * 12 + "".join(f"{c[:11]:>12}" for c in cols))
        for e in LABELS:
            print("    " + f"{e:12}" + "".join(f"{s['confusion'][e][c]:>12}" for c in cols))


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="labeled JSONL")
    ap.add_argument("--out", default="grounding_eval.json", help="where to write all replies and the summary")
    ap.add_argument("--runs", default="on,off", help="which runs: on, off or on,off")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[:args.limit]
    bad = [it for it in items if it.get("expected_label") not in LABELS]
    if bad:
        sys.exit(f"Unknown expected_label in: {bad[0]}")

    import main  # loads both models (slow); imported here so --help works without a GPU
    import grounding as gr

    results = {}
    for which in [w.strip().lower() for w in args.runs.split(",") if w.strip()]:
        rows = run(items, which == "on", main, gr)
        results[which] = {"summary": summarize(rows, which == "on"), "rows": rows}
    for r in results.values():
        print_summary(r["summary"])
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main_cli()
