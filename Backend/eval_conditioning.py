"""
Does the question actually match the answer it was given?

The round-trip check in eval_roundtrip.py asks a different thing: can the model's
own QA head recover the answer from the question it just wrote. That hides the
real problem, because the QA head is itself only about 70% accurate on questions
a human wrote, so it passes bad questions and fails good ones.

This script asks something the QA head cannot fudge: when the answer is a date,
does the question ask about time. When the answer is a person, does it ask who.
Those are the cases where there is only one right question word, so a wrong one
is a real error, not a matter of taste.

It needs no model and no GPU. It reads eval_results.jsonl, which
eval_roundtrip.py already writes.

    python eval_conditioning.py
    python eval_conditioning.py some_other_results.jsonl
"""
import json
import re
import sys
from collections import defaultdict

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "eval_results.jsonl"

# A bare year or a 3-4 digit number on its own. Deliberately strict: "150,000"
# is a quantity and belongs to "how many", not "when".
BARE_YEAR = re.compile(r"^\s*\d{3,4}\s*$")

TIME_Q = {
    "en":  re.compile(r"^\s*(when|in what year|in which year|what year|which year|by what year|during what)", re.I),
    "fil": re.compile(r"^\s*(kailan|sa anong taon|anong taon|aling taon|noong anong|sa aling taon)", re.I),
}
PERSON_Q = {
    "en":  re.compile(r"^\s*(who|whose|whom)\b", re.I),
    "fil": re.compile(r"^\s*(sino|kanino|kaninong)\b", re.I),
}
# In Filipino si/ni/kay before a name marks a person with no ambiguity.
FIL_PERSON_MARK = r"\b(si|ni|kay|sina|nina|kina)\s+"
TITLE_CASE = re.compile(r"^(?:[A-Z][\w.'’-]*\s*){1,4}$")


def is_date_answer(row):
    return bool(BARE_YEAR.match(row["answer"]))


def is_person_answer(row):
    a = row["answer"].strip()
    if not TITLE_CASE.match(a):
        return False
    if row["lang"] == "fil":
        return re.search(FIL_PERSON_MARK + re.escape(a.split()[0]), row["context"], re.I) is not None
    # English: a title or an honorific right before the name is the safest signal.
    return re.search(r"\b(Mr|Mrs|Ms|Dr|President|King|Queen|Prince|Princess|Sir|Lord|Pope|"
                     r"Minister|Senator|Governor|General)\.?\s+" + re.escape(a.split()[0]),
                     row["context"]) is not None


CASES = [
    ("answer is a date",   is_date_answer,   TIME_Q,   "a time question"),
    ("answer is a person", is_person_answer, PERSON_Q, "a who question"),
]


def main():
    try:
        rows = [json.loads(line) for line in open(RESULTS, encoding="utf-8") if line.strip()]
    except FileNotFoundError:
        raise SystemExit(f"{RESULTS} not found. Run eval_roundtrip.py first, it writes that file.")

    need = {"lang", "answer", "context", "question", "gen_question"}
    missing = need - set(rows[0])
    if missing:
        raise SystemExit(f"{RESULTS} is missing {sorted(missing)}.")

    print(f"{len(rows)} rows from {RESULTS}\n")
    print("Each case below has only one right question word, so the model's score")
    print("can be read straight against the human score on the same rows.\n")

    header = f"{'case':<22}{'lang':<6}{'n':>5}{'human':>9}{'model':>9}{'gap':>8}"
    print(header)
    print("-" * len(header))

    worst = []
    for label, picker, pattern, _what in CASES:
        for lang in ("en", "fil"):
            sel = [r for r in rows if r["lang"] == lang and picker(r)]
            if not sel:
                continue
            pat = pattern[lang]
            human = sum(1 for r in sel if pat.match(r["question"]))
            model = sum(1 for r in sel if pat.match(r["gen_question"]))
            h, m = human / len(sel), model / len(sel)
            flag = "  <-- look" if (h - m) >= 0.25 and len(sel) >= 10 else ""
            print(f"{label:<22}{lang:<6}{len(sel):>5}{h:>8.0%}{m:>8.0%}{h - m:>8.0%}{flag}")
            if flag:
                worst.append((label, lang, sel, pat))

    # Which question words the model actually reaches for, against the data it
    # was trained to imitate. A model that has collapsed onto the commonest word
    # shows up here as one row far above the human share.
    print("\nQuestion words used, model vs the human questions in the same file")
    for lang in ("en", "fil"):
        sel = [r for r in rows if r["lang"] == lang]
        if not sel:
            continue
        words = ("ano", "sino", "kailan", "saan", "bakit", "paano", "ilan") if lang == "fil" \
            else ("what", "who", "when", "where", "why", "how many", "how", "which")
        print(f"\n  {lang} (n={len(sel)})")
        print(f"    {'word':<10}{'human':>8}{'model':>8}")
        counted_h = counted_m = 0
        for w in words:
            p = re.compile(r"^\s*" + w + r"\b", re.I)
            h = sum(1 for r in sel if p.match(r["question"]))
            m = sum(1 for r in sel if p.match(r["gen_question"]))
            counted_h += h
            counted_m += m
            print(f"    {w:<10}{h / len(sel):>7.0%}{m / len(sel):>7.0%}")
        print(f"    {'anything else':<10}{1 - counted_h / len(sel):>4.0%}{1 - counted_m / len(sel):>7.0%}")

    for label, lang, sel, pat in worst:
        bad = [r for r in sel if not pat.match(r["gen_question"])]
        print(f"\n=== {lang}, {label}: {len(bad)} wrong out of {len(sel)} ===")
        for r in bad[:6]:
            print(f"  answer  : {r['answer']}")
            print(f"  human   : {r['question'][:95]}")
            print(f"  model   : {r['gen_question'][:95]}\n")

    if not worst:
        print("\nNo case where the model trails the human score by 25 points or more.")


if __name__ == "__main__":
    main()
