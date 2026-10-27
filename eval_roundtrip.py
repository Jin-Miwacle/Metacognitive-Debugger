"""
Round-trip evaluation for the mT5 question generator.

For each held-out row:
  1. generate a question from (answer, passage)   <- same call as main.py make_question()
  2. answer that question with the QA head         <- same call as main.py answer_matches()
  3. PASS if the predicted answer matches the original answer (same rule as main.py)

Also reports how well the QA head answers the GOLD question (checks the QA task on its own).

Put this file in the Backend folder (next to qg_format.py and train_data_clean.jsonl) and run:
    python eval_roundtrip.py

Settings (environment variables):
    MODEL          model to test               (default: Aiken1016/MT5_Second_model)
    DATA_FILE      the file you trained on     (default: train_data_clean.jsonl)
    EVAL_FILE      OPTIONAL extra held-out jsonl, same format. Best honest test for Filipino.
    N_PER_GROUP    max rows per language/group (default: 150)
    EXPECTED_ROWS  sanity check on DATA_FILE   (default: 7266)
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

from qg_format import qg_input, qa_input

MODEL = os.getenv("MODEL", "Aiken1016/MT5_Second_model")
DATA_FILE = os.getenv("DATA_FILE", "train_data_clean.jsonl")
EVAL_FILE = os.getenv("EVAL_FILE", "")
N_PER_GROUP = int(os.getenv("N_PER_GROUP", "150"))
EXPECTED_ROWS = int(os.getenv("EXPECTED_ROWS", "7266"))
SRC_LEN = 768
SEED = 42


# ---- identical to train_mt5_1.py -------------------------------------------------------
def load_rows(path, verbose=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ctx, q, a = r.get("context", "").strip(), r.get("question", "").strip(), r.get("answer", "").strip()
            lang = r.get("lang", "en")
            if not (ctx and q and a) or a.lower() not in ctx.lower():
                continue
            rows.append({"context": ctx, "question": q, "answer": a, "lang": lang})
    return rows


def split_like_training(rows):
    """Recreates the exact validation rows used by train_mt5_1.py (row-level split, seed 42)."""
    rows = list(rows)
    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.05))
    return rows[n_val:], rows[:n_val]      # train_rows, val_rows
# -----------------------------------------------------------------------------------------


# ---- identical to main.py answer_matches ------------------------------------------------
def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def matches(predicted, answer):
    p, a = _norm(predicted), _norm(answer)
    return bool(p) and (a in p or p in a)
# -----------------------------------------------------------------------------------------


def load_model():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        print("tokenizer not found with the model, using google/mt5-small (same vocab)")
        tok = AutoTokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"model={MODEL} device={device}")

    def gen(prompt, **kw):
        x = tok(prompt, return_tensors="pt", truncation=True, max_length=SRC_LEN).to(device)
        with torch.inference_mode():
            out = model.generate(**x, max_new_tokens=48, num_beams=4, use_cache=True, **kw)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    return gen


def evaluate(gen, rows):
    results = []
    for r in rows:
        q = gen(qg_input(r["answer"], r["context"], r["lang"]), no_repeat_ngram_size=3, early_stopping=True)
        pred_on_generated = gen(qa_input(q, r["context"], r["lang"]))
        pred_on_gold = gen(qa_input(r["question"], r["context"], r["lang"]))
        results.append({
            **r,
            "gen_question": q,
            "pred_on_generated": pred_on_generated,
            "pred_on_gold": pred_on_gold,
            "roundtrip_pass": matches(pred_on_generated, r["answer"]),
            "qa_gold_pass": matches(pred_on_gold, r["answer"]),
        })
    return results


def main():
    rows = load_rows(DATA_FILE)
    print(f"{len(rows)} usable rows in {DATA_FILE}")
    if len(rows) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows. This is not the file the model was trained on, "
              f"so the held-out split below may include training rows. Results would be too optimistic.")
    train_rows, val_rows = split_like_training(rows)
    train_ctx = {r["context"] for r in train_rows}

    groups = defaultdict(list)
    for r in val_rows:
        seen = r["context"] in train_ctx
        groups[(r["lang"], "passage SEEN in training" if seen else "passage UNSEEN")].append(r)
    if EVAL_FILE:
        ext = load_rows(EVAL_FILE)
        for r in ext:
            groups[(r["lang"], "external EVAL_FILE")].append(r)

    random.seed(SEED)
    sample = {}
    for k, v in groups.items():
        random.shuffle(v)
        sample[k] = v[:N_PER_GROUP]

    print("\nGroup sizes (held-out rows available -> rows tested):")
    for k in sorted(groups):
        print(f"  {k[0]:>3} | {k[1]:<26} {len(groups[k]):>4} -> {len(sample[k])}")
    if not any(k[0] == "fil" and k[1] != "passage SEEN in training" for k in groups):
        print("\n  NOTE: no Filipino rows with an unseen passage. The Filipino numbers below are on passages "
              "the model saw in training (with different questions), so they are optimistic. "
              "Provide EVAL_FILE with unseen Filipino passages for an honest Filipino score.")

    gen = load_model()
    all_results = []
    print("\n=== RESULTS ===")
    print(f"{'lang':<5}{'group':<28}{'n':>5}{'round-trip pass':>18}{'QA on gold Q':>15}")
    for k in sorted(sample):
        if not sample[k]:
            continue
        res = evaluate(gen, sample[k])
        for x in res:
            x["group"] = k[1]
        all_results += res
        n = len(res)
        rt = sum(x["roundtrip_pass"] for x in res)
        qa = sum(x["qa_gold_pass"] for x in res)
        print(f"{k[0]:<5}{k[1]:<28}{n:>5}{rt:>12} ({rt / n:5.1%}){qa:>9} ({qa / n:5.1%})")

    with open("eval_results.jsonl", "w", encoding="utf-8") as f:
        for x in all_results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("\nsaved every row to eval_results.jsonl")

    for lang in ("fil", "en"):
        fails = [x for x in all_results if x["lang"] == lang and not x["roundtrip_pass"]]
        print(f"\n=== {lang.upper()}: 6 example round-trip FAILURES ({len(fails)} total) ===")
        for x in fails[:6]:
            print(f"- passage : {x['context'][:110]}...")
            print(f"  answer  : {x['answer']}")
            print(f"  gen Q   : {x['gen_question']}")
            print(f"  QA reply: {x['pred_on_generated']}")


if __name__ == "__main__":
    main()
