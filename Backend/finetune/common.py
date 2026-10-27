"""Shared helpers for fine-tuning and evaluating the Socratic tutor.

IMPORTANT: Backend/main.py should build its prompts with prepare_messages()
from this file, so inference sees exactly the same format as training.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test",
}


def normalize_split(value):
    return SPLIT_ALIASES.get(str(value).strip().lower())


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_from_filename(stem):
    s = stem.lower()
    for key, split in (("train", "train"), ("valid", "val"), ("val", "val"),
                       ("dev", "val"), ("test", "test")):
        if key in s:
            return split
    return None


def load_dataset(path):
    """Return {'train': [...], 'val': [...], 'test': [...]}.

    `path` can be a folder (every *.jsonl in it, e.g. chat_train.jsonl,
    chat_val.jsonl, chat_test.jsonl) or a single file. A row's own "split"
    field wins; otherwise the split is guessed from the file name.
    """
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"No .jsonl files found at {path}")
    out = defaultdict(list)
    for f in files:
        guess = _split_from_filename(f.stem)
        for r in load_jsonl(f):
            split = normalize_split(r.get("split")) or guess
            if split:
                out[split].append(r)
    return out


# --------------------------------------------------------------------------
# Chat formatting
# --------------------------------------------------------------------------
_SYSTEM_OK = {}


def supports_system(tok):
    """Gemma-2 style templates reject (or drop) the system role."""
    key = id(tok)
    if key not in _SYSTEM_OK:
        try:
            out = tok.apply_chat_template(
                [{"role": "system", "content": "SYSMARKER"},
                 {"role": "user", "content": "u"}],
                tokenize=False,
            )
            _SYSTEM_OK[key] = "SYSMARKER" in out
        except Exception:
            _SYSTEM_OK[key] = False
    return _SYSTEM_OK[key]


def prepare_messages(tok, messages):
    """Return messages the tokenizer's chat template can render.

    If the template has no system role, the system prompt is merged into the
    first user turn. Call this ONCE on the full conversation, then slice.
    """
    if supports_system(tok) or not messages or messages[0]["role"] != "system":
        return messages
    system_text = messages[0]["content"]
    rest = [dict(m) for m in messages[1:]]
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = system_text + "\n\n---\n\n" + rest[0]["content"]
    return rest


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build_example(tok, messages, max_len):
    """Tokenize one conversation; only assistant tokens get labels.

    Every assistant turn in the conversation is trained on (system and
    student turns are masked with -100). Returns None if too long.
    """
    msgs = prepare_messages(tok, messages)
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = encode(tok, full_text)
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    n_sup = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False)
        start, end = len(encode(tok, prefix)), min(len(encode(tok, upto)), len(ids))
        labels[start:end] = ids[start:end]
        n_sup += max(0, end - start)
    if n_sup == 0:
        return None
    return {"input_ids": ids, "labels": labels}


# --------------------------------------------------------------------------
# Loss (memory-lean: only compute logits where there is a label)
# --------------------------------------------------------------------------
def to_tensors(ex, device="cuda"):
    ids = torch.tensor([ex["input_ids"]], device=device)
    return ids, torch.ones_like(ids), torch.tensor([ex["labels"]], device=device)


def lean_loss(model, input_ids, attention_mask, labels):
    """Cross-entropy over supervised tokens only (batch size 1).

    Gemma has a 256k vocabulary, so full-sequence logits are huge. Asking the
    model for logits only at the positions we need saves ~1-2 GB of VRAM.
    """
    shift = labels[0, 1:]
    keep = (shift != -100).nonzero(as_tuple=True)[0]
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                logits_to_keep=keep, use_cache=False)
    logits = out.logits[0].float()
    targets = shift[keep]
    return F.cross_entropy(logits, targets), int(targets.numel())


@torch.no_grad()
def eval_loss(model, examples):
    """Token-weighted mean loss over assistant tokens."""
    model.eval()
    total, count = 0.0, 0
    for ex in examples:
        ids, am, lb = to_tensors(ex)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, k = lean_loss(model, ids, am, lb)
        total += loss.item() * k
        count += k
    return total / max(count, 1)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(name, adapter=None):
    from packaging import version
    import transformers
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = dict(
        quantization_config=bnb,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # recommended for Gemma-2 (attention soft-capping)
    )
    # Keep the (huge) embedding matrix in bf16 instead of fp32.
    if version.parse(transformers.__version__) >= version.parse("4.56"):
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model


# --------------------------------------------------------------------------
# Socratic-quality heuristics (used by evaluate.py)
# --------------------------------------------------------------------------
def parse_correct(system_text):
    m = re.search(r"Correct answer \(for the tutor only[^)]*\):\s*([a-d])\.\s*(.+)",
                  system_text, re.I)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).strip().lower().rstrip(".")


def leaks_answer(reply, letter, text, last_user, turn_no):
    """Heuristic: does the tutor reveal the answer before the student says it?

    From turn 2 on, a reply is exempt if the student's latest message already
    contains the correct option text (the tutor is then confirming, not leaking).
    Turn 1 is never exempt because the student's first message pastes all options.
    """
    r = reply.lower()
    if turn_no > 0 and text and text in last_user.lower():
        return False
    if text and text in r:
        return True
    if letter and re.search(
        rf"\b(correct\s+)?(answer|choice|option)\s*(is|would be|should be|:)\s*\(?{letter}\b", r
    ):
        return True
    return False