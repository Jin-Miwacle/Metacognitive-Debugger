"""
Reading Partner backend.

The widget (index.html) talks to three addresses on this server:
    POST /partner   ->   { "reply": "..." }       a hint while the student is debugging
    POST /quiz      ->   { "questions": [ ... ] } a quiz written from the student's text
    POST /checkin   ->   { "prompt": "..." }      a short mid-reading check-in question

Usage:
    1. On terminal, enter: huggingface-cli login
    2. Complete steps given and paste token
    3. Run python main.py
    4. Open index.html to test the website    
"""
import json
import os
import random
import time
from typing import List, Optional
import re

# from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from pathlib import Path
from peft import PeftModel
import contextlib
import threading

import grounding as gr  # rule-based checks against the passage (see GROUNDING_RULES_README.md)


# Load SEALION Model
SEALION_MODEL_NAME = os.getenv("SEALION_MODEL", "aisingapore/Gemma-SEA-LION-v3-9B-IT")
ADAPTER_PATH = os.getenv(
    "SEALION_ADAPTER",
    str(Path(__file__).resolve().parent / "checkpoints" / "socratic-lora-v2" / "best"),
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"Loading base model {SEALION_MODEL_NAME} + adapter {ADAPTER_PATH} (this may take a moment)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(SEALION_MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    SEALION_MODEL_NAME,
    quantization_config=quantization_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

if ADAPTER_PATH.strip().lower() == "none":
    print("[adapter] SEALION_ADAPTER=none: running the base SEA-LION model only. "
          "Quizzes work, but the reading coach will not use the Socratic adapter.")
elif (
    # a local folder that is missing or has no adapter_config.json (a "user/repo" id is left to the Hub)
    (os.path.isdir(ADAPTER_PATH) and not os.path.isfile(os.path.join(ADAPTER_PATH, "adapter_config.json")))
    or (not os.path.exists(ADAPTER_PATH) and not re.fullmatch(r"[\w.\-]+/[\w.\-]+", ADAPTER_PATH))
):
    raise SystemExit(
        f"No adapter found at {ADAPTER_PATH}\n"
        "Copy your trained adapter folder there, point SEALION_ADAPTER at it, "
        "or set SEALION_ADAPTER=none to run without the adapter."
    )
else:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

model.generation_config.max_length = None
print("SEALION Model loaded successfully!")

# Load T5 Model
from qg_format import qg_input, qa_input, detect_lang

T5_MODEL_NAME = os.getenv("QG_MODEL", "Aiken1016/MT5_Second_model")  # fix the username if different
QG_CHECK_ANSWER = os.getenv("QG_CHECK_ANSWER", "1") != "0"

print(f"Loading mT5 question-generation model: {T5_MODEL_NAME}...")
qg_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
qg_device = torch.device(os.getenv("QG_DEVICE", "cpu"))
qg_model.to(qg_device).eval()

def _qg_generate(prompt: str, **gen_kwargs) -> str:
    inputs = qg_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(qg_device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = qg_model.generate(**inputs, max_new_tokens=48, num_beams=4, use_cache=True, **gen_kwargs)
    return qg_tokenizer.decode(out[0], skip_special_tokens=True).strip()

def make_question(answer, passage, lang):
    return _qg_generate(qg_input(answer, passage, lang), no_repeat_ngram_size=3, early_stopping=True)

def answer_matches(question, answer, passage, lang):
    if not QG_CHECK_ANSWER:
        return True
    predicted = _qg_generate(qa_input(question, passage, lang))
    n = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return bool(n(predicted)) and (n(answer) in n(predicted) or n(predicted) in n(answer))

# How many questions to try per answer phrase, and how long an answer phrase may be.
# Measured on held-out FilQuAD: Filipino answers of 4-8 words passed the round-trip check
# about 49% of the time, 1-3 words about 66%. English showed no such drop.
QG_CANDIDATES = max(1, int(os.getenv("QG_CANDIDATES", "3")))
QG_SAMPLE = os.getenv("QG_SAMPLE", "1") != "0"
MAX_ANSWER_WORDS = {
    "fil": int(os.getenv("QG_MAX_WORDS_FIL", "3")),
    "en": int(os.getenv("QG_MAX_WORDS_EN", "8")),   # 8 = longest answer mT5 was trained on
}

def make_questions(answer, passage, lang, k=QG_CANDIDATES):
    """Top-k beam candidates for one answer phrase, best first, without duplicates.
    The first one is the same question make_question() returns."""
    inputs = qg_tokenizer(qg_input(answer, passage, lang), return_tensors="pt",
                          truncation=True, max_length=768)
    inputs = {key: v.to(qg_device) for key, v in inputs.items()}
    with torch.inference_mode():
        out = qg_model.generate(**inputs, max_new_tokens=48, num_beams=max(4, k),
                                num_return_sequences=k, no_repeat_ngram_size=3,
                                early_stopping=True, use_cache=True)
    seen, result = set(), []
    for seq in out:
        q = qg_tokenizer.decode(seq, skip_special_tokens=True).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)

    n_beam = len(result)

    # Beam search often returns near-identical questions. When it gives fewer than k different
    # ones, first try diverse beam search (beam groups that are penalised for repeating each other),
    # then hot sampling. QG_SAMPLE=0 turns both off.
    def _add(seqs):
        for seq in seqs:
            if len(result) >= k:
                break
            q = qg_tokenizer.decode(seq, skip_special_tokens=True).strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                result.append(q)

    n_diverse = 0
    if QG_SAMPLE and len(result) < k:
        try:
            with torch.inference_mode():
                diverse = qg_model.generate(**inputs, max_new_tokens=48, num_beams=6, num_beam_groups=3,
                                            diversity_penalty=1.5, num_return_sequences=6,
                                            no_repeat_ngram_size=3, do_sample=False, use_cache=True)
            _add(diverse)
        except Exception as e:
            print(f"[Quiz] Diverse beam search unavailable ({type(e).__name__}: {str(e)[:80]}); using sampling only")
        n_diverse = len(result) - n_beam
    if QG_SAMPLE and len(result) < k:
        with torch.inference_mode():
            sampled = qg_model.generate(**inputs, max_new_tokens=48, do_sample=True, top_k=50, top_p=0.95,
                                        temperature=1.3, num_return_sequences=k + 4,
                                        no_repeat_ngram_size=3, use_cache=True)
        _add(sampled)
    print(f"[Quiz] Candidates for '{answer}': {n_beam} from beam search, {n_diverse} from diverse beams, "
          f"{len(result) - n_beam - n_diverse} sampled")
    return result

def _cmp(text):
    return re.sub(r"\W+", " ", str(text).casefold()).strip()

def _contains(a, b):
    """Whole-word containment: '1' is not inside '1524'."""
    return f" {a} " in f" {b} "

def option_matches_span(option, span):
    """True if SEA-LION's wording of the correct answer still says what the verified phrase says."""
    o, s_ = _cmp(option), _cmp(span)
    if not o or not s_:
        return False
    if _contains(s_, o) or _contains(o, s_):
        return True
    span_words = set(s_.split())
    return len(span_words & set(o.split())) / len(span_words) >= 0.5

def clean_distractors(distractors, correct_option, span):
    """Drop duplicates and any distractor that equals, contains or sits inside the correct
    answer (or the verified phrase), since those can be correct too."""
    refs = [r for r in (_cmp(correct_option), _cmp(span)) if r]
    keep, seen = [], set()
    for d in distractors:
        c = _cmp(d)
        if not c or c in seen:
            continue
        if any(_contains(c, r) or _contains(r, c) for r in refs):
            continue
        seen.add(c)
        keep.append(str(d).strip())
    return keep

# ---- extra question checks -------------------------------------------------------------
_ANSWER_IS_QUANTITY = re.compile(r"\d")   # digits only: "ninety-one steps" is an ordinary noun phrase
_QUANTITY_QUESTION = re.compile(
    r"\bhow (?:many|much|long|far|tall|high|old|big|large|deep|heavy|often|wide)\b"
    r"|\bwhen\b|\bwhat (?:year|time|age|number|date)\b"
    r"|\b(?:gaano|ilan|ilang|kailan|magkano|taon|taas|haba|bigat|laki|lalim|bilang|edad|oras|petsa)\b",
    re.I)

# SAME_SENTENCE_CHECK=0 turns off the "distractor in the answer's sentence" rejection
# if it throws away too many questions on long-sentenced passages.
SAME_SENTENCE_CHECK = os.getenv("SAME_SENTENCE_CHECK", "1") != "0"

_NOT_PAST = {"exceed", "proceed", "succeed", "indeed", "hundred", "bleed", "breed", "speed", "agreed"}
_GARBLED_DO = re.compile(
    r"\b(?i:did|does|do)\s+(?:(?i:she|he|they|it|i|you|we)|[A-Z][a-z]+)\s+"
    r"((?i:\w{3,}ed|went|came|took|made|saw|ran|got|gave|began|wrote|found|told|knew|kept|left|felt|"
    r"brought|taught|was|were))\b")

def question_is_garbled(question):
    """English only: mT5 sometimes writes 'What did she turned ...?' (did/does/do + a past-tense verb).
    A pattern check, not a grammar checker: it only catches this one mistake."""
    m = _GARBLED_DO.search(question)
    return bool(m) and m.group(1).lower() not in _NOT_PAST

QG_MAX_COPY = int(os.getenv("QG_MAX_COPY", "8"))

def copies_passage(question, passage, run=None):
    """True if the question repeats `run` or more words in a row straight from the passage
    (mT5 sometimes glues two clauses of the passage into one 'question')."""
    run = run or QG_MAX_COPY
    q, p = _cmp(question).split(), f" {_cmp(passage)} "
    return any(" " + " ".join(q[i:i + run]) + " " in p for i in range(len(q) - run + 1))

def question_fits_answer(question, answer):
    """A number or amount answer needs a question that asks for a number or amount
    ('What is the peak called?' cannot have '2,463 metres' as its answer).
    Only short answers containing a digit are treated as quantities, so
    'ninety-one steps' or 'three long flashes and one short' are not forced into a 'how many' question."""
    a = answer.strip()
    if len(a.split()) > 3 or not _ANSWER_IS_QUANTITY.search(a):
        return True
    return bool(_QUANTITY_QUESTION.search(question))

def sentences_with(passage, span):
    """The passage sentences that contain the verified answer span."""
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    s_ = _cmp(span)
    return [x for x in sentences if _contains(s_, _cmp(x))]

def distractor_in_answer_sentence(distractors, passage, span):
    """First distractor that is also written in the same sentence as the answer, or None.
    The passage then says it too, so it can be defensible (Albay and Bicol in one sentence)."""
    sents = sentences_with(passage, span)
    for d in distractors:
        c = _cmp(d)
        if c and any(_contains(c, _cmp(x)) for x in sents):
            return d
    return None

def support_for(item, idx, option_items, total):
    """SEA-LION's entry for this question, matched by question text rather than by position.
    Falls back to position only when it returned exactly one entry per question."""
    key = _cmp(item["q"])
    for entry in option_items:
        if isinstance(entry, dict) and _cmp(entry.get("q", "")) == key:
            return entry
    if len(option_items) == total and isinstance(option_items[idx], dict):
        return option_items[idx]
    return None

app = FastAPI(title="Reading Partner backend")

ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
print(f"CORS allowed origins: {ALLOWED_ORIGINS}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SYSTEM_SUPPORTED = None
_GEN_LOCK = threading.Lock()

def _system_supported():
    global _SYSTEM_SUPPORTED
    if _SYSTEM_SUPPORTED is None:
        try:
            out = tokenizer.apply_chat_template(
                [{"role": "system", "content": "SYSMARKER"}, {"role": "user", "content": "u"}],
                tokenize=False)
            _SYSTEM_SUPPORTED = "SYSMARKER" in out
        except Exception:
            _SYSTEM_SUPPORTED = False
        print(f"[ask_sealion] chat template supports system role: {_SYSTEM_SUPPORTED}")
    return _SYSTEM_SUPPORTED

def _normalize(messages):
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    turns = [m for m in messages if m["role"] != "system" and m["content"].strip()]
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "(The student selected part of the text and asked for help.)"})
    merged = []
    for m in turns:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    if system is None:
        return merged
    if _system_supported():
        return [{"role": "system", "content": system}] + merged
    merged[0]["content"] = system + "\n\n---\n\n" + merged[0]["content"]
    return merged

def ask_sealion(system_rules: str, prompt: str, history: Optional[List["Turn"]] = None, *,
              temperature: float, max_tokens: int, use_adapter: bool = True) -> str:
    
    messages = [{"role": "system", "content": system_rules}]
    for turn in history or []:
        messages.append({"role": "assistant" if turn.role == "coach" else "user", "content": turn.text})
    messages.append({"role": "user", "content": prompt})
 
    formatted_prompt = tokenizer.apply_chat_template(
        _normalize(messages), tokenize=False, add_generation_prompt=True)
 
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
 
    stop_ids = [tokenizer.eos_token_id]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn is not None and end_of_turn != tokenizer.unk_token_id:
        stop_ids.append(end_of_turn)
 
    do_sample = temperature > 0.0
    adapter_off = (not use_adapter) and hasattr(model, "disable_adapter")
    ctx = model.disable_adapter() if adapter_off else contextlib.nullcontext()
    # One generation at a time. FastAPI runs these endpoints in a thread pool, so two
    # students at once would otherwise generate in parallel on one 8 GB GPU, and one
    # request's disable_adapter() would switch the adapter off under the other.
    with _GEN_LOCK, ctx, torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
 

# =====================================================================
# 1) HINTS  (POST /partner)
# =====================================================================
class Turn(BaseModel):
    role: str                      # "student" or "coach"
    text: str
    mode: Optional[str] = None 

class Ask(BaseModel):
    mode: str = "debug"            # from the widget: "debug", "followup", "stuck" (button) or "explain";
                                   # "giveup" is accepted but the backend decides that itself (should_give_up)
    problem: Optional[str] = None  # "vocab", "lost", "connect", "why", or "other" (typed by the student)
    problem_text: Optional[str] = None  # the student's own description, set when problem == "other"
    passage: str = ""              # words selected
    paragraph: str = ""            # paragraph around it
    full_text: str = ""            # the whole reading; the grounding rules check against this
    goal: Optional[str] = None     # reading goal
    goal_hint: Optional[str] = None  # a word shared by the selection and the goal (widget's keyword check)
    message: Optional[str] = None  # student latest reply
    history: List[Turn] = []       # conversation history
    response_language: Optional[str] = None  # saved Coach language setting: "en", "fil", or None to auto-detect

COACH_RULES = """You are a warm, encouraging reading coach for Grade 7 students. 

Each request contains an "Instruction for this turn". The backend has already decided what kind of turn this is. Follow that instruction exactly, even if it means not asking a question.

General Rules:
- OUTPUT ONLY YOUR SPOKEN RESPONSE AS THE COACH. Do NOT write "Coach:", "Student:", or simulate any dialogue back-other.
- Reply ONLY in the language named in the request.
- Keep your response under 45 words.
- Never use stock praise like "Good job."
- Never invent facts, characters, or plot points outside of the provided context.
- Never ask the student to say something "in your own words" or to repeat an explanation back to you.
- Never use analogies, metaphors, comparisons or examples from outside the text (no "it's like when you...", no "imagine...", no "have you ever..."). Talk only about what the text says.
- Use plain, easy-to-understand words."""

CLASSIFIER_RULES = """You are an intent classifier for a reading coach. You get the target word/passage, the coach's last message, and the student's reply.

Classify the student's reply into EXACTLY one of these three words:
- "success": The student shows they understand the target (gives a correct meaning, a correct translation such as a Filipino synonym, or correctly confirms a correct explanation).
- "stuck": The student gives up, says they don't know, asks to be told directly, or is completely lost.
- "followup": Anything else: a wrong or partial answer, another question, or continuing the conversation.
 
Output ONLY the exact category word. No punctuation, no explanation."""

_EN_REQUEST = re.compile(r"\b(in english|english please|english po|sa english|sa ingles|ingles po)\b", re.I)
_FIL_REQUEST = re.compile(r"\b(in tagalog|in filipino|tagalog please|filipino please|sa tagalog|sa filipino|tagalog po)\b", re.I)
_FIL_HINT = re.compile(r"\b(ang|ng|mga|sa|ay|na|hindi|ko|po|ito|siya|nila|kami|tayo|ba)\b", re.I)

# Words that may surround a language request without making the message an actual answer.
_REQUEST_FILLER = {
    "please", "pls", "plz", "po", "naman", "lang", "paki", "pwede", "puwede", "can", "could", "you",
    "speak", "talk", "reply", "answer", "explain", "say", "use", "in", "sa", "only", "again", "ulit",
    "instead", "the", "language", "it", "that", "mag", "ka", "mo", "ba", "gusto", "ko", "sana", "ako",
}

def language_request(text):
    if not text:
        return None
    for lang, pattern in (("en", _EN_REQUEST), ("fil", _FIL_REQUEST)):
        if pattern.search(text):
            rest = pattern.sub(" ", text)
            if all(w in _REQUEST_FILLER for w in re.findall(r"[a-z']+", rest.lower())):
                return lang
            return None
    return None

_LANG_ALIASES = {
    "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-ph": "en",
    "fil": "fil", "filipino": "fil", "tagalog": "fil", "tl": "fil", "fil-ph": "fil", "tl-ph": "fil",
}

def normalize_language(value):
    if not value:
        return None
    return _LANG_ALIASES.get(str(value).strip().lower())

def pick_language(ask):
    texts = [t.text for t in ask.history if t.role == "student"] + [ask.message or ""]
    for text in reversed(texts):
        lang = language_request(text)
        if lang:
            return lang
    # 2. The saved setting from the Coach language control.
    saved = normalize_language(ask.response_language)
    if saved:
        return saved
    # 3. Fallback: detect from the passage.
    sample = f"{ask.passage} {ask.paragraph}"
    return "fil" if len(_FIL_HINT.findall(sample)) >= 2 else "en"

def stuck_level(ask):
    earlier = sum(1 for t in ask.history if t.role == "coach" and t.mode == "stuck")
    return earlier + 1

_PROMPT_LEAK = re.compile(
    r"(<start_of_turn>|<end_of_turn>|^\s*(user|model|assistant)\s*$|Context paragraph:|"
    r"Instruction for this turn:|Student's latest reply:|Language for your reply:|Reading goal( \(the student's own words\))?:|"
    r"Key sentences from the text)",
    re.I | re.M,
)
 
def clean_reply(text):
    m = _PROMPT_LEAK.search(text)
    if m:
        text = text[:m.start()]
    text = re.sub(r'^(Coach|Assistant|AI|Reading Partner)\s*:\s*', '', text.strip(), flags=re.IGNORECASE)
    return text.strip()
 
def last_coach_line(ask):
    for t in reversed(ask.history):
        if t.role == "coach":
            return t.text
    return ""
 
def classify_intent(message, target_passage="", last_coach=""):
    if not message:
        return "followup"
    prompt = (f"Target word/passage: \"{target_passage}\"\n"
              f"Coach's last message: \"{last_coach[:300]}\"\n"
              f"Student reply: \"{message[:300]}\"\nCategory:")
    
    raw_output = ask_sealion(CLASSIFIER_RULES, prompt, history=None, temperature=0.1, max_tokens=5, use_adapter=False)
    cleaned = raw_output.strip().lower()

    if "success" in cleaned:
        return "success"
    if "stuck" in cleaned:
        return "stuck"
    return "followup"

_QUESTION_BY_PROBLEM = {
    "vocab":   "ONE simple question to help them figure out what \"{p}\" means using the words around it in the text.",
    "lost":    "ONE simple question about what is happening here.",
    "connect": "ONE simple question pointing back to the sentence right before it.",
    "why":     "ONE simple question to look for an earlier clue in the text.",
}

def _full_explanation(p):
    return (f"The student has been stuck several times. Now explain \"{p}\" clearly and simply using "
            f"the context only, with no analogy or outside example. Do NOT ask them to repeat or restate it. End with "
            f"one short, friendly line inviting them to keep reading.")

def _re_explanation(p, before):
    return (f"You already explained \"{p}\" like this: \"{before[:300]}\". The student is still confused, so "
            f"explain it again in a DIFFERENT way: do NOT reuse those sentences. Use shorter sentences and "
            f"simpler words, and quote the exact words from the text that show it. Use the context only, with "
            f"no analogy or outside example. Do NOT ask them to repeat or restate it. End with one short, "
            f"friendly line inviting them to keep reading.")

def earlier_explanations(ask):
    return [t.text for t in ask.history if t.role == "coach" and t.mode == "giveup"]

def repeats_earlier(reply, earlier):
    # Near-copies count too: most of the words the same as an earlier explanation.
    words = set(gr._words(reply))
    for e in earlier:
        old = set(gr._words(e))
        if words and old and len(words & old) / len(words | old) >= 0.7:
            return True
    return False

_RE_EXPLAIN_FALLBACK = {
    "en": "Let's look at the exact words in the text again: \"{quote}\" Read that sentence slowly. "
          "Keep reading, and highlight anything else that's confusing.",
    "fil": "Balikan natin ang eksaktong salita sa teksto: \"{quote}\" Basahin mo ito nang dahan-dahan. "
           "Magpatuloy sa pagbasa, at i-highlight ang anumang nakalilito pa.",
}

def re_explain_fallback(ask, lang, g):
    # Built from the text itself, quoting a sentence not quoted before, so it doesn't repeat either.
    sents = gr.split_sentences(ask.paragraph or "")
    ids = gr.selection_ids(sents, ask.passage or "")
    picks = list(g.evidence if g is not None else []) + [sents[i - 1] for i in ids] + sents
    replies = [_RE_EXPLAIN_FALLBACK[lang].format(quote=gr._short_quote(s, 20)) for s in picks]
    said = {t.text for t in ask.history if t.role == "coach"}
    fresh = [r for r in replies if r not in said]
    if fresh:
        return fresh[0]
    return replies[0] if replies else _KEEP_READING[lang]

def goal_instruction(ask):
    if not ask.goal or ask.mode in ("repeat", "stuck"):
        return ""
    if ask.mode == "success":
        return (" Then add ONE short sentence on how understanding this helps with their reading goal, "
                "if it does.")
    if ask.goal_hint:
        return (f" This part is related to their reading goal (it mentions \"{ask.goal_hint}\"), so "
                f"connect your question to that goal.")
    return (" Only if this part clearly connects to their reading goal, mention that link briefly; "
            "otherwise do not mention the goal.")

def problem_said(ask):
    if ask.problem == "other" and ask.problem_text and ask.problem_text.strip():
        return ask.problem_text.strip()[:200]
    return None

def mode_instruction(ask):
    p = ask.passage or "this part"
    said = problem_said(ask)
    if said:
        question = (f"ONE short, simple question aimed at exactly the trouble they described "
                    f"(\"{said}\"), not at some other difficulty you notice in the passage.")
    else:
        question = _QUESTION_BY_PROBLEM.get(ask.problem or "lost", "ONE simple guiding question.").format(p=p)
    text = _mode_instruction(ask, p, question)
    if said and ask.mode in ("stuck", "giveup"):
        text += f" They had described the trouble as: \"{said}\"."
    return text + goal_instruction(ask)

def _mode_instruction(ask, p, question):
    if ask.mode == "success":
        return (f"The student has understood \"{p}\". Confirm they are right, "
                f"restate the meaning in ONE short sentence using the context, and tell them they "
                f"are ready to move on to more passages. Do NOT ask any question.")
    if ask.mode == "stuck":
        level = stuck_level(ask)
        if level == 1:
            return (f"The student is stuck. Do NOT give the answer or explain \"{p}\". Point them to the "
                    f"exact sentence in the context that holds the clue (quote a few words from it), "
                    f"then ask ONE easier question, such as a yes/no or either/or question.")
        if level == 2:
            return (f"The student is still stuck. Do NOT give the full answer. Explain ONE piece of the "
                    f"idea behind \"{p}\" using the context, then ask the student to finish the idea.")
        return _full_explanation(p)
    if ask.mode == "giveup":
        before = earlier_explanations(ask)
        if before:
            return _re_explanation(p, before[-1])
        return _full_explanation(p)
    if ask.mode == "explain":
        return (f"Explain \"{p}\" in 2 to 4 short sentences at a Grade 7 level, using only the context and "
                f"key sentences given. If the text doesn't say something, say \"the text doesn't say\". Do NOT "
                f"ask them to repeat or restate it; end with one short line inviting them to keep reading.")
    if ask.mode == "repeat":
        return "The student asked you to change language. Repeat your last question in that language. Nothing else."
    if ask.mode == "followup":
        return (f"Respond briefly to what the student just said (don't say whether it's right yet), "
                f"then ask {question}")
    return f"Ask the student {question}"
 
def explaining(ask):
    return ask.mode in ("explain", "giveup") or (ask.mode == "stuck" and stuck_level(ask) >= 3)

def build_hint_prompt(ask, lang, g=None, retry_note=""):
    passage = ask.passage or "this part"
    lines = []
    if ask.goal:
        lines.append(f"Reading goal (the student's own words): \"{ask.goal[:200]}\"")
    if ask.paragraph:
        lines.append(f"Context paragraph: {ask.paragraph[:600]}")
    else:
        lines.append(f"Context: None provided. Treat '{passage}' as a general vocabulary term "
                     f"without inventing external characters or plot points.")
    if g is not None and g.evidence:
        lines.append("Key sentences from the text (quote only these, word for word; the student cannot see this "
                     "list, so say \"the text\", never \"the key sentence\"):\n"
                     + "\n".join(f"- {s}" for s in g.evidence))
    lines.append(f"The student is working on: \"{passage}\".")
    if ask.message:
        lines.append(f"Student's latest reply: \"{ask.message[:300]}\"")
    lines.append(f"Language for your reply: {'English' if lang == 'en' else 'Filipino (natural Taglish is OK)'}")
    instruction = mode_instruction(ask)
    branch = gr.branch_instruction(g, explaining(ask)) if g is not None else ""
    lines.append(f"\nInstruction for this turn: {instruction}" + (f" {branch}" if branch else ""))
    if retry_note:
        lines.append(retry_note)
    lines.append("\nOutput ONLY your single spoken response as the coach.")
    return "\n".join(lines)

SUCCESS_FALLBACK = {
    "en":  "That's right! You've figured out what \"{p}\" means. You're ready to move on to more passages.",
    "fil": "Tama! Nakuha mo na ang ibig sabihin ng \"{p}\". Handa ka nang magpatuloy sa susunod na babasahin.",
}

# ---- Grounding rules (Backend/grounding.py). GROUNDING_RULES=0 turns R1-R5 off for comparison runs.
def grounding_on():
    return os.getenv("GROUNDING_RULES", "1") != "0"

def rules_passage(ask):
    # The whole reading when the widget sends it; older widgets only send the paragraph.
    return (ask.full_text or ask.paragraph or "")[:20000]

def student_question(ask):
    if ask.message:
        return ask.message
    return (problem_said(ask) or "") if ask.mode == "debug" else ""

def run_grounding(ask):
    """R0-R3 for this turn, or None when the rules are off or the turn needs none."""
    # "success" is checked too: the classifier can call a wrong answer a success.
    if not grounding_on() or ask.mode == "repeat":
        return None
    earlier = [problem_said(ask) or ""] + [t.text for t in ask.history if t.role == "student"]
    g = gr.ground(ask_sealion, question=student_question(ask), passage=rules_passage(ask),
                  selection=ask.passage, problem=ask.problem, mode=ask.mode,
                  full_passage=bool(ask.full_text), earlier=earlier,
                  coach_line=last_coach_line(ask), claim_check=True)
    print(g.log_line())
    for note in g.notes:
        print(f"    {note}")
    return g

_GROUNDING_RETRY = ("\nImportant: use only names, numbers and quotes that appear in the text, copied word for "
                    "word, and do not change the spelling of any word from the text.")
_HEDGE_RETRY = (" Make clear this is a clue, not something the text says directly (use words like \"might\" "
                "or \"seems\").")
_REPEAT_RETRY = ("\nImportant: your last answer repeated your earlier explanation almost word for word. Write a "
                 "new explanation with different sentences.")
_AGREED_RETRY =" The text says something different from what the student said, so do NOT start by agreeing."
_NOT_STATED_RETRY = (" Start by saying plainly that the text doesn't say this (in Filipino: \"Hindi sinabi ng "
                     "teksto...\").")

def reply_problems(ask, reply, g):
    """R5 for every reply, plus R4 for INFERABLE ones and the "say it's not stated" check for NOT_STATED."""
    problems = gr.check_reply(reply, rules_passage(ask), ask.message or "", ask.history, ask.passage)
    if g is not None and g.label == gr.INFERABLE and not gr.is_meaning(g) and not gr.is_hedged(reply):
        problems.append("R4:unhedged")
    if g is not None and g.label == gr.NOT_STATED and not gr.says_not_stated(reply):
        problems.append("NOT_STATED:not_said")
    if g is not None and g.label == gr.CONTRADICTED and gr.agrees(reply):
        problems.append("R6:agreed")
    return problems

# "Say it in your own words" reads as a test and confuses students; the adapter sometimes adds it anyway.
_RESTATE_REQUEST = re.compile(
    r"[^.?!]*\b(own words|sariling (mga )?(salita|pananalita)|restate|say it back|explain it back)\b[^.?!]*[.?!]?\s*",
    re.I)

# "The key sentence says..." refers to a list in the prompt that the student never sees.
_KEY_SENTENCE = re.compile(r"\b(the\s+)?(first|second|third|last|other\s+)?\s*key\s+sentences?\b", re.I)

def name_the_text(reply):
    def fix(m):
        return "The text" if m.group(0)[0].isupper() else "the text"
    return _KEY_SENTENCE.sub(fix, reply)

def drop_restate_requests(reply):
    trimmed = _RESTATE_REQUEST.sub("", reply).strip()
    return trimmed or reply

# Analogies and outside examples ("it's like when you carry your lunchbox") confuse students more
# than they help, and the adapter adds them even when told not to.
_ANALOGY = re.compile(
    r"\b(like when|like how|just like|kind of like|sort of like|imagine|think of it|have you ever|"
    r"for example|for instance|similar to|the same way|parang (kapag|kung|noong|ikaw|tayo)|"
    r"katulad (ng|nang) kapag|tulad (ng|nang) kapag|halimbawa|isipin mo|naranasan mo na)\b"
    r"|^\W*(like|parang)\b", re.I)

def _without_analogy(sentence):
    m = _ANALOGY.search(sentence)
    if not m:
        return sentence
    # "Mandy held the eggs, like when you carry your lunchbox." keeps "Mandy held the eggs."
    head = re.split(r"\s*[,;—–-]\s*(?:\w+\s+){0,2}$", sentence[:m.start()])[0].rstrip(" ,;—–-")
    if head != sentence[:m.start()].rstrip() and len(head.split()) >= 3:
        return head + "."
    return ""

def drop_analogies(reply):
    sents = gr.split_sentences(reply)
    kept = [k for k in (_without_analogy(s) for s in sents) if k]
    if kept == sents:
        return reply
    print(f"[rules] removed analogies: {reply!r} -> {kept}")
    return " ".join(kept)

_KEEP_READING = {"en": "Keep reading, and highlight anything else that's confusing.",
                 "fil": "Magpatuloy sa pagbasa, at i-highlight ang anumang nakalilito pa."}

def drop_trailing_questions(reply, lang):
    # A full explanation that ends on a new question restarts the hint ladder it was meant to end.
    sents = gr.split_sentences(reply)
    kept = list(sents)
    while kept and kept[-1].rstrip().endswith("?"):
        kept.pop()
    if not kept or len(kept) == len(sents):
        return reply
    return " ".join(kept + [_KEEP_READING[lang]])

def _generate_reply(ask, prompt, lang, temperature):
    # Explain mode states the answer, which the Socratic adapter is trained not to do.
    raw_response = ask_sealion(COACH_RULES, prompt, history=ask.history, temperature=temperature,
                               max_tokens=150, use_adapter=ask.mode != "explain")
    reply = name_the_text(drop_analogies(drop_restate_requests(clean_reply(raw_response))))
    if explaining(ask):
        reply = drop_trailing_questions(reply, lang)

    if ask.mode == "success" and "?" in reply:
        reply = SUCCESS_FALLBACK[lang].format(p=ask.passage or "this part")
    if not reply:
        reply = ("Let's look at that part of the text again. What do you notice?" if lang == "en"
                 else "Balikan natin ang bahaging iyon ng teksto. Ano ang napapansin mo?")
    return reply

_CLOSING = re.compile(r"\b(keep (reading|going)|highlight|magpatuloy|try the next sentence)\b", re.I)

def usable(ask, kept):
    """What's left after R7 still does the turn's job: something besides a closing line, and for a
    hint, a question (a hint whose only question was off-text has nothing to ask)."""
    sents = [s for s in gr.split_sentences(kept) if not _CLOSING.search(s)]
    # At least one line has to be about the text ("That's a lot for one person!" alone explains nothing).
    passage = gr.split_sentences(rules_passage(ask))
    if not any(gr._term_in(t, p) for s in sents for t in gr.key_terms(s) for p in passage):
        return False
    return ask.mode == "success" or explaining(ask) or any(s.rstrip().endswith("?") for s in sents)

def fact_checked(ask, reply):
    """R7, then the explanation's no-trailing-question rule again (removing a line can leave one last)."""
    kept, invented = remove_invented(ask, reply)
    if explaining(ask):
        kept = drop_trailing_questions(kept, pick_language(ask))
    return kept, invented

def remove_invented(ask, reply):
    """R7: drop the reply's statements about the story that the text doesn't back, and word the
    guesses it gives clues for as guesses."""
    invented, note, unhedged = gr.check_facts(ask_sealion, reply, gr.split_sentences(rules_passage(ask)),
                                              ask.passage, student=ask.message or "")
    print(f"[grounding] {note}")
    if not invented and not unhedged:
        return reply, []
    if invented:
        print(f"[grounding] removed invented details: {invented}")
    if unhedged:
        print(f"[grounding] worded as guesses: {unhedged}")
    return gr.apply_fact_check(reply, invented, unhedged, pick_language(ask)), invented

def generate_hint(ask, g=None):
    lang = pick_language(ask)
    reply, template = _checked_reply(ask, g, lang)
    # FACT_CHECK=0 skips R7 (one extra model call per reply) while keeping the other rules.
    if template or not grounding_on() or not rules_passage(ask) or os.getenv("FACT_CHECK", "1") == "0":
        return reply
    kept, invented = fact_checked(ask, reply)
    if not invented:
        return kept          # unchanged, or with its guesses now worded as guesses
    if usable(ask, kept):
        return kept
    # Nothing but a closing line is left: try once more, told what not to say.
    note = ("\nImportant: the text does NOT say or mention these things, so do not say or ask them: "
            + " ".join(f"\"{s}\"" for s in invented) + " Only say and ask about what the key sentences and the "
            "context paragraph say.")
    print("[grounding] not enough left after removing invented details -> regenerating once")
    reply = _generate_reply(ask, build_hint_prompt(ask, lang, g, retry_note=note), lang, 0.3)
    if not reply_problems(ask, reply, g):
        kept, invented = fact_checked(ask, reply)
        if usable(ask, kept):
            return kept
    print("[grounding] still invented -> fallback template")
    if ask.mode == "success":
        return SUCCESS_FALLBACK[lang].format(p=ask.passage or "this part")
    if explaining(ask):
        return re_explain_fallback(ask, lang, g)
    return gr.fallback_reply(g or gr.GroundingResult(), lang)

def _checked_reply(ask, g, lang):
    """The reply after R4-R6 and the repeat check, and whether it is a code-built template."""
    temperature = 0.3 if ask.mode in ("success", "repeat") else 0.7
    reply = _generate_reply(ask, build_hint_prompt(ask, lang, g), lang, temperature)
    before = earlier_explanations(ask) if ask.mode == "giveup" else []
    if before and repeats_earlier(reply, before):
        print(f"[rules] explanation repeats an earlier one -> regenerating once: {reply!r}")
        reply = _generate_reply(ask, build_hint_prompt(ask, lang, g, retry_note=_REPEAT_RETRY), lang, 0.9)
        if repeats_earlier(reply, before):
            print(f"[rules] still repeating -> text-based fallback: {reply!r}")
            return re_explain_fallback(ask, lang, g), True
    if not grounding_on() or not rules_passage(ask):
        return reply, False

    problems = reply_problems(ask, reply, g)
    if not problems:
        return reply, False
    bad_quotes = [p[len("quote:"):] for p in problems if p.startswith("quote:")]
    if bad_quotes and len(bad_quotes) == len(problems):
        # Only the quotation marks are wrong (the words aren't the text's). Removing them keeps a
        # reply that may be fine; regenerating risks a worse one. R7 still checks what it says.
        fixed = gr.unquote(reply, bad_quotes)
        if not reply_problems(ask, fixed, g):
            print(f"[grounding] quotes not from the text -> removed the quotation marks: {reply!r}")
            return fixed, False
    print(f"[grounding] reply failed {problems} -> regenerating once: {reply!r}")
    note = (_GROUNDING_RETRY + (_HEDGE_RETRY if "R4:unhedged" in problems else "")
            + (_NOT_STATED_RETRY if "NOT_STATED:not_said" in problems else "")
            + (_AGREED_RETRY if "R6:agreed" in problems else ""))
    reply = _generate_reply(ask, build_hint_prompt(ask, lang, g, retry_note=note), lang,
                            0.3 if temperature > 0.3 else 0.0)
    problems = reply_problems(ask, reply, g)
    if [p for p in problems if p != "R4:unhedged"]:
        print(f"[grounding] still failing {problems} -> fallback template: {reply!r}")
        if ask.mode == "success":
            return SUCCESS_FALLBACK[lang].format(p=ask.passage or "this part"), True
        return gr.fallback_reply(g or gr.GroundingResult(), lang), True
    if problems:
        print("[grounding] still unhedged -> adding the INFERABLE hedge")
        return gr.TEMPLATES[lang]["hedge"] + reply, False
    return reply, False


# =====================================================================
# 2) QUIZ  (POST /quiz)  - the question-generation module
# =====================================================================
class QuizAsk(BaseModel):
    text: str
    goal: Optional[str] = None
    n: int = 3

QUIZ_RULES = """You write JSON-formatted reading-comprehension multiple choice questions.
You MUST output valid JSON matching this schema exactly:
{
  "questions": [
    {
      "kind": "Literal",
      "q": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0,
      "why": "Explanation why.",
      "evidence": "exact quote"
    }
  ]
}
Keep options balanced and write clear questions based ONLY on the text."""


# Curly quotes -> straight, one character for one, so positions in the passage are kept.
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

def find_evidence(evidence: str, passage: str) -> Optional[str]:
    """Locate the model's quote in the passage and return the passage's own text for it.

    The same words must appear in the same order, but line breaks, spacing, punctuation
    between words, curly vs straight quotes and case may differ: models routinely join
    the lines of a poem with ", " or turn ’ into '. Returns None if the words are not there.
    """
    words = re.findall(r"[\w']+", evidence.translate(_QUOTES))
    if not words:
        return None
    pattern = r"(?<![\w'])" + r"[^\w']+".join(re.escape(w) for w in words) + r"(?![\w'])"
    m = re.search(pattern, passage.translate(_QUOTES), re.IGNORECASE)
    if not m:
        return None
    return " ".join(passage[m.start():m.end()].split())

def generate_quiz(ask: QuizAsk) -> list:
    n = max(2, min(ask.n, 5))
    passage = ask.text.strip()[:4000]

    if not passage:
        raise ValueError("The passage cannot be empty.")
    
    lang = detect_lang(passage)
    max_words = MAX_ANSWER_WORDS.get(lang, 8)
    why_language = "Filipino" if lang == "fil" else "English"
    _clock = [time.perf_counter(), time.perf_counter()]   # [last lap, start]

    def lap(label):
        now = time.perf_counter()
        print(f"[Quiz][time] {label}: {now - _clock[0]:.1f}s")
        _clock[0] = now
    # Ask for a few extra targets: some questions fail the mT5 answer check and are dropped.
    n_targets = min(n + 3, 8)

    def parse_json_response(raw: str) -> dict:
        clean = raw.strip()

        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

        return json.loads(clean)

    # STEP 1: Use SEA-LION to identify answer phrases in the passage.
    # T5 requires an answer phrase to be highlighted in the context.
    answer_prompt = f"""
Read the passage and select exactly {n_targets} DISTINCT answer targets for a
reading-comprehension quiz.

QUESTION MIX:
- Select approximately half literal and half inferential targets.
- Include at least one inferential target whenever the passage supports it.

LITERAL:
- The answer is directly stated in the passage.
- The student can find the answer without drawing a conclusion.

INFERENTIAL:
- The answer requires connecting clues or drawing a reasonable conclusion.
- Choose targets that support questions about causes, motivations,
  implications, or lessons.
- Do not label a directly stated fact as inferential merely to meet the mix.

TARGET SELECTION:
- Each answer target must be a short phrase of at most {max_words} words (a name,
  date, number, place, term or short noun phrase), never a full sentence.
- Copy each answer target exactly from the passage.
- Each target must be ONE continuous phrase: a single noun phrase, name, place, number or date.
  Never a list of several items, never words joined with commas or semicolons, and never a bare
  adjective or description (for example not "dark, gloomy").
- For literal targets, select a meaningful fact or detail.
- For inferential targets, select a passage detail that provides evidence
  for a conclusion or interpretation.
- Select different parts of the passage.
- Do not repeat answer targets.
- Do not invent facts.

Return valid JSON only, using this format:
{{
  "items": [
    {{
      "kind": "Literal",
      "answer": "exact phrase copied from the passage"
    }},
    {{
      "kind": "Inferential",
      "answer": "exact phrase copied from the passage"
    }}
  ]
}}

Return exactly {n_targets} items if the passage supports that many distinct targets.

Passage:
{passage}
"""

    raw_answers = ask_sealion(
        "Select answer phrases from the supplied passage. Return valid JSON only.",
        answer_prompt,
        temperature=0.2,
        max_tokens=500,
        use_adapter=False,  # the Socratic adapter is trained to avoid stating answers
    )

    lap("step 1, SEA-LION picks answer targets")
    try:
        answer_data = parse_json_response(raw_answers)
        answer_items = answer_data.get("items", [])
    except Exception as e:
        raise RuntimeError(
            f"Could not parse answer phrases from SEA-LION: {e}"
        )

    # STEP 2: Use T5 to generate the actual question wording.
    generated = []
    used_questions = set()
    used_answers = set()

    for item in answer_items:
        if len(generated) >= n:
            break

        answer = str(item.get("answer", "")).strip()
        kind = str(item.get("kind", "Literal")).strip()

        if not answer:
            continue

        if re.search(r"[,;:]\s|\s[-–—]\s", answer):
            print(f"[Quiz] Skipped (target is a list or fragment): {answer}")
            continue

        start = passage.lower().find(answer.lower())
        if start == -1:
            print(f"[Quiz] Skipped (target not found word for word in the passage): {answer}")
            continue

        exact_answer = passage[start:start + len(answer)]

        if exact_answer.lower() in used_answers:
            print(f"[Quiz] Skipped (target already used): {exact_answer}")
            continue

        if len(exact_answer.split()) > max_words:
            print(f"[Quiz] Skipped (answer longer than {max_words} words): {exact_answer}")
            continue

        # Try the best few questions for this answer; keep the first one mT5 can answer back.
        question = None
        for candidate in make_questions(exact_answer, passage, lang):
            if candidate.lower() in used_questions:
                print(f"[Quiz] Skipped (same question as an earlier one): {candidate}")
                continue
            if lang == "en" and question_is_garbled(candidate):
                print(f"[Quiz] Dropped (ungrammatical question): {candidate}")
                continue
            if copies_passage(candidate, passage):
                print(f"[Quiz] Dropped (question copies {QG_MAX_COPY}+ words in a row from the passage): {candidate}")
                continue
            if not question_fits_answer(candidate, exact_answer):
                print(f"[Quiz] Dropped (answer is a number or amount but the question does not ask for one): "
                      f"{candidate} -> {exact_answer}")
                continue
            if answer_matches(candidate, exact_answer, passage, lang):
                question = candidate
                break
            print(f"[Quiz] Dropped (mT5 answer mismatch): {candidate}")
        if not question:
            print(f"[Quiz] No usable question for target: {exact_answer}")
            continue

        used_answers.add(exact_answer.lower())
        used_questions.add(question.lower())

        generated.append({
            "kind": kind,
            "q": question,
            "correct_text": exact_answer
        })

    lap("step 2, mT5 questions + answer checks (CPU unless QG_DEVICE=cuda)")
    if not generated:
        raise RuntimeError(
            "T5 could not generate questions from the selected answer phrases."
        )

    print(f"[Quiz] Requested questions: {n}")
    print(f"[Quiz] Answer targets returned: {len(answer_items)}")
    print(f"[Quiz] T5 questions generated: {len(generated)}")
    print(f"[Quiz] T5 questions: {generated}")

    # STEP 3: Use SEA-LION to create the answer options and supporting details.
    # Keep the question wording generated by T5 unchanged.
    questions_for_sealion = [
        {
            "kind": item["kind"],
            "q": item["q"],
            "correct_answer": item["correct_text"]
        }
        for item in generated
    ]

    
    options_prompt = f"""
Create {len(questions_for_sealion)} reading-comprehension multiple-choice questions.

PASSAGE:
{passage}

QUESTIONS AND CORRECT ANSWERS:
{json.dumps(questions_for_sealion, ensure_ascii=False, indent=2)}

RULES:

GENERAL:
- Use ONLY facts supported by the passage.
- Keep every supplied question exactly unchanged.
- Preserve the supplied "kind".
- Generate exactly 4 answer choices per question.
- Ensure exactly one option is correct based on the passage.
- Include the correct answer exactly once.
- Create 3 plausible but clearly incorrect distractors.
- Make all 4 choices distinct. Do not repeat ideas or use synonyms.
- Never use "None of the above" or "All of the above".
- Set "answer" to the zero-based index of the correct option.
- Verify that the selected answer index points to the correct option.

ANSWER CHOICES:
- Write the correct option as a direct answer to the question.
- Prefer a concise answer of 2–8 words when grammatically appropriate.
- Do not simply copy a sentence or clause from the passage.
- Identify what the question is asking before writing the answer.
- For "How" questions, describe the process or method.
- For "What helps" questions, identify the thing that provides the help.
- For "Why" questions, state the reason.
- For "What causes" questions, identify the cause.
- Preserve the original meaning of the supplied correct answer.
- The correct option must answer the question independently.
- Do not use a passage sentence as an option merely because it contains
  the correct information.
- Evidence must remain a separate, exact quote from the passage.
- The correct option must preserve the meaning of the supplied correct answer.
- Do not make the correct option unnecessarily long just to match the passage word for word.
- Rewrite the supplied correct answer into a concise answer of 2–8 words.
- The correct option must directly answer the question, not reproduce
  a complete sentence from the passage.
- Do not copy the correct option verbatim into the evidence field.
- Evidence must be an exact quote from the passage, preferably a
  complete supporting clause or sentence.
- The evidence should provide context that supports the answer.
- All distractors must relate to the subject of the question and
  passage, but must be demonstrably incorrect.
- Do not introduce unrelated concepts that are absent from the passage.
- Avoid using a fragment from the passage as an answer when a concise,
  grammatically complete answer can be written.
- A distractor must not be a broader or narrower version of the correct answer (for example a
  region, country or category that contains the answer), and must not be something the passage
  also says is true for this question. Only one option may be a correct answer.
- Never create two options that express the same action or fact.
- Distractors must be clearly different from the correct answer.
- Do not change the subject of an action in a way that makes
  two options partially correct.
- Ensure the correct answer index points to the option that
  best answers the question.
- Generate exactly 4 DISTINCT answer options for every question.
- Exactly ONE option must be correct.
- The other 3 options must be plausible but incorrect based on the passage.
- NEVER copy the correct answer into multiple options.
- NEVER repeat an option or use minor wording changes that preserve the same meaning.
- Keep options concise and directly answer the question.
- If the supplied correct answer is a long sentence, shorten it while preserving its essential meaning.
- Make each distractor different from the correct answer and from the other distractors.
- Before returning the JSON, verify that all 4 options are distinct and only one answers the question correctly.
- If you cannot create 3 distinct, plausible distractors, reconsider the question and its answer.

QUESTION CHECK:
- Set "question_ok" to false if the supplied question and correct answer do not fit each other:
  the question asks for a different kind of thing than the answer (for example it asks for a name
  but the answer is a number), it reverses who does what to whom, more than one answer
  would be correct according to the passage, or it is not a natural, grammatical question in the
  language of the passage. Otherwise set it to true.
- Never reinterpret or reword the supplied question inside "q" or "why". Report the problem with
  "question_ok": false instead; "q" itself must always stay unchanged.
- When "question_ok" is false, write "better_question": one short, natural question in the language of
  the passage whose only correct answer is exactly the supplied correct answer. Use your own wording
  (do not copy more than 6 words in a row from the passage). When "question_ok" is true,
  set "better_question" to an empty string.

WHY:
- Write the explanation in {why_language}, the language of the passage.
- The explanation must match the question exactly as written.
- Use only {why_language} words in the explanation. Do not mix in words from another language
  (for example no English words such as "offer" in a Filipino explanation).
- Explain briefly why the correct answer answers the question.
- Use your own words rather than copying the evidence verbatim.
- Do not introduce facts that are unsupported by the passage.

EVIDENCE:
- Evidence MUST be copied exactly from the original PASSAGE.
- Evidence must be a continuous substring of the original passage.
- Select the complete sentence or shortest relevant excerpt that
  directly supports the correct answer.
- Prefer a complete supporting sentence when it provides useful context.
- Do not use the correct answer option as the evidence.
- Do not copy the answer option and merely add punctuation.
- Do not paraphrase, summarize, or invent evidence.
- Evidence and answer serve different purposes:
  - answer: a concise response selected from the four options.
  - evidence: the original passage text supporting that response.
- Before returning JSON, verify that every evidence value appears
  exactly in the original passage and supports its corresponding answer.
- If no supporting quote can be found, do not invent one.

OUTPUT:
- Return valid JSON only.
- Do not include Markdown fences or text outside the JSON.
- Keep the original question text and kind unchanged.

Required format:
{{
  "questions": [
    {{
      "kind": "Literal",
      "q": "Question text unchanged",
      "question_ok": true,
      "better_question": "",
      "correct_option": "Concise correct answer",
      "distractors": [
        "Incorrect option A",
        "Incorrect option B",
        "Incorrect option C"
      ],
      "why": "Brief explanation in your own words",
      "evidence": "Exact quote from the passage"
    }}
  ]
}}
"""

    raw_options = ask_sealion(
    "You are a careful quiz writer. Follow every quiz rule exactly. "
    "Return only valid JSON with concise options, a correct answer index, "
    "a matching explanation, and evidence copied from the passage.",
        options_prompt,
        temperature=0.2,
        max_tokens=1200,
        use_adapter=False,  # base model: this step must state the correct answer
    )

    lap("step 3, SEA-LION writes the options")
    try:
        options_data = parse_json_response(raw_options)
        option_items = options_data.get("questions", [])

    
    except json.JSONDecodeError as e:
        print("\n[Quiz] SEA-LION returned invalid JSON.")
        print(f"[Quiz] JSON error: {e}")
        print("[Quiz] Attempting to repair JSON...")

        repaired = re.sub(r",\s*([}\]])", r"\1", raw_options.strip())

        # Close missing JSON structures when the output is truncated.
        try:
            options_data = parse_json_response(repaired)
        except json.JSONDecodeError:
            repaired = repaired.rstrip()

            stack = []
            in_string = False
            escaped = False

            for char in repaired:
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char in "[{":
                        stack.append("]" if char == "[" else "}")
                    elif char in "]}":
                        if stack and stack[-1] == char:
                            stack.pop()

            if in_string:
                repaired += '"'

            repaired += "".join(reversed(stack))

            options_data = parse_json_response(repaired)

        option_items = options_data.get("questions", [])
        print("[Quiz] JSON repair succeeded.")

    except Exception as e:
        raise RuntimeError(
            f"Could not parse multiple-choice options from SEA-LION: {e}"
        )


    # STEP 4: Combine T5 questions with SEA-LION's supporting quiz fields.
    cleaned = []

    for idx, item in enumerate(generated):
        support = support_for(item, idx, option_items, len(generated))
        if support is None:
            print(f"[Quiz] No options returned for: {item['q']}")
            continue
        correct_text = item["correct_text"]

        flag = support.get("question_ok")
        print(f"[Quiz] SEA-LION question_ok={flag!r} for: {item['q']}")
        if str(flag).strip().lower() in ("false", "no", "0"):
            better = str(support.get("better_question", "")).strip()
            ok = bool(better) and _cmp(better) != _cmp(item["q"])
            ok = ok and not copies_passage(better, passage)
            ok = ok and not (lang == "en" and question_is_garbled(better))
            ok = ok and question_fits_answer(better, correct_text)
            ok = ok and answer_matches(better, correct_text, passage, lang)
            if not ok:
                print(f"[Quiz] Rejected (SEA-LION says the question does not fit its answer, "
                      f"and its rewrite {better!r} failed the checks): {item['q']}")
                continue
            print(f"[Quiz] Rewrote question: {item['q']} -> {better}")
            item = dict(item, q=better)

        correct_option = str(support.get("correct_option", "")).strip()
        distractors = support.get("distractors", [])

        # The keyed answer must still say what mT5 verified, otherwise the "why" and the
        # evidence may explain a different answer.
        if not option_matches_span(correct_option, correct_text):
            print(f"[Quiz] Rejected: option '{correct_option}' drifted from verified answer "
                  f"'{correct_text}' for: {item['q']}")
            continue

        if (
            not correct_option
            or not isinstance(distractors, list)
            or len(distractors) != 3
        ):
            print(f"[Quiz] Rejected malformed options for: {item['q']}")
            continue

        distractors = [str(option).strip() for option in distractors]

        if any(not option for option in distractors):
            print(f"[Quiz] Rejected empty option for: {item['q']}")
            continue

        distractors = clean_distractors(distractors, correct_option, correct_text)
        if len(distractors) != 3:
            print(f"[Quiz] Rejected: a distractor repeated or overlapped the answer for: {item['q']}")
            continue

        clash = distractor_in_answer_sentence(distractors, passage, correct_text) if SAME_SENTENCE_CHECK else None
        if clash:
            print(f"[Quiz] Rejected: distractor '{clash}' is in the same passage sentence as the answer "
                  f"'{correct_text}', so it may also be correct: {item['q']}")
            continue

        all_options = [correct_option] + distractors
        normalized_options = [
            " ".join(option.casefold().split())
            for option in all_options
        ]

        if len(set(normalized_options)) != 4:
            print(f"[Quiz] Rejected duplicate options for: {item['q']}")
            continue

        random.shuffle(all_options)

        options = all_options
        answer_index = options.index(correct_option)

        # Reject malformed option lists.
        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) for option in options)
            or not isinstance(answer_index, int)
            or isinstance(answer_index, bool)
            or not 0 <= answer_index < 4
        ):
            print(f"[Quiz] Rejected malformed options for: {item['q']}")
            print(f"[Quiz] Options received: {options}")
            continue

        options = [option.strip() for option in options]

        if any(not option for option in options):
            print(f"[Quiz] Rejected empty option for: {item['q']}")
            continue

        # Reject exact duplicate options, ignoring capitalization.
        normalized_options = [
            " ".join(option.casefold().split())
            for option in options
        ]

        if len(set(normalized_options)) != 4:
            print(f"[Quiz] Rejected duplicate options for: {item['q']}")
            print(f"[Quiz] Options received: {options}")
            continue

        # Normalize text for comparisons.
        def normalize_text(text):
            return " ".join(text.casefold().split())

        final_answer_index = answer_index
        correct_option = options[final_answer_index]

        # Reject answers that are too long to be concise.
        word_count = len(correct_option.split())

        if word_count > 12:
            print(
            f"[Quiz] Warning: Long correct option "
            f"({word_count} words): {item['q']}"
        )
        print(f"[Quiz] Correct option: {correct_option}")

        # Reject a correct option that simply repeats the evidence.
        evidence = str(support.get("evidence", "")).strip()
        why = str(support.get("why", "")).strip()

        if not evidence:
            print(f"[Quiz] Rejected missing evidence: {item['q']}")
            continue

        found = find_evidence(evidence, passage)
        if not found:
            print(
                f"[Quiz] Rejected evidence not found in passage: "
                f"{item['q']}"
            )
            print(f"[Quiz] Evidence: {evidence}")
            continue
        evidence = found   # the passage's own wording, not the model's copy of it

        if not why or normalize_text(why) == normalize_text(evidence):
            print(
                f"[Quiz] Rejected missing or repeated explanation: "
                f"{item['q']}"
            )
            continue

        # Final validation applies to EVERY question.
        if len(options) != 4 or any(not option for option in options):
            print(f"[Quiz] Rejected invalid final options: {item['q']}")
            continue

        normalized_options = [
            normalize_text(option) for option in options
        ]

        if len(set(normalized_options)) != 4:
            print(f"[Quiz] Rejected duplicate final options: {item['q']}")
            continue

        cleaned.append({
            "kind": item["kind"],
            "q": item["q"],
            "options": options,
            "answer": final_answer_index,
            "why": why,
            "evidence": evidence
        })

    print(f"[Quiz] Questions returned by SEA-LION: {len(option_items)}")
    print(f"[Quiz] Questions surviving validation: {len(cleaned)}")

    if not cleaned:
        print("[Quiz] DEBUG: No questions survived final validation.")
        print("[Quiz] Generated T5 questions:", generated)
        print("[Quiz] SEA-LION option data:", option_items)

        raise RuntimeError(
            "No valid quiz questions survived validation. "
            "Check the backend terminal for DEBUG output."
        )

    print(f"[Quiz][time] TOTAL: {time.perf_counter() - _clock[1]:.1f}s")
    return cleaned


# =====================================================================
# 3) CHECK-INS  (POST /checkin)
# =====================================================================
class CheckinAsk(BaseModel):
    goal: Optional[str] = None
    text: str = ""
    struggles: List[str] = []   # spots the student has already debugged
    previous: List[str] = []    # check-in questions already asked


CHECKIN_RULES = """You are a reading coach briefly interrupting a student in the middle of reading.
Write ONE short question (under 25 words) that makes them check their own understanding.
Pick one kind: summarize so far in one sentence; connect what they read to their goal; predict what comes next;
name the most confusing bit; explain a key idea in their own words.
Use details from the text so the question fits THIS reading. Never give the answer.
Do not repeat or closely copy a previous question. No greeting, no praise, no extra explanation. Plain words.
Reply with the question only."""


def generate_checkin(ask: CheckinAsk) -> str:
    goal = f"Student's reading goal: \"{ask.goal[:200]}\"\n" if ask.goal else ""
    prompt = f"{goal}Text context: {ask.text[:3000]}\nGenerate a single checking question."
    return ask_sealion(CHECKIN_RULES, prompt, history=None, temperature=0.8, max_tokens=60)


# =====================================================================
# The addresses the widget calls
# =====================================================================
@app.get("/")
def health():
    return {
        "status": "ok",
        "endpoints": ["/partner", "/quiz", "/checkin"],
    }

# After this many answers that did not show understanding, the coach explains outright.
GIVEUP_AFTER = int(os.getenv("GIVEUP_AFTER", "3"))

# "What do you mean?" and "huh?" are questions, but they say the student didn't understand.
_CONFUSED = re.compile(
    r"^\s*(huh|ha|what|ano|what do you mean|what does (that|it|this) mean|ano ang ibig mong sabihin|"
    r"i (still )?(don'?t|do not) (get|understand)( it)?|(hindi|di) ko (gets|maintindihan|naiintindihan|alam))"
    r"\s*[?!.]*\s*$", re.I)

_ACK = re.compile(
    r"^\s*(ok(ay)?|k|sige|oo|thanks?( you)?|thank u|ty|salamat( po)?|got it|i see|ah+|oh+|gets( ko)?)"
    r"\s*[.!]*\s*$", re.I)

def _is_answer(text):
    # Questions back to the coach and "in English please" are not attempts at the answer,
    # except a plain "I don't get it", which is the student saying they didn't understand.
    if not text or language_request(text):
        return False
    return not gr.is_question(text) or bool(_CONFUSED.match(text))

def should_give_up(ask):
    """Explain outright once the student has made GIVEUP_AFTER real attempts, once per thread."""
    if any(t.role == "coach" and t.mode == "giveup" for t in ask.history):
        return False
    if not _is_answer(ask.message):
        return False
    attempts = sum(1 for t in ask.history if t.role == "student" and _is_answer(t.text)) + 1
    return attempts >= GIVEUP_AFTER

def _fail(what: str, e: Exception):
    # The widget falls back to its offline demo if any of these fail.
    print(f"Error while {what}:", repr(e))
    raise HTTPException(status_code=502, detail=f"The AI could not do that ({what}). Check the server window for details.")

@app.post("/partner")
def partner(ask: Ask):
    try:
        requested = ask.mode
        if requested in ("stuck", "explain"):
            ask.mode = requested
        elif not ask.message:
            ask.mode = "debug"
        elif language_request(ask.message) and len(ask.message.split()) <= 5:
            ask.mode = "repeat"
        else:
            ask.mode = classify_intent(ask.message, target_passage=ask.passage,
                                       last_coach=last_coach_line(ask))
            # The backend decides when to explain outright; a "giveup" request from the widget
            # (older widgets send it on the second reply, whatever it says) is not trusted.
            if ask.mode in ("followup", "stuck") and should_give_up(ask):
                ask.mode = "giveup"
        # The third time the student is stuck (typed or the button), explain outright. This catches
        # "what do you mean?" turns, which should_give_up doesn't count as attempts.
        # Once the coach has explained outright, being stuck again gets another explanation,
        # not the hint ladder from the start.
        gave_up = any(t.role == "coach" and t.mode == "giveup" for t in ask.history)
        if ask.mode == "stuck" and (gave_up or stuck_level(ask) >= GIVEUP_AFTER):
            ask.mode = "giveup"

        g = run_grounding(ask)
        if ask.mode == "success" and g is not None and (g.label == gr.CONTRADICTED or g.missing_names):
            # The classifier called it understood, but the text says otherwise (R6) or the answer
            # names someone the text never mentions: don't confirm it or close the thread.
            print(f"[grounding] success overruled by {g.rule}")
            ask.mode = "followup"
        if gave_up and ask.mode == "followup" and _ACK.match(ask.message or ""):
            # "ok" after the explanation: close the turn instead of asking a fresh question.
            reply_text = _KEEP_READING[pick_language(ask)]
        elif g is not None and g.label == gr.NOT_IN_TEXT:
            reply_text = gr.fallback_reply(g, pick_language(ask))  # R1: fixed template, no model call
            if ask.mode in ("stuck", "giveup") and requested != "stuck":
                ask.mode = "followup"  # asking about a name is not a rung on the hint ladder
        else:
            reply_text = generate_hint(ask, g)

        print("\n" + "-" * 40)
        stuck_info = f" (level {stuck_level(ask)})" if ask.mode == "stuck" else ""
        print(f"[API REQUEST] /partner | Mode: {ask.mode}{stuck_info} | Problem: {ask.problem}")
        print(f"   Target: {ask.passage} | Student said: {ask.message!r} | History turns: {len(ask.history)}"
              f" | Language: {pick_language(ask)} (setting: {ask.response_language!r})")
        print(f"[API RESPONSE]: {reply_text}")
        print("-" * 40)
        response = {"reply": reply_text, "mode": ask.mode}
        if g is not None:
            response["grounding"] = {"label": g.label, "rule": g.rule, "evidence_ids": g.evidence_ids}
        return response
    except Exception as e:
        print("Error in /partner:", repr(e))
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/quiz")
def quiz(ask: QuizAsk):
    try:
        return {"questions": generate_quiz(ask)}
    except Exception as e:
        _fail("making the quiz", e)


@app.post("/checkin")
def checkin(ask: CheckinAsk):
    try:
        return {"prompt": generate_checkin(ask)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
    print(model.device)

    import uvicorn
    print("\nStarting FastAPI server for HTML widget on http://localhost:8000 ...\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)