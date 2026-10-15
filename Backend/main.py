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

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

model.generation_config.max_length = None
print("SEALION Model loaded successfully!")

# Load T5 Model
T5_MODEL_NAME = "Aiken1016/MT5_Final_Model"

print(f"Loading T5 question-generation model: {T5_MODEL_NAME}...")

# The model was saved with transformers 5.x, which writes extra_special_tokens as a list;
# 4.x expects a dict and crashes on load. Passing an empty dict works on both.
qg_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME, extra_special_tokens={})
# mT5 has its own output layer (lm_head), separate from the input embeddings, and the
# checkpoint stores both. Its config says tie_word_embeddings=true (written by 5.x), which
# makes 4.x drop the trained lm_head and reuse the embeddings, so every output is garbage
# ("01.01.201 ...") that never ends. Force them untied.
qg_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME, tie_word_embeddings=False)

# CPU by default: the mT5 is 2.3 GB in fp32, and next to the 4-bit 9B model (~6.5 GB)
# it would not fit in 8 GB of VRAM. Set QG_DEVICE=cuda to try the GPU anyway.
qg_device = torch.device(os.getenv("QG_DEVICE", "cpu"))
qg_model.to(qg_device)
qg_model.eval()

print("T5 question-generation model loaded successfully!")

app = FastAPI(title="Reading Partner backend")

# Lets the widget (a different address) talk to this backend. For the hosted site, set
# ALLOWED_ORIGINS to its address(es), comma-separated, e.g.
#   $env:ALLOWED_ORIGINS = "https://jin-miwacle.github.io"
# Unset = allow any origin (fine for local testing).
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
        print(f"[ask_llama] chat template supports system role: {_SYSTEM_SUPPORTED}")
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

def ask_llama(system_rules: str, prompt: str, history: Optional[List["Turn"]] = None, *,
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
    mode: str = "debug"            # from the widget: "debug", "followup", "stuck" (button) or "giveup"
    problem: Optional[str] = None  # "vocab", "lost", "connect", "why", or "other" (typed by the student)
    problem_text: Optional[str] = None  # the student's own description, set when problem == "other"
    passage: str = ""              # words selected
    paragraph: str = ""            # paragraph around it
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
    """'en' / 'fil' only if the message is JUST a language request ("in english please",
    "sa tagalog po"). An answer that mentions a language ("dusk sa tagalog ay takipsilim")
    is not a request, so it does not switch the language or become a "repeat" turn."""
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
    """Map whatever the frontend sends ("en", "English", "tl", "auto", ...) to "en", "fil" or None."""
    if not value:
        return None
    return _LANG_ALIASES.get(str(value).strip().lower())

def pick_language(ask):
    # 1. A request typed in this thread wins ("in english please"). History is per debug
    #    thread, so the override lasts for this passage and the saved setting returns after.
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
    r"Instruction for this turn:|Student's latest reply:|Language for your reply:|Reading goal( \(the student's own words\))?:)",
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
    
    raw_output = ask_llama(CLASSIFIER_RULES, prompt, history=None, temperature=0.1, max_tokens=5, use_adapter=False)
    cleaned = raw_output.strip().lower()

    if "success" in cleaned:
        return "success"
    if "stuck" in cleaned:
        return "stuck"
    return "followup"

_QUESTION_BY_PROBLEM = {
    "vocab":   "ONE simple question to help them figure out what \"{p}\" means using context or everyday experience.",
    "lost":    "ONE simple question about what is happening here.",
    "connect": "ONE simple question pointing back to the sentence right before it.",
    "why":     "ONE simple question to look for an earlier clue in the text.",
}

def _full_explanation(p):
    return (f"The student has been stuck several times. Now explain \"{p}\" clearly and simply using "
            f"the context, with one everyday example. End by asking them to say it in their own words.")

def goal_instruction(ask):
    """How strongly this turn should point at the student's reading goal.

    Only when the selection shares a word with the goal (goal_hint) is the link stated
    outright; otherwise the coach may mention the goal only if the link is real, so it
    does not force a connection onto an unrelated word."""
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
    """The student's own description of the trouble, or None (only for problem "other")."""
    if ask.problem == "other" and ask.problem_text and ask.problem_text.strip():
        return ask.problem_text.strip()[:200]
    return None

def mode_instruction(ask):
    p = ask.passage or "this part"
    said = problem_said(ask)
    if said:
        # An f-string, not .format(): the text comes from a student, and a "{" in it
        # would make .format() raise and turn the request into a 502.
        question = (f"ONE short, simple question aimed at exactly the trouble they described "
                    f"(\"{said}\"), not at some other difficulty you notice in the passage.")
    else:
        question = _QUESTION_BY_PROBLEM.get(ask.problem or "lost", "ONE simple guiding question.").format(p=p)
    text = _mode_instruction(ask, p, question)
    # Stuck and give-up turns write their own instruction and never use `question`,
    # so carry the description into them explicitly.
    if said and ask.mode in ("stuck", "giveup"):
        text += f" They had described the trouble as: \"{said}\"."
    return text + goal_instruction(ask)

def _mode_instruction(ask, p, question):
    if ask.mode == "success":
        return (f"The student has understood \"{p}\". Confirm they are right in your own words, "
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
        return _full_explanation(p)
    if ask.mode == "repeat":
        return "The student asked you to change language. Repeat your last question in that language. Nothing else."
    if ask.mode == "followup":
        return (f"Respond briefly to what the student just said (don't say whether it's right yet), "
                f"then ask {question}")
    return f"Ask the student {question}"
 
def build_hint_prompt(ask, lang):
    passage = ask.passage or "this part"
    lines = []
    if ask.goal:
        lines.append(f"Reading goal (the student's own words): \"{ask.goal[:200]}\"")
    if ask.paragraph:
        lines.append(f"Context paragraph: {ask.paragraph[:600]}")
    else:
        lines.append(f"Context: None provided. Treat '{passage}' as a general vocabulary term "
                     f"without inventing external characters or plot points.")
    lines.append(f"The student is working on: \"{passage}\".")
    if ask.message:
        lines.append(f"Student's latest reply: \"{ask.message[:300]}\"")
    lines.append(f"Language for your reply: {'English' if lang == 'en' else 'Filipino (natural Taglish is OK)'}")
    lines.append(f"\nInstruction for this turn: {mode_instruction(ask)}")
    lines.append("\nOutput ONLY your single spoken response as the coach.")
    return "\n".join(lines)
 
SUCCESS_FALLBACK = {
    "en":  "That's right! You've figured out what \"{p}\" means. You're ready to move on to more passages.",
    "fil": "Tama! Nakuha mo na ang ibig sabihin ng \"{p}\". Handa ka nang magpatuloy sa susunod na babasahin.",
}
 
def generate_hint(ask):
    lang = pick_language(ask)
    prompt = build_hint_prompt(ask, lang)
    temperature = 0.3 if ask.mode in ("success", "repeat") else 0.7
    raw_response = ask_llama(COACH_RULES, prompt, history=ask.history, temperature=temperature, max_tokens=150)
    reply = clean_reply(raw_response)
 
    if ask.mode == "success" and "?" in reply:
        reply = SUCCESS_FALLBACK[lang].format(p=ask.passage or "this part")
    if not reply:
        reply = ("Let's look at that part of the text again. What do you notice?" if lang == "en"
                 else "Balikan natin ang bahaging iyon ng teksto. Ano ang napapansin mo?")
    return reply


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

    def parse_json_response(raw: str) -> dict:
        clean = raw.strip()

        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

        return json.loads(clean)

    # STEP 1: Use Llama to identify answer phrases in the passage.
    # T5 requires an answer phrase to be highlighted in the context.
    answer_prompt = f"""
Read the passage and select exactly {n} DISTINCT answer targets for a
reading-comprehension quiz.

QUESTION MIX:
- If n = 2, select 1 literal target and 1 inferential target.
- If n >= 3, select approximately half literal and half inferential targets.
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
- Copy each answer target exactly from the passage.
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

Return exactly {n} items if the passage supports that many distinct targets.

Passage:
{passage}
"""

    raw_answers = ask_llama(
        "Select answer phrases from the supplied passage. Return valid JSON only.",
        answer_prompt,
        temperature=0.2,
        max_tokens=500,
        use_adapter=False,  # the Socratic adapter is trained to avoid stating answers
    )

    try:
        answer_data = parse_json_response(raw_answers)
        answer_items = answer_data.get("items", [])
    except Exception as e:
        raise RuntimeError(
            f"Could not parse answer phrases from Llama: {e}"
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

        # Find the exact phrase in the original passage, ignoring case.
        start = passage.lower().find(answer.lower())
        if start == -1:
            continue

        exact_answer = passage[start:start + len(answer)]

        if exact_answer.lower() in used_answers:
            continue

        prompt = f"generate question: answer: {exact_answer} context: {passage}"

        inputs = qg_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )
        inputs = {key: value.to(qg_device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = qg_model.generate(
                **inputs,
                max_new_tokens=48,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        question = qg_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if not question or question.lower() in used_questions:
            continue

        used_answers.add(exact_answer.lower())
        used_questions.add(question.lower())

        generated.append({
            "kind": kind,
            "q": question,
            "correct_text": exact_answer
        })

    if not generated:
        raise RuntimeError(
            "T5 could not generate questions from the selected answer phrases."
        )

    print(f"[Quiz] Requested questions: {n}")
    print(f"[Quiz] Answer targets returned: {len(answer_items)}")
    print(f"[Quiz] T5 questions generated: {len(generated)}")
    print(f"[Quiz] T5 questions: {generated}")

    # STEP 3: Use Llama to create the answer options and supporting details.
    # Keep the question wording generated by T5 unchanged.
    questions_for_llama = [
        {
            "kind": item["kind"],
            "q": item["q"],
            "correct_answer": item["correct_text"]
        }
        for item in generated
    ]

    
    options_prompt = f"""
Create {len(questions_for_llama)} reading-comprehension multiple-choice questions.

PASSAGE:
{passage}

QUESTIONS AND CORRECT ANSWERS:
{json.dumps(questions_for_llama, ensure_ascii=False, indent=2)}

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

WHY:
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

Passage:
{passage}

Questions and correct answers:
{json.dumps(questions_for_llama, ensure_ascii=False)}
"""

    raw_options = ask_llama(
    "You are a careful quiz writer. Follow every quiz rule exactly. "
    "Return only valid JSON with concise options, a correct answer index, "
    "a matching explanation, and evidence copied from the passage.",
        options_prompt,
        temperature=0.2,
        max_tokens=1200,
        use_adapter=False,  # base model: this step must state the correct answer
    )

    try:
        options_data = parse_json_response(raw_options)
        option_items = options_data.get("questions", [])

    
    except json.JSONDecodeError as e:
        print("\n[Quiz] Llama returned invalid JSON.")
        print(f"[Quiz] JSON error: {e}")
        print("[Quiz] Attempting to repair JSON...")

        import re

        repaired = re.sub(r",\s*([}\]])", r"\1", raw_options.strip())

        # Close missing JSON structures when the output is truncated.
        try:
            options_data = parse_json_response(repaired)
        except json.JSONDecodeError:
            # Remove an incomplete trailing comma, if present.
            repaired = repaired.rstrip()

            # Add missing closing brackets/braces based on open structures.
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
            f"Could not parse multiple-choice options from Llama: {e}"
        )


    # STEP 4: Combine T5 questions with Llama's supporting quiz fields.
    cleaned = []

    for item, support in zip(generated, option_items):
        correct_text = item["correct_text"]
        correct_option = str(support.get("correct_option", "")).strip()
        distractors = support.get("distractors", [])

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

        all_options = [correct_option] + distractors
        normalized_options = [
            " ".join(option.casefold().split())
            for option in all_options
        ]

        if len(set(normalized_options)) != 4:
            print(f"[Quiz] Rejected duplicate options for: {item['q']}")
            continue

        import random
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

        # Trust Llama's selected answer index instead of replacing
        # its concise answer with the original passage answer.
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

    print(f"[Quiz] Questions returned by Llama: {len(option_items)}")
    print(f"[Quiz] Questions surviving validation: {len(cleaned)}")

    if not cleaned:
        print("[Quiz] DEBUG: No questions survived final validation.")
        print("[Quiz] Generated T5 questions:", generated)
        print("[Quiz] Llama option data:", option_items)

        raise RuntimeError(
            "No valid quiz questions survived validation. "
            "Check the backend terminal for DEBUG output."
        )

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
    return ask_llama(CHECKIN_RULES, prompt, history=None, temperature=0.8, max_tokens=60)


# =====================================================================
# The addresses the widget calls
# =====================================================================
@app.get("/")
def health():
    """Open http://localhost:8000 in a browser to check the server is running."""
    return {
        "status": "ok",
        "endpoints": ["/partner", "/quiz", "/checkin"],
    }

def _fail(what: str, e: Exception):
    # The widget falls back to its offline demo if any of these fail.
    print(f"Error while {what}:", repr(e))
    raise HTTPException(status_code=502, detail=f"The AI could not do that ({what}). Check the server window for details.")

@app.post("/partner")
def partner(ask: Ask):
    try:
        requested = ask.mode
        if requested == "stuck":
            ask.mode = "stuck"    # the "Still stuck" button: an explicit signal, sent with no message
        elif not ask.message:
            ask.mode = "debug"
        elif language_request(ask.message) and len(ask.message.split()) <= 5:
            ask.mode = "repeat"   # "in english please" is not an answer
        else:
            ask.mode = classify_intent(ask.message, target_passage=ask.passage,
                                       last_coach=last_coach_line(ask))
            # The widget sends "giveup" after a few unresolved follow-ups. Explain fully
            # then, unless the student just showed they understood.
            if requested == "giveup" and ask.mode != "success":
                ask.mode = "giveup"
 
        reply_text = generate_hint(ask)
 
        print("\n" + "-" * 40)
        stuck_info = f" (level {stuck_level(ask)})" if ask.mode == "stuck" else ""
        print(f"[API REQUEST] /partner | Mode: {ask.mode}{stuck_info} | Problem: {ask.problem}")
        print(f"   Target: {ask.passage} | Student said: {ask.message!r} | History turns: {len(ask.history)}"
              f" | Language: {pick_language(ask)} (setting: {ask.response_language!r})")
        print(f"[API RESPONSE]: {reply_text}")
        print("-" * 40)
        return {"reply": reply_text, "mode": ask.mode}
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