"""
Reading Partner backend.

The widget (index.html) talks to three addresses on this server:
    POST /partner   ->   { "reply": "..." }       a hint while the student is debugging
    POST /quiz      ->   { "questions": [ ... ] } a quiz written from the student's text
    POST /checkin   ->   { "prompt": "..." }      a short mid-reading check-in question
"""
import json
import os
import random
from typing import List, Optional
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = os.getenv("SEALLMS_MODEL", "seallms_v3_1_5b_chat_finetuned")

print(f"Loading local model {MODEL_NAME} (this may take a moment)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map={"": 0},
    low_cpu_mem_usage=True
)

model.generation_config.max_length = None
print("Model loaded successfully!")

app = FastAPI(title="Reading Partner backend")
app = FastAPI(title="Reading Partner backend (SeaLLMs v3 1.5B Chat)")

# Lets the widget (a different address) talk to this backend.
# For the real launch, replace "*" with your real website address.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def ask_llama(system_rules: str, prompt: str, history: Optional[List['Turn']] = None, *, temperature: float, max_tokens: int) -> str:
    messages = [{"role": "system", "content": system_rules}]
    
    # Inject conversation history natively using proper roles
    if history:
        for turn in history:
            role = "assistant" if turn.role == "coach" else "user"
            messages.append({"role": role, "content": turn.text})
            
    # Add current prompt as final user message
    messages.append({"role": "user", "content": prompt})
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = f"<|im_start|>system\n{system_rules}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    
    do_sample = temperature > 0.0
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_ids = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response_text.strip()


# =====================================================================
# 1) HINTS  (POST /partner)
# =====================================================================
class Turn(BaseModel):
    role: str   # "student" or "coach"
    text: str


class Ask(BaseModel):
    mode: str = "debug"            # "debug" (first hint), "followup", or "stuck"
    problem: Optional[str] = None  # "vocab", "lost", "connect", or "why"
    passage: str = ""              # words selected
    paragraph: str = ""            # paragraph around it
    goal: Optional[str] = None     # reading goal
    message: Optional[str] = None  # student latest reply
    history: List[Turn] = []       # conversation history

COACH_RULES = """You are a warm, encouraging reading coach for Grade 7 students. 

CRITICAL BEHAVIORAL MODES (Follow these strictly):
1. CHECK THE TARGET PASSAGE: If the student's reply matches or contains the target word/passage, they are RIGHT. You MUST validate them briefly in your own words, confirm they've got it, and tell them they can move on. DO NOT ask another question.
2. IF THE STUDENT SAYS "I don't know", "hindi ko alam", "just tell me", or asks what something means: STOP asking questions. Give a clear, direct, simple explanation immediately.
3. OTHERWISE (Socratic Mode): Ask ONE simple guiding question to help them think.

General Rules:
- OUTPUT ONLY YOUR SPOKEN RESPONSE AS THE COACH. Do NOT write "Coach:", "Student:", or simulate any dialogue back-other.
- LANGUAGE RULE: Match the language of the target passage or student message. If the word or text is in Filipino/Tagalog (e.g., 'handaan'), your entire response MUST be in Filipino or natural Taglish. Never reply in English to a Filipino input.
- Keep your response under 45 words.
- Never use stock praise like "Good job."
- Never invent facts, characters, or plot points outside of the provided context.
- Use plain, easy-to-understand words."""

CLASSIFIER_RULES = """You are an intent classifier for a reading coach. 
Classify the student's latest message into EXACTLY one of these three words:
- "success": The student understands, got it right, or says they get it/understand.
- "stuck": The student gives up, says they don't know, asks to be told directly, or is completely lost.
- "followup": The student is answering normally, asking another question, or continuing the conversation.

Output ONLY the exact category word. No punctuation, no explanation."""

def classify_intent(message: str, target_passage: str = "") -> str:
    if not message:
        return "followup"
    
    prompt = f"Target word/passage: \"{target_passage}\"\nStudent message: \"{message}\"\nCategory:"
    raw_output = ask_llama(CLASSIFIER_RULES, prompt, history=None, temperature=0.1, max_tokens=5)
    
    cleaned = raw_output.strip().lower()
    if "success" in cleaned:
        return "success"
    if "stuck" in cleaned:
        return "stuck"
    return "followup"

def build_hint_prompt(ask: Ask) -> str:
    passage = ask.passage or "this part"
    problem_key = ask.problem or "lost"
    
    lines = []
    if ask.goal:
        lines.append(f"Reading goal: {ask.goal}")

    if ask.paragraph:
        lines.append(f"Context paragraph: {ask.paragraph[:600]}")
    else:
        lines.append(f"Context: None provided. Treat '{passage}' as a general vocabulary term without inventing external characters or plot points.")
        
    lines.append(f"The student is stuck on: \"{passage}\".")
    
    if problem_key == "vocab":
        lines.append(f"Instruction: Ask the student ONE simple question to help them figure out what \"{passage}\" means using context or everyday experience.")
    elif problem_key == "lost":
        lines.append(f"Instruction: Ask the student ONE simple question about what is happening here.")
    elif problem_key == "connect":
        lines.append(f"Instruction: Ask the student ONE simple question pointing back to the sentence right before it.")
    elif problem_key == "why":
        lines.append(f"Instruction: Ask the student ONE simple question to look for an earlier clue in the text.")
    else:
        lines.append(f"Instruction: Ask the student ONE simple guiding question.")
        
    if ask.message:
        lines.append(f"\nStudent's latest reply: \"{ask.message[:300]}\"")
        
    lines.append("\nOutput ONLY your single spoken response as the coach. Do not write any labels, lists, or explanations.")
    return "\n".join(lines)


def generate_hint(ask: Ask) -> str:
    prompt = build_hint_prompt(ask)
    raw_response = ask_llama(COACH_RULES, prompt, history=ask.history, temperature=0.7, max_tokens=150)
    
    cleaned_response = re.sub(r'^(Coach|Assistant|AI|Reading Partner)\s*:\s*', '', raw_response, flags=re.IGNORECASE)
    return cleaned_response.strip()


# =====================================================================
# 2) QUIZ  (POST /quiz)  - the question-generation module
# =====================================================================
class QuizAsk(BaseModel):
    text: str
    goal: Optional[str] = None
    n: int = 4


class QuizQuestion(BaseModel):
    kind: str            # "Literal" or "Inference"
    q: str
    options: List[str]
    answer: int          # 0-based index of the correct option
    why: str
    evidence: str        # a short phrase from the text that supports the answer


class Quiz(BaseModel):
    questions: List[QuizQuestion]


QUIZ_RULES = """You write reading-comprehension quiz questions for students, using ONLY the text you are given.
Rules:
- Write exactly the number of multiple-choice questions you are asked for.
- Mix the types: about half "Literal" (the answer is stated in the text) and about half "Inference" (the answer must be worked out from clues in the text).
- Exactly 4 options per question, and only one clearly correct.
- Wrong options must be believable: common misreadings, details from the text used the wrong way, or things that are true in general but not supported by this text. Never silly or obviously wrong.
- Keep the correct option about the same length as the others.
- Do not ask about facts that are not in the text. Never use "all of the above" or "none of the above".
- "answer" is the 0-based index of the correct option.
- "why" is one or two simple sentences explaining how the text supports the answer. For inference questions, name the clue. Do not refer to options by letter or number.
- "evidence" is a short phrase copied exactly from the text that supports the answer.
- Use clear, simple language.
"""


def generate_quiz(ask: QuizAsk) -> list:
    n = max(2, min(ask.n, 5))
    prompt = f"Create {n} reading comprehension questions based on this text:\n\n{ask.text[:4000]}\n\nReturn strictly valid JSON."
    raw_response = ask_llama(QUIZ_RULES, prompt, history=None, temperature=0.3, max_tokens=1024)
    
    try:
        clean_json = raw_response
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
        data = json.loads(clean_json)
        questions = data.get("questions", [])
    except Exception as e:
        print("JSON Parse Error:", e, "Raw output was:", raw_response)
        raise RuntimeError("Model failed to output clean JSON for the quiz.")
        
    cleaned = []
    for q in quiz.questions:
        if len(q.options) != 4 or len(set(q.options)) != 4 or not 0 <= q.answer < 4:
            continue  # skip badly formed questions
        correct = q.options[q.answer]
        options = q.options[:]
        random.shuffle(options)  # models like to put the right answer in the same spot
        cleaned.append({
            "kind": q.kind if q.kind in ("Literal", "Inference") else "Question",
            "q": q.q,
            "options": options,
            "answer": options.index(correct),
            "why": q.why,
            "evidence": q.evidence,
        })
    if not cleaned:
        raise RuntimeError("Gemini did not return any usable questions.")
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
    prompt = f"Text context: {ask.text[:3000]}\nGenerate a single checking question."
    return ask_llama(CHECKIN_RULES, prompt, history=None, temperature=0.8, max_tokens=60)


# =====================================================================
# The addresses the widget calls
# =====================================================================
@app.get("/")
def health():
    """Open http://localhost:8000 in a browser to check the server is running."""
    return {
        "status": "ok",
        "gemini_connected": client is not None,
        "model": GEMINI_MODEL if client else None,
        "endpoints": ["/partner", "/quiz", "/checkin"],
    }


def _fail(what: str, e: Exception):
    # The widget falls back to its offline demo if any of these fail.
    print(f"Error while {what}:", repr(e))
    raise HTTPException(status_code=502, detail=f"The AI could not do that ({what}). Check the server window for details.")


@app.post("/partner")
def partner(ask: Ask):
    try:
        if ask.message:
            ask.mode = classify_intent(ask.message, target_passage=ask.passage)
        else:
            ask.mode = "debug"
            
        reply_text = generate_hint(ask)
        
        print("\n" + "-"*40)
        print(f"📥 [API REQUEST] /partner | Mode: {ask.mode} | Problem: {ask.problem}")
        print(f"   Passage/Message: {ask.passage or ask.message}")
        print(f"📤 [API RESPONSE]: {reply_text}")
        print("-" * 40)
        
        return {"reply": reply_text}
    except Exception as e:
        print("❌ Error in /partner:", repr(e))
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