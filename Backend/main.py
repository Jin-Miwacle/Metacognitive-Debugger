"""
Reading Partner backend (temporary brain: Gemini).

The widget (index.html) talks to three addresses on this server:

    POST /partner   ->  { "reply": "..." }        a hint while the student is debugging
    POST /quiz      ->  { "questions": [ ... ] }  a quiz written from the student's text
    POST /checkin   ->  { "prompt": "..." }       a short mid-reading check-in question

Only the functions marked "THE AI CALL" talk to Gemini. When you switch to your
own fine-tuned model later, those are the only places you change.
"""
import json
import os
import random
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()  # reads the .env file (your secret key lives there)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# If you get a "model not found" error, open Google AI Studio, copy the name
# of a current "Flash" model, and put it in your .env file as GEMINI_MODEL.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = None
types = None
if GEMINI_API_KEY:
    from google import genai
    from google.genai import types  # noqa: F811

    client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Reading Partner backend")

# Lets the widget (a different address) talk to this backend.
# For the real launch, replace "*" with your real website address.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def ask_gemini(system_rules: str, prompt: str, *, temperature: float, max_tokens: int, schema=None):
    """THE AI CALL. Every feature goes through this one function."""
    if client is None:
        raise RuntimeError("No Gemini key found. Check your .env file.")
    config = dict(
        system_instruction=system_rules,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config),
    )


# =====================================================================
# 1) HINTS  (POST /partner)
# =====================================================================
class Turn(BaseModel):
    role: str   # "student" or "coach"
    text: str


class Ask(BaseModel):
    mode: str = "debug"            # "debug" (first hint), "followup", or "stuck"
    problem: Optional[str] = None  # "vocab", "lost", "connect", or "why"
    passage: str = ""              # the words the student selected
    paragraph: str = ""            # the paragraph around it
    goal: Optional[str] = None     # the student's reading goal
    message: Optional[str] = None  # the student's latest reply
    history: List[Turn] = []       # the conversation so far about this spot


# The coach's personality and rules. Improving this text is the fastest way
# to improve the hints.
COACH_RULES = """You are a warm, sharp reading coach talking with one student about one confusing spot in a text.
Your job is to help them notice HOW they are reading, not to read the text for them.

Always:
- Never give the full answer, and never paraphrase the whole passage for them.
- Use ONE move per reply: a guiding question, a strategy to try, a clue to look for, or one tiny first step.
- Use the student's actual words from the passage, so your reply could not be pasted onto a different sentence.
- Plain, friendly words. Under 60 words. No more than two questions.
- Do not start two replies the same way. Look at what the coach already said in the conversation and do something different.

Praise rules:
- NEVER say "Good attempt", "Great question", "Good job", "Great start", "Nice try" or similar stock praise.
- Do not praise or thank the student for clicking a button or asking for help.
- Only comment on the student's work if they actually wrote an explanation, and then react to the SPECIFIC thing they said
  (what is right, or what is missing), not with general praise.
"""

MODE_TEXT = {
    "debug": ("This is the FIRST message. The student has not written anything yet, so do not praise or thank them. "
              "Go straight to your one move."),
    "followup": ("The student just replied. Read what they wrote. If it is basically right, confirm it in a few words "
                 "and invite them to mark it clear. If it is partly right, name the part that works and ask about the "
                 "missing piece. If it is off, or they say 'idk', do not tell them they are wrong; shrink the task into "
                 "one smaller question."),
    "stuck": ("The student tried and is STILL stuck. Break the passage into two tiny steps they can do. You may explain "
              "the meaning of a hard word, but still do not give the full answer."),
}

# (what is going wrong, ideas the coach may choose ONE of)
PROBLEMS = {
    "vocab": ("The student does not know a word or phrase.",
              "use clues from nearby sentences; skip the word and reread; swap in a simpler word that fits; look at word parts"),
    "lost": ("The student read it but lost the main point.",
             "say it back in ten words; ask who did what; cover it and retell it; split a long sentence at the commas"),
    "connect": ("The student cannot see how this connects to the sentences before it.",
                "compare with the sentence before; guess the link word (because, but, so, then); ask what changed between the two"),
    "why": ("The student cannot tell why something happened; it is implied, not stated.",
            "separate what the text says from what the student guesses; hunt for an earlier clue; predict, then check"),
}


def build_hint_prompt(ask: Ask) -> str:
    what, ideas = PROBLEMS.get(ask.problem or "", ("The student is confused.", "any helpful reading strategy"))
    lines = []
    if ask.goal:
        lines.append(f"The student's reading goal: {ask.goal}")
    if ask.paragraph:
        lines.append(f"The paragraph: {ask.paragraph[:2000]}")
    lines.append(f"The part the student is confused by: {ask.passage[:600]}")
    lines.append(f"What is going wrong: {what}")
    lines.append(f"Ideas you may pick ONE from (vary them): {ideas}")

    if ask.history:
        lines.append("\nConversation so far:")
        for turn in ask.history[-8:]:
            who = "Student" if turn.role == "student" else "Coach"
            lines.append(f"{who}: {turn.text[:600]}")
    if ask.message:
        lines.append(f"\nThe student's latest message: {ask.message[:600]}")

    lines.append("\n" + MODE_TEXT.get(ask.mode, MODE_TEXT["debug"]))
    lines.append("Write only what the coach says next.")
    return "\n".join(lines)


def generate_hint(ask: Ask) -> str:
    response = ask_gemini(COACH_RULES, build_hint_prompt(ask), temperature=0.9, max_tokens=800)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty reply.")
    return text


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
    n = max(2, min(ask.n, 8))
    lines = [f"Write {n} questions."]
    if ask.goal:
        lines.append(f"The student's reading goal was: {ask.goal}. Make at least one question serve that goal if the text allows.")
    lines.append("\nTEXT:\n" + ask.text[:8000])
    response = ask_gemini(QUIZ_RULES, "\n".join(lines), temperature=0.5, max_tokens=4000, schema=Quiz)

    quiz = response.parsed
    if not isinstance(quiz, Quiz):
        quiz = Quiz.model_validate_json(response.text or "{}")

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
    lines = []
    if ask.goal:
        lines.append(f"The student's reading goal: {ask.goal}")
    if ask.struggles:
        lines.append("Spots they found confusing: " + " | ".join(s[:150] for s in ask.struggles[-5:]))
    if ask.previous:
        lines.append("Questions already asked (do not repeat): " + " | ".join(p[:150] for p in ask.previous[-5:]))
    lines.append("\nTEXT:\n" + ask.text[:6000])
    response = ask_gemini(CHECKIN_RULES, "\n".join(lines), temperature=0.9, max_tokens=600)
    text = (response.text or "").strip().strip('"')
    if not text:
        raise RuntimeError("Gemini returned an empty reply.")
    return text


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
        return {"reply": generate_hint(ask)}
    except Exception as e:
        _fail("writing a hint", e)


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
        _fail("writing a check-in", e)
