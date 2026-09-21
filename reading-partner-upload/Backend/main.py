"""
Reading Partner backend (temporary brain: Gemini).

The widget (index.html) sends a request to  POST /partner
and this program asks Gemini for a hint, then sends the reply back.

Only ONE function talks to the AI: generate_reply().
When you switch to your own fine-tuned model later, that is the only
function you need to change. The widget stays exactly the same.
"""
import os
from typing import Optional

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


class Ask(BaseModel):
    mode: str = "debug"            # "debug", "followup", or "stuck"
    problem: Optional[str] = None  # "vocab", "lost", "connect", or "why"
    passage: str = ""              # the words the student selected
    paragraph: str = ""            # the paragraph around it
    goal: Optional[str] = None     # the student's reading goal
    message: Optional[str] = None  # the student's reply (follow-ups)


# ---- The coach's rules. This is the "scaffolding logic": improve it over time. ----
COACH_RULES = """You are a reading coach helping a student who is confused by part of a text.
Rules:
- Do NOT give the answer or explain the whole passage for them.
- Give one small strategy or ask one or two guiding questions.
- Use simple, friendly words.
- Keep your reply under 60 words.
- If the student wrote an explanation, say what is good about it first, then help them improve it.
"""

PROBLEM_TEXT = {
    "vocab": "The student does not know a word or phrase.",
    "lost": "The student read it but lost the main point.",
    "connect": "The student cannot see how this connects to the sentences before it.",
    "why": "The student cannot tell why something happened; it is implied, not stated.",
}


def build_prompt(ask: Ask) -> str:
    lines = []
    if ask.goal:
        lines.append(f"The student's reading goal: {ask.goal}")
    if ask.paragraph:
        lines.append(f"The paragraph: {ask.paragraph[:2000]}")
    lines.append(f"The part the student is confused by: {ask.passage[:600]}")
    lines.append("What is going wrong: " + PROBLEM_TEXT.get(ask.problem or "", "The student is confused."))

    if ask.mode == "followup" and ask.message:
        lines.append(f"The student just wrote: {ask.message[:600]}")
        lines.append("Respond to what they wrote.")
    elif ask.mode == "stuck":
        lines.append("The student is STILL stuck after trying. Break it into two tiny steps they can do. "
                     "You may explain the meaning of a hard word, but still do not give the full answer.")
    else:
        lines.append("Write your first hint.")
    return "\n".join(lines)


def generate_reply(ask: Ask) -> str:
    """The ONLY function that talks to the AI. Swap this out later."""
    if client is None:
        # Demo mode: no key set, so we return a clearly labeled fake reply.
        return ("(Demo mode: no Gemini key found.) Reread the part you selected once, slowly, "
                "then tell me in your own words what it says.")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(ask),
        config=types.GenerateContentConfig(
            system_instruction=COACH_RULES,
            temperature=0.7,
            max_output_tokens=800,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty reply.")
    return text


@app.get("/")
def health():
    """Open http://localhost:8000 in a browser to check the server is running."""
    return {
        "status": "ok",
        "gemini_connected": client is not None,
        "model": GEMINI_MODEL if client else None,
    }


@app.post("/partner")
def partner(ask: Ask):
    try:
        return {"reply": generate_reply(ask)}
    except Exception as e:  # the widget falls back to its demo replies if this fails
        print("Error while asking Gemini:", repr(e))
        raise HTTPException(status_code=502, detail="The AI could not answer. Check the server window for details.")
