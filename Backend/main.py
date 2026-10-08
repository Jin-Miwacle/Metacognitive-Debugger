"""
Reading Partner backend (Brain: unsloth/Llama-3.2-3B-Instruct-bnb-4bit).

The widget (index.html) talks to three addresses on this server:
    POST /partner   ->  { "reply": "..." }       a hint while the student is debugging
    POST /quiz      ->  { "questions": [ ... ] } a quiz written from the student's text
    POST /checkin   ->  { "prompt": "..." }      a short mid-reading check-in question

Usage:
    1. On terminal, enter: huggingface-cli login
    2. Complete steps given and paste token
    3. Run python main.py
    4. Open index.html to test the website

"""
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("LLAMA_3B_4BIT_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")

print(f"Loading local model {MODEL_NAME} (this may take a moment)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.generation_config.max_length = None
print("Model loaded successfully!")

app = FastAPI(title="Reading Partner backend (Llama 3.2)")

# Lets the widget talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def ask_llama(system_rules: str, prompt: str, *, temperature: float, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Explicitly move input tensors to the model's device (handles offloading safely)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    do_sample = temperature > 0.0
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the newly generated text response
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
    mode: str = "debug"            # "debug", "followup", or "stuck"
    problem: Optional[str] = None  # "vocab", "lost", "connect", or "why"
    passage: str = ""              # words selected
    paragraph: str = ""            # paragraph around it
    goal: Optional[str] = None     # reading goal
    message: Optional[str] = None  # student latest reply
    history: List[Turn] = []       # conversation history

COACH_RULES = """You are a warm, encouraging reading coach for Grade 7 students. 

CRITICAL BEHAVIORAL MODES (Follow these strictly):
1. CHECK THE TARGET PASSAGE: If the student's reply matches or contains the target word/passage, they are RIGHT. You MUST validate them briefly in your own words, confirm they've got it, and tell them they can move on. DO NOT ask another question.
2. IF THE STUDENT SAYS "I don't know", "just tell me", or asks what something means: STOP asking questions. Give a clear, direct, simple explanation immediately.
3. OTHERWISE (Socratic Mode): Ask ONE simple guiding question to help them think.

General Rules:
- Keep your response under 45 words.
- Never use stock praise like "Good job."
- Use plain, easy-to-understand words."""

MODE_TEXT = {
    "debug": "This is the FIRST message. Go straight to your one move.",
    "followup": "The student replied. If right, confirm briefly and mark clear. If partly right, build on it. If off, shrink the task.",
    "stuck": "The student is STILL stuck. Break the passage into two tiny steps.",
    "success": "The student understands the concept now. Validate their realization briefly, confirm they've got it, and tell them they're ready to continue reading. Do NOT ask another question.",
}

PROBLEMS = {
    "vocab": (
        "The student does not know a word.", 
        lambda p: f'The student is stuck on "{p}". Unless they asked for a direct definition, guide them to look at the surrounding context words or break down "{p}" with ONE simple question.'
    ),
    "lost": (
        "The student lost the main point.", 
        lambda p: f'The student lost the point around "{p}". Unless they asked for a summary, ask them a guiding question about who is speaking or what action just happened right here.'
    ),
    "connect": (
        "The student cannot connect sentences.", 
        lambda p: f'The student cannot connect ideas around "{p}". Ask them a guiding question pointing back to the sentence right before it.'
    ),
    "why": (
        "The student cannot tell why something happened.", 
        lambda p: f'The student is confused why "{p}" happened. Ask them a guiding question to hunt for an earlier clue in the text.'
    ),
}

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
    raw_output = ask_llama(CLASSIFIER_RULES, prompt, temperature=0.1, max_tokens=5)
    
    cleaned = raw_output.strip().lower()
    if "success" in cleaned:
        return "success"
    if "stuck" in cleaned:
        return "stuck"
    return "followup"

def build_hint_prompt(ask: Ask) -> str:
    passage = ask.passage or "this part"
    problem_key = ask.problem or "lost"
    
    what, get_instruction = PROBLEMS.get(problem_key, PROBLEMS["lost"])
    task_instruction = get_instruction(passage) if callable(get_instruction) else get_instruction

    lines = []
    lines.append("Target audience: Grade 7 student (keep language simple and clear).")
    if ask.goal:
        lines.append(f"Goal: {ask.goal}")
    if ask.paragraph:
        lines.append(f"Paragraph context: {ask.paragraph[:600]}")
    lines.append(f"Target passage: \"{passage}\"")
    lines.append(f"Problem type: {what}")
    lines.append(f"Task: {task_instruction}")
    
    if ask.history:
        lines.append("\nConversation History:")
        for turn in ask.history:
            speaker = "Coach" if turn.role == "coach" else "Student"
            lines.append(f"{speaker}: {turn.text}")
            
    if ask.message:
        lines.append(f"\nStudent (Latest): {ask.message[:300]}")
        
    lines.append(f"\nMode: {MODE_TEXT.get(ask.mode, MODE_TEXT['debug'])}")
    lines.append("Write ONLY the coach's next response.")
    return "\n".join(lines)

def generate_hint(ask: Ask) -> str:
    return ask_llama(COACH_RULES, build_hint_prompt(ask), temperature=0.7, max_tokens=150)


# =====================================================================
# 2) QUIZ  (POST /quiz)
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

def generate_quiz(ask: QuizAsk) -> list:
    n = max(2, min(ask.n, 5))
    prompt = f"Create {n} reading comprehension questions based on this text:\n\n{ask.text[:4000]}\n\nReturn strictly valid JSON."
    raw_response = ask_llama(QUIZ_RULES, prompt, temperature=0.3, max_tokens=1024)
    
    try:
        # Extract json if wrapped in markdown code blocks
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
    for q in questions:
        if "options" in q and len(q["options"]) == 4:
            cleaned.append(q)
    if not cleaned:
        raise RuntimeError("No valid questions parsed from model output.")
    return cleaned


# =====================================================================
# 3) CHECK-INS  (POST /checkin)
# =====================================================================
class CheckinAsk(BaseModel):
    goal: Optional[str] = None
    text: str = ""
    struggles: List[str] = []

CHECKIN_RULES = """Write ONE short check-in question (under 25 words) testing understanding of the text. No greeting or extra text. Question only."""

def generate_checkin(ask: CheckinAsk) -> str:
    prompt = f"Text context: {ask.text[:3000]}\nGenerate a single checking question."
    return ask_llama(CHECKIN_RULES, prompt, temperature=0.8, max_tokens=60)


# =====================================================================
# FastAPI Endpoints
# =====================================================================
@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME, "endpoints": ["/partner", "/quiz", "/checkin"]}

@app.post("/partner")
def partner(ask: Ask):
    try:
        if ask.message:
            ask.mode = classify_intent(ask.message)
        else:
            ask.mode = "debug"
            
        return {"reply": generate_hint(ask)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/quiz")
def quiz(ask: QuizAsk):
    try:
        return {"questions": generate_quiz(ask)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/checkin")
def checkin(ask: CheckinAsk):
    try:
        return {"prompt": generate_checkin(ask)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# =====================================================================
# Terminal Testing Interface
# =====================================================================
def run_terminal_loop():
    print("\n" + "="*50)
    print(" READING PARTNER: TERMINAL TEST MODE (Llama 3.2)")
    print("="*50)
    while True:
        print("\nChoose action:")
        print("  1. Test Hint (/partner)")
        print("  2. Test Quiz (/quiz)")
        print("  3. Test Check-in (/checkin)")
        print("  4. Exit")
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            passage = input("Enter confusing passage text: ")
            problem = input("Problem type (vocab / lost / connect / why): ") or "vocab"
            ask_obj = Ask(mode="debug", problem=problem, passage=passage)
            print("\n[Coach thinking...]")
            print(f"\nCoach: {generate_hint(ask_obj)}")
            
        elif choice == "2":
            text = input("Paste text for quiz generation: ")
            ask_obj = QuizAsk(text=text, n=2)
            print("\n[Generating Quiz...]")
            try:
                questions = generate_quiz(ask_obj)
                for i, q in enumerate(questions, 1):
                    print(f"\nQ{i} ({q.get('kind')}): {q.get('q')}")
                    for idx, opt in enumerate(q.get('options')):
                        print(f"  [{idx}] {opt}")
                    print(f"  Answer Index: {q.get('answer')} | Why: {q.get('why')}")
            except Exception as e:
                print("Error:", e)
                
        elif choice == "3":
            text = input("Paste reading text snippet: ")
            ask_obj = CheckinAsk(text=text)
            print("\n[Generating Check-in...]")
            print(f"\nCheck-in Question: {generate_checkin(ask_obj)}")
            
        elif choice == "4":
            print("Exiting terminal mode. Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminal":
        run_terminal_loop()
    else:
        import uvicorn
        print("\nStarting FastAPI server for HTML widget on http://localhost:8000 ...")
        print("To run terminal testing instead, use: python backend.py terminal\n")
        uvicorn.run(app, host="127.0.0.1", port=8000)