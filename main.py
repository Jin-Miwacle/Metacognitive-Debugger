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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

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
        max_tokens=500
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

        highlighted = (
            passage[:start]
            + "<hl> "
            + exact_answer
            + " <hl>"
            + passage[start + len(exact_answer):]
        )

        prompt = f"generate question: {highlighted}"

        inputs = qg_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {key: value.to(qg_device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = qg_model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4
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
        max_tokens=1200
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

        if evidence not in passage:
            print(
                f"[Quiz] Rejected evidence not found in passage: "
                f"{item['q']}"
            )
            print(f"[Quiz] Evidence: {evidence}")
            continue

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

# Separate T5 model used only for quiz question generation
T5_MODEL_NAME = "valhalla/t5-base-qg-hl"

print(f"Loading T5 question-generation model: {T5_MODEL_NAME}...")

qg_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)

qg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qg_model.to(qg_device)
qg_model.eval()

print("T5 question-generation model loaded successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminal":
        run_terminal_loop()
    else:
        import uvicorn
        print("\nStarting FastAPI server for HTML widget on http://localhost:8000 ...")
        print("To run terminal testing instead, use: python backend.py terminal\n")
        uvicorn.run(app, host="127.0.0.1", port=8000)

