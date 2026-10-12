
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "valhalla/t5-base-qg-hl"

print(f"Loading T5 question-generation model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def generate_question(context: str, answer: str) -> str:
    """Generate one question about a highlighted answer in the context."""

    context = context.strip()
    answer = answer.strip()

    if not context or not answer:
        raise ValueError("Both context and answer are required.")

    highlighted = context.replace(
        answer,
        f"<hl> {answer} <hl>",
        1
    )

    if highlighted == context:
        raise ValueError("The answer was not found in the context.")

    prompt = f"generate question: {highlighted}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )

    question = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return question


if __name__ == "__main__":
    passage = (
        "Plants use sunlight, water, and carbon dioxide "
        "to produce glucose and oxygen."
    )

    answer = "glucose and oxygen"

    print("Passage:", passage)
    print("Answer:", answer)
    print("Generated question:", generate_question(passage, answer))
