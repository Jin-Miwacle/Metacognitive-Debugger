
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from qg_format import qg_input, qa_input, detect_lang

MODEL_NAME = "MT5_Second_model"   # your OUTPUT_DIR or the HF repo id
SRC_LEN = 768                     # same as train_mt5.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

def _gen(prompt, max_new_tokens=64):
    x = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SRC_LEN)
    with torch.inference_mode():
        out = model.generate(**x, max_new_tokens=max_new_tokens, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def generate_question(context, answer, lang=None):
    lang = lang or detect_lang(context)
    return _gen(qg_input(answer, context, lang))

def answer_question(context, question, lang=None):
    lang = lang or detect_lang(context)
    return _gen(qa_input(question, context, lang))  


if __name__ == "__main__":
    passage = (
        "Plants use sunlight, water, and carbon dioxide "
        "to produce glucose and oxygen."
    )

    answer = "glucose and oxygen"

    print("Passage:", passage)
    print("Answer:", answer)
    print("Generated question:", generate_question(passage, answer))
