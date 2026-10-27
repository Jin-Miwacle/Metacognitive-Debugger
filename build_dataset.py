from qg_format import qg_input, qa_input

def to_examples(row):
    return [
        {"src": qg_input(row["answer"], row["context"], row["lang"]), "tgt": row["question"]},
        {"src": qa_input(row["question"], row["context"], row["lang"]), "tgt": row["answer"]},
    ]

def tokenize_all(rows, tok):
    examples = [e for row in rows for e in to_examples(row)]
    data = []
    for e in examples:
        x = tok(e["src"], max_length=768, truncation=True)
        x["labels"] = tok(text_target=e["tgt"], max_length=64, truncation=True)["input_ids"]
        data.append(x)
    return data