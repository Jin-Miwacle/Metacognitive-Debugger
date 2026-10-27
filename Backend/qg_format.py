import re

HL = "<hl>"  # plain text marker; mT5 splits it into normal pieces, so no new tokens or embedding resize

_FIL_HINT = re.compile(r"\b(ang|ng|mga|sa|ay|na|hindi|ko|po|ito|siya|nila|kami|tayo|ba)\b", re.I)

def detect_lang(text: str) -> str:
    return "fil" if len(_FIL_HINT.findall(text[:1500])) >= 4 else "en"

def qg_input(answer: str, context: str, lang: str) -> str:
    """Task 1: (answer, passage) -> question. The answer is highlighted in the passage."""
    start = context.lower().find(answer.lower())
    if start >= 0:
        end = start + len(answer)
        context = f"{context[:start]}{HL} {context[start:end]} {HL}{context[end:]}"
    return f"generate question | lang: {lang} | answer: {answer} | context: {context}"

def qa_input(question: str, context: str, lang: str) -> str:
    """Task 2: (question, passage) -> answer. Used to check the generated question."""
    return f"answer question | lang: {lang} | question: {question} | context: {context}"