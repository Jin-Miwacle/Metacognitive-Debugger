"""
Pick the best of the mT5 candidates instead of the first one that is not rejected.

Today main.py loops the candidates and keeps the first that passes
question_fits_answer(), which only rejects a quantity mismatch. Everything else
passes, so when the model offers

    1. The Banaue Rice Terraces are carved into?        (dropped, no question word)
    2. What is the Banaue Rice Terraces carved into?    (kept)
    3. Where are the Banaue Rice Terraces carved?       (never looked at)

for the answer "Ifugao province", the pipeline keeps number 2 even though
number 3 is the right question and the model already wrote it.

This scores every candidate and sorts, so the right question word wins when the
model happens to produce it. It changes no model and needs no retraining.
"""
import re

# ---- what kind of thing is the answer ---------------------------------------
_QUANTITY  = re.compile(r"\d")
_SPELLED_NUMBER = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
    r"hundred|thousand|million|billion|half|dozen|"
    r"isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampu|labing|dalawampu|tatlumpu|daan|libo|milyon)\b", re.I)
_BARE_YEAR = re.compile(r"^\s*\d{3,4}\s*$")
_MONTHS    = (r"january|february|march|april|may|june|july|august|september|october|november|december|"
              r"enero|pebrero|marso|abril|mayo|hunyo|hulyo|agosto|setyembre|oktubre|nobyembre|disyembre")
_DATE      = re.compile(rf"\b(\d{{3,4}}\s*(AD|BC|CE|BCE)?|{_MONTHS}|siglo|century)\b", re.I)
_DURATION  = re.compile(r"\b(years?|months?|days?|hours?|minutes?|weeks?|taon|buwan|araw|oras|beses|times)\b"
                        r"(?!\s*$)|\b(years?|taon)\s+(old|ago|long)\b", re.I)
_UNIT      = re.compile(r"\b(nm|mm|cm|km|m|kg|g|lb|ft|in|percent|porsyento|metres?|meters?|miles?|"
                        r"degrees?|members?|people|votes?|seats?)\b", re.I)

# A place: the passage puts a locative word right before it, or the name itself
# carries one of these words.
_PLACE_WORD = re.compile(r"\b(province|provinces|city|cities|town|village|island|islands|mountain|"
                         r"mountains|river|lake|region|district|valley|street|country|"
                         r"probinsya|lalawigan|lungsod|bayan|pulo|bundok|ilog|lawa|rehiyon|purok|kalye|bansa)\b", re.I)
_PLACE_PREP = r"(?:in|at|on|into|from|near|across|sa|mula\s+sa|papuntang|patungo\s+sa)\s+"
_PERSON_WORD = re.compile(r"\b(people|peoples|tribe|folk|community|settlers|farmers|workers|"
                          r"mamamayan|katutubo|magsasaka|manggagawa)\b", re.I)
_PERSON_MARK = r"(?:si|ni|kay|sina|nina|kina|Mr|Mrs|Ms|Dr|President|King|Queen|Prince|Princess|Sir|Lord|Pope|Senator|Governor|General)\.?\s+"
_TITLE_CASE = re.compile(r"^(?:[A-Z][\w.'’-]*[\s-]*){1,4}$")


def answer_kind(answer, passage=""):
    """quantity, date, person, place or other. Conservative: anything unclear is other."""
    a = (answer or "").strip()
    if not a:
        return "other"
    if _BARE_YEAR.match(a):
        return "date"
    # An amount carries its unit with it ("2,000 years old", "1,550 nm"), while a
    # period does not. The unit decides, otherwise every year looks like both.
    if (_DURATION.search(a) or _UNIT.search(a)) and (_QUANTITY.search(a) or _SPELLED_NUMBER.search(a)):
        return "quantity"
    # A period beats a plain number: "ika-19 na siglo" holds a digit but is a date.
    if len(a.split()) <= 5 and _DATE.search(a):
        return "date"
    if len(a.split()) <= 4 and (_QUANTITY.search(a) or _SPELLED_NUMBER.search(a)):
        # "2,000 years old" and "apat na beses" are both amounts
        return "quantity"
    if _PLACE_WORD.search(a):
        return "place"
    if _PERSON_WORD.search(a):
        return "person"
    if _TITLE_CASE.match(a) and passage:
        first = re.escape(a.split()[0])
        if re.search(_PERSON_MARK + first, passage):
            return "person"
        if re.search(_PLACE_PREP + first, passage, re.I):
            return "place"
    return "other"


# ---- what kind of thing does the question ask for ---------------------------
# Searched anywhere in the question, not only at the start, because a question
# often fronts a phrase first: "Along with electronic pop, what type...?" and
# "Sa kabila ng ano, nagpatibay...?" are both ordinary questions.
_Q = {
    "quantity": re.compile(r"\b(how many|how much|how old|how long|how far|ilan|ilang|gaano ka|magkano)\b", re.I),
    "date":     re.compile(r"\b(when|in what year|in which year|what year|which year|by what year|during what|"
                           r"kailan|kailang|anong taon|aling taon|noong anong|anong siglo|aling siglo)\b", re.I),
    "person":   re.compile(r"\b(who|whose|whom|sino|sinong|kanino|kaninong)\b", re.I),
    "place":    re.compile(r"\b(where|saan|saang|nasaan)\b", re.I),
}
_ANY_QUESTION_WORD = re.compile(
    r"\b(who|whose|whom|what|which|when|where|why|how|"
    r"sino|sinong|kanino|kaninong|ano|anong|alin|aling|kailan|saan|saang|nasaan|bakit|paano|paanong|"
    r"ilan|ilang|gaano|magkano)\b", re.I)


def has_question_word(question):
    """True if the text carries a question word in either language."""
    return bool(_ANY_QUESTION_WORD.search(question or ""))


def question_kind(question):
    q = question or ""
    # The earliest question word wins, so "What year" beats a later "how".
    best, at = "other", len(q) + 1
    for kind, pat in _Q.items():
        m = pat.search(q)
        if m and m.start() < at:
            best, at = kind, m.start()
    return best


def question_fits_answer(question, answer, passage=""):
    """Hard reject. Keeps main.py's existing rule and adds the obvious mismatches."""
    if not question or not question.strip().endswith("?"):
        return False
    if not _ANY_QUESTION_WORD.search(question):
        return False
    a_kind, q_kind = answer_kind(answer, passage), question_kind(question)
    # Only a confident clash rejects. When either side reads as "other" the kind
    # is unknown, and guessing there threw out good questions.
    # A counting question needs a counted answer. "How many rainforests...?"
    # cannot be answered by "rainforests above".
    if q_kind == "quantity" and not (_QUANTITY.search(answer) or _SPELLED_NUMBER.search(answer)):
        return False
    if "other" in (a_kind, q_kind):
        return True
    CLASH = {
        ("quantity", "person"), ("quantity", "place"),
        ("person", "quantity"), ("person", "place"), ("person", "date"),
        ("place", "quantity"), ("place", "person"), ("place", "date"),
        ("date", "person"), ("date", "place"),
    }
    return (a_kind, q_kind) not in CLASH


def question_score(question, answer, passage=""):
    """Lower is better. Used to sort candidates, not to reject them."""
    a_kind, q_kind = answer_kind(answer, passage), question_kind(question)
    score = 0
    if a_kind != "other":
        score += 0 if q_kind == a_kind else 10          # the question word fits the answer
    if not question.strip().endswith("?"):
        score += 5
    words = len(question.split())
    if words < 4:
        score += 3                                       # "Where did the Terraces have?" is a stub
    if re.search(r"\b(\w+)\s+\1\b", question, re.I):
        score += 2                                       # a repeated word
    if answer and answer.strip().lower() in question.lower():
        score += 8                                       # the question gives its own answer away
    return score


def rank_questions(candidates, answer, passage=""):
    """The same candidates, best first. Nothing is removed, so a caller that
    already has its own filters keeps them and simply sees the best one first.
    Ties keep the model's own order, so the beam's first choice still wins a tie."""
    return sorted(candidates,
                  key=lambda c: (0 if question_fits_answer(c, answer, passage) else 1,
                                 question_score(c, answer, passage),
                                 candidates.index(c)))


def best_question(candidates, answer, passage=""):
    """The candidate to keep, or None when every one of them is unusable.
    Ties keep the model's own order, so the beam's first choice still wins a tie."""
    usable = [c for c in candidates if question_fits_answer(c, answer, passage)]
    if not usable:
        return None
    return min(usable, key=lambda c: (question_score(c, answer, passage), candidates.index(c)))
