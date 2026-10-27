"""
Q1 to Q6: checks on a finished multiple-choice item.

The coach has R0 to R11 because a rule written into a prompt is a request and a
rule written in code is a guarantee. The quiz builder has about sixty rules, all
of them in the prompt, and none of them checked afterwards. The only check on
option length prints a warning and keeps the item.

These are the giveaways that let a student pick the right answer without having
understood the passage. Each returns a short code, the same shape as
grounding.check_reply, so main.py can regenerate once and then drop the item.

    from quizrules import check_item
    problems = check_item(question, options, answer_index, passage, lang)
"""
import re

__all__ = ["check_item", "check_label", "label_for", "answer_cue_free", "Q_CODES"]

# There is no check here for an absurd distractor. Telling "aliens from Mars"
# from "the seasonal rain" needs world knowledge, and every regular-expression
# version of it threw away good distractors on real output: "western Mindanao"
# and "central Visayas" are exactly what you want against "northern Luzon".
# A check that cries wolf is worse than no check, so that one is left to a human.

Q_CODES = {
    "Q1": "the correct option is much longer than the distractors",
    "Q2": "only the correct option is lifted from the passage",
    "Q3": "the question's own wording only fits the correct option",
    "Q4": "only the correct option is the kind of thing the question asks for",
    "Q6": "a distractor is contained in, or contains, the correct option",
    "Q7": "the item is labelled Inferential but the question is answerable word for word",
}

# A question is inferential when answering it needs a step past copying: a reason,
# a cause, a purpose, a method, or a conclusion. Everything else is literal.
_INFERENTIAL_Q = re.compile(
    r"\b(why|what caused|what made|what led to|for what purpose|what does .* suggest|"
    r"what can you (tell|conclude|infer)|how did .* manage|how does .* explain|"
    r"bakit|anong dahilan|ano ang dahilan|ano ang ipinapahiwatig|ano ang maaaring|"
    r"paano nangyari|ano ang natutunan)\b", re.I)

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

# Words too common to count as shared subject matter.
_STOP = set("""
a an the of in on at to for from by with and or but is are was were be been being
this that these those it its as not no than then so such which who whom whose what
when where why how all any each more most other some only own same too very can will just
ang ng mga sa ay na at o pero kung dahil ito iyon siya sila kami tayo ako ikaw
ni kay si para nang may mayroon hindi wala rin din lang lamang pa ba
""".split())


def _words(s):
    return [w.lower() for w in _WORD.findall(s or "")]


def _content(s):
    return {w for w in _words(s) if w not in _STOP and len(w) > 2}


def _norm(s):
    return " ".join(_words(s))


def _in_passage(option, passage):
    """The option's words appear in the passage in that order."""
    o, p = _norm(option), _norm(passage)
    return bool(o) and o in p


# ---------------------------------------------------------------- Q4 helpers
_NUMBER = re.compile(r"\d")
_SPELLED = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"hundred|thousand|million|isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampu|daan|libo)\b", re.I)
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_PLACE_WORD = re.compile(
    r"\b(province|city|town|village|island|mountain|mountains|river|lake|region|district|valley|"
    r"country|street|probinsya|lalawigan|lungsod|bayan|pulo|bundok|ilog|lawa|rehiyon|kalye|bansa)\b", re.I)
_PERSON_WORD = re.compile(
    r"\b(people|peoples|tribe|folk|community|farmers|settlers|workers|friars|priests|monks|"
    r"soldiers|traders|colonists|villagers|engineers|scientists|students|teachers|"
    r"mamamayan|katutubo|magsasaka|manggagawa|pari|sundalo|mangangalakal|guro|estudyante)\b", re.I)
_TITLE_CASE = re.compile(r"^(?:[A-Z][\w.'’-]*[\s-]*){1,4}$")

_ASKS = {
    "quantity": re.compile(r"\b(how many|how much|how old|how long|ilan|ilang|gaano ka|magkano)\b", re.I),
    "date":     re.compile(r"\b(when|what year|which year|in what year|kailan|anong taon|aling taon)\b", re.I),
    "person":   re.compile(r"\b(who|whose|whom|sino|sinong|kanino|kaninong)\b", re.I),
    "place":    re.compile(r"\b(where|saan|saang|nasaan)\b", re.I),
}


def _asks_for(question):
    q, best, at = question or "", None, len(question or "") + 1
    for kind, pat in _ASKS.items():
        m = pat.search(q)
        if m and m.start() < at:
            best, at = kind, m.start()
    return best


def _looks_like(option, kind, passage=""):
    o = (option or "").strip()
    if not o:
        return False
    if kind == "quantity":
        return bool(_NUMBER.search(o) or _SPELLED.search(o))
    if kind == "date":
        return bool(_YEAR.search(o) or re.search(r"\b(siglo|century|AD|BC|CE|BCE)\b", o, re.I))
    if kind == "place":
        if _PLACE_WORD.search(o):
            return True
        return bool(_TITLE_CASE.match(o) and re.search(
            r"\b(in|at|from|near|sa|mula sa|papuntang)\s+" + re.escape(o.split()[0]), passage, re.I))
    if kind == "person":
        if _PERSON_WORD.search(o):
            return True
        return bool(_TITLE_CASE.match(o) and re.search(
            r"\b(si|ni|kay|sina|Mr|Mrs|Ms|Dr|President|King|Queen|Sir|Lord)\.?\s+"
            + re.escape(o.split()[0]), passage))
    return False


# ---------------------------------------------------------------- the checks
def check_item(question, options, answer_index, passage, lang="en", *,
               length_ratio=1.8, length_margin=4):
    """Violations in one finished item. An empty list means nothing gives it away."""
    problems = []
    if not options or not (0 <= answer_index < len(options)):
        return ["Q0:bad_item"]

    correct = options[answer_index]
    distractors = [o for i, o in enumerate(options) if i != answer_index]
    if not distractors:
        return ["Q0:bad_item"]
    kind = _asks_for(question)

    # Q1. The giveaway everyone knows: the correct option is written from the
    # passage and the distractors are tossed off, so the long one is the answer.
    c_len = len(_words(correct))
    longest_d = max(len(_words(d)) for d in distractors)
    if c_len > longest_d + length_margin and c_len > longest_d * length_ratio:
        problems.append(f"Q1:length_cue:{c_len}v{longest_d}")

    # Q2. A good distractor is invented, so the answer being the only one from the
    # passage is normal and not a flaw. What gives an item away is the answer
    # being a LONG verbatim lift while the others are short inventions: the
    # student just matches the longest run of passage words.
    if c_len >= 5 and _in_passage(correct, passage) \
            and not any(_in_passage(d, passage) for d in distractors):
        problems.append(f"Q2:answer_is_a_{c_len}_word_lift")

    # Q3. The stem's grammar fits one option only.
    stem = (question or "").rstrip("? ").rstrip()
    if re.search(r"\b(an)\s*$", stem, re.I):
        vowel = [o for o in options if o[:1].lower() in "aeiou"]
        if len(vowel) == 1 and _norm(vowel[0]) == _norm(correct):
            problems.append("Q3:grammar_cue:an")
    elif re.search(r"\b(a)\s*$", stem, re.I):
        cons = [o for o in options if o[:1].lower() not in "aeiou"]
        if len(cons) == 1 and _norm(cons[0]) == _norm(correct):
            problems.append("Q3:grammar_cue:a")

    # Q4. "Where..." with one place and three non-places answers itself.
    if kind:
        fitting = [o for o in options if _looks_like(o, kind, passage)]
        if len(fitting) == 1 and _norm(fitting[0]) == _norm(correct):
            problems.append(f"Q4:only_answer_is_a_{kind}")

    # Q6. One option inside another leaves a student choosing between two halves
    # of the same fact.
    cn = _norm(correct)
    for d in distractors:
        dn = _norm(d)
        if dn and cn and dn != cn and (dn in cn or cn in dn):
            problems.append(f"Q6:distractor_overlaps_answer:{d[:40]}")

    return problems


def label_for(question, correct_option, passage):
    """Literal or Inferential, judged from the finished question rather than taken
    on trust from whatever labelled the answer phrase before the question existed."""
    if _INFERENTIAL_Q.search(question or ""):
        return "Inferential"
    # If the answer is not the passage's own words, the student had to put
    # something together to reach it.
    if correct_option and not _in_passage(correct_option, passage):
        return "Inferential"
    return "Literal"


def check_label(kind, question, correct_option, passage):
    """Q7. One violation if the stated kind does not match the question."""
    stated = (kind or "").strip().lower()
    if stated not in ("literal", "inferential"):
        return []
    real = label_for(question, correct_option, passage).lower()
    if stated == real:
        return []
    return [f"Q7:labelled_{stated}_but_reads_{real}"]


def answer_cue_free(question, options, answer_index, passage, lang="en"):
    """True when nothing in the item points at the answer."""
    return not check_item(question, options, answer_index, passage, lang)
