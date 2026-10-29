"""
Grounding rules for the Reading Partner (spec: GROUNDING_RULES_README.md).

Code makes the decisions; the model only does two narrow jobs (label a question against
numbered sentences, list the meanings a phrase could have), and everything it cites is
checked here against the passage.

    R0  gate        run R1-R3 only for questions (or explain mode / a first vocab turn)
    R1  names       a name in the question that the passage never mentions -> NOT_IN_TEXT
    R2  evidence    code picks candidate sentences, model labels, code verifies the ids
    R3  ambiguity   meaning questions: model lists senses, code keeps the supported ones
    R4  hedging     INFERABLE replies must sound like an inference
    R5  output      names, numbers and quotes in every reply must come from the sources
    R6  claims      an answer the text contradicts is not agreed with (model labels, code verifies ids)
    R7  facts       statements in the coach's reply that the text doesn't back are removed (same pattern)
    R8  asking      a hint turn has to ask the student something, not state the answer
    R9  language    the reply is in the language that was chosen for it
    R10 repetition  the reply is not a near-copy of something already said in this thread

Pure functions only: no torch, no model. The model call is passed in as `ask`, with the
same signature as main.ask_sealion, so the tests can use a fake.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

STATED, INFERABLE, NOT_STATED = "STATED", "INFERABLE", "NOT_STATED"
NOT_IN_TEXT, AMBIGUOUS, SKIPPED = "NOT_IN_TEXT", "AMBIGUOUS", "SKIPPED"
CONTRADICTED = "CONTRADICTED"     # R6: the student's answer disagrees with the text


@dataclass
class GroundingResult:
    label: str = SKIPPED          # STATED | INFERABLE | NOT_STATED | NOT_IN_TEXT | AMBIGUOUS | SKIPPED
    rule: str = "R0:not_a_question"
    evidence_ids: list = field(default_factory=list)
    evidence: list = field(default_factory=list)       # the verified sentences, verbatim
    missing_names: list = field(default_factory=list)
    senses: list = field(default_factory=list)         # R3: [{"sense": "...", "evidence": [ids]}]
    notes: list = field(default_factory=list)          # trace lines for logs / the thesis
    target: str = ""              # R3: the phrase, spelled exactly as in the passage
    question: str = ""
    term: str = ""                # what the passage does mention (for the NOT_STATED template)
    claim: str = ""               # R6's verdict on what the student said, when one was asked for

    def log_line(self) -> str:
        line = (f"[grounding] label={self.label} rule={self.rule} "
                f"missing={self.missing_names} evidence={self.evidence_ids}")
        if self.senses:
            line += f" senses={[s['sense'] for s in self.senses]}"
        if self.target:
            line += f" target={self.target!r}"
        return line


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', " ": " "})


def norm(s: str) -> str:
    """Case, spacing and curly-quote differences removed."""
    s = unicodedata.normalize("NFKC", s or "").translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _words(s: str) -> list:
    return re.findall(r"\w+(?:['-]\w+)*", norm(s))


def _has_word(phrase: str, text: str) -> bool:
    """Whole-word, case-insensitive match (Michelle matches Michelle's, not Michelleville)."""
    phrase = norm(phrase)
    return bool(phrase) and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", norm(text)) is not None


_STOP = set("""
a an the and or but nor so if then than that this these those there here it its is are was were be been
being am do does did done to of in on at for with from by as about into over under after before not no yes
can could would should will shall may might must has have had having also just very really what who whom
whose where when why how which i you he she we they me him her us them my your his our their mine yours
hers ours theirs said says say tell told mean means meaning word words part text passage story sentence
line page about please pls know dont don't idk okay ok thing something someone one
ang ng mga sa si ni kay sina nina kina ay na at o pero ito iyan iyon yan yun siya sila niya nila ako ko
ikaw ka mo kami tayo natin namin ba po ano anong sino sinong saan kailan bakit paano alin ilan kanino
nasaan ibig sabihin kahulugan nito dito doon may mayroon meron wala hindi oo lang din rin naman kung
kasi dahil para nang pa nga yung isa
""".split())

# Capitalized words that are not names even mid-sentence. The second block is
# words that commonly OPEN a sentence: without them, checking a sentence's first
# word (see extract_names' include_first) would read "Suddenly" as a character.
_NOT_NAMES = _STOP | set("""
look think read notice try great good let let's now remember okay hmm well english filipino tagalog
taglish grade coach student tingnan basahin subukan tama magaling sige teksto
""".split()) | set("""
knowing looking reading keeping holding listening watching noticing thinking trying picking
choosing starting seeing saying telling asking getting putting taking making working going
nice exactly listen watch spot compare count name follow
""".split()) | set("""
summarize summarise predict connect recall retell restate list state give write say show match
decide guess review skim scan underline highlight mark explain describe picture
tukuyin ibuod hulaan ikonekta isalaysay ilarawan ipaliwanag suriin markahan
""".split()) | set("""
after again almost already also although always another any because before begin both but check
choose close consider describe despite during each either even every except explain find first
focus following for from good great here however if imagine instead keep last later let look
maybe meanwhile more most next no notice now once only open or other perhaps pick picture
read remember right second see several since so some start still such suddenly take tell than
everyone everybody people someone somebody anyone anybody nobody colour color size time reason point
part thing things word words sentence sentences line lines story character characters author
that then there these they think this those though through together too try unless until use
very watch well what when where whether which while who why with without yes yet your
ang anong ano alin bago bakit basahin bawat dahil dapat dito doon gayunpaman halimbawa hanapin
higit huwag ikaw isipin isa kahit kailan kailangan kapag karaniwan kasi katulad kaya kung
lahat lamang madalas mag maging maaari marahil masdan maski matapos mayroon minsan mula muli
ngayon pagkatapos pansinin para pero pumili sabihin sakali samakatuwid sana sapagkat siguro
simula sinong subalit subukan sundin tama tapos tignan tingnan tuwing tulad una unang
""".split())

_TITLES = {"mr", "mrs", "ms", "dr", "gng", "g", "bb", "mang", "aling", "lola", "lolo", "tita", "tito",
           "ate", "kuya", "sir", "miss", "madam", "ma'am", "gin", "san", "santa", "sta", "sto"}

_PERSON_MARKERS = {"si", "ni", "kay", "sina", "nina", "kina"}


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Gng|Bb|G|St|Jr|Sr|Sto|Sta)\.", re.I)
# Split after . ! ? (and any closing quote/bracket right after it) followed by spaces.
_SENT_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+")


def split_sentences(text: str) -> list:
    """Sentences of the passage, in order. Sentence N is sentences[N - 1]."""
    out = []
    for line in re.split(r"\n+", text or ""):
        line = _ABBREV.sub(lambda m: m.group(1) + "\x00", line.strip())
        for s in _SENT_END.split(line):
            s = s.replace("\x00", ".").strip()
            if s:
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# R0 gate
# ---------------------------------------------------------------------------
_QWORDS = ("who|whom|whose|what|what's|whats|where|when|why|how|which|sino|sinong|ano|anong|saan|saang|"
           "kailan|bakit|paano|paanong|alin|ilan|kanino|nasaan")
_Q_START = re.compile(rf"^\W*({_QWORDS})\b", re.I)


_Q_ANYWHERE = re.compile(rf"\b({_QWORDS})\b", re.I)
# Yes/no questions: "is ethan her brother?", "did it rain?", "totoo ba...?"
_AUX_START = re.compile(r"^\W*(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had|"
                        r"isn't|wasn't|doesn't|didn't|totoo|may|mayroon|meron)\b", re.I)


def is_question(message: Optional[str]) -> bool:
    """A question to the coach. A short reply that only ends in "?" ("her grandmother?") is a
    hesitant answer, not a question, unless it has a question word or opens like a yes/no
    question ("is ethan her brother?")."""
    m = (message or "").strip()
    if not m:
        return False
    if _Q_START.match(m):
        return True
    return m.endswith("?") and (len(m.split()) > 4 or bool(_Q_ANYWHERE.search(m)) or bool(_AUX_START.match(m)))


_MEANING_Q = re.compile(r"\b(mean|means|meaning|define|definition|ibig\s+sabihin|kahulugan|ibig-sabihin)\b", re.I)


def is_meaning_question(message: Optional[str]) -> bool:
    return bool(_MEANING_Q.search(message or ""))


# ---------------------------------------------------------------------------
# R1 names
# ---------------------------------------------------------------------------
_CAP_SEQ = re.compile(r"[A-ZÀ-ÖØ-Þ][\w'’-]*(?:[ \t]+[A-ZÀ-ÖØ-Þ][\w'’-]*)*")
_SENT_START = re.compile(r"(^|[.!?:;\n]|^\W)[\s\"'“‘(]*$")
# Lowercase names the student typed: "who is michelle", "where is michelle", "what about michelle",
# "sino si michelle", "kay michelle".
_LOWER_NAME = re.compile(
    r"(?:^\W*(?:(?:who|where)\s+(?:is|was|are|were|did|does)|who's|whos|where's|wheres|"
    r"(?:what|how)\s+about)\s+"
    r"|\b(?:tell|talk)\s+me\s+(?:about|more\s+about)\s+"
    r"|\b(?:explain|describe|about|regarding)\s+"
    r"|\b(?:what|who|where|when|why|how)\s+(?:did|does|is|was|are|were)\s+"
    # "i dont get what michelle did": the name sits between the question word and
    # the verb, so the verb has to be looked for ahead rather than matched before.
    r"|\b(?:what|who|why|how|bakit|paano|anong?)\s+"
    # No copulas here: "what colour was the lamp" is a question about a thing,
    # not about someone named Colour. Only verbs that take a doer.
    r"(?=[^\W\d_][\w'-]*\s+(?:did|does|said|says|meant|means|wanted|felt"
    r"|thought|knew|saw|went|ginawa|sinabi)\b)"
    r"|\b(?:si|sina|kay|kina|ni|nina|nasaan|tungkol\s+kay)\s+)"
    r"([^\W\d_][\w'-]*)"
    r"(?:\s+([^\W\d_][\w'-]*))?", re.I)


_CONTRACTION = re.compile(r"'(re|ll|ve|d|m|t|s)$", re.I)


def _clean_token(t: str) -> str:
    t = t.replace("’", "'")
    return re.sub(r"'s$", "", t, flags=re.I).strip("'-")


def _not_a_name(token: str) -> bool:
    """Ordinary words, including the contractions that open a sentence.

    Checking a sentence's first word made "You're right" read as a character
    called You're, which would fire R5 on a large share of perfectly good
    replies. The part before the apostrophe is what decides it, so O'Brien
    still counts as a name.
    """
    low = token.lower()
    if low in _NOT_NAMES or low in _PERSON_MARKERS:
        return True
    base = _CONTRACTION.sub("", low)
    return base != low and (base in _NOT_NAMES or base in _CONTRACTION_BASES)


_CONTRACTION_BASES = set("""
don doesn didn isn aren wasn weren won wouldn couldn shouldn hasn haven hadn ain mustn needn
shan daren oughtn mightn
""".split())

# Filipino verbs carry their marking in the word itself, and a command opens a
# sentence capitalised exactly where a name would: "Ipaliwanang kung bakit...".
# Reading one as a character made the coach answer "the passage doesn't mention
# Ipaliwanang". Only the first word of a sentence is judged this way, since
# that is the only place the capital proves nothing.
# Only prefixes that mark a verb and nothing else. Short ones like "ma" and
# "na" are left out on purpose: they would swallow Maria and Nanding.
_FIL_VERB_PREFIX = ("ipa", "ipag", "ipinag", "ipang", "magpa", "nagpa", "magsi", "nagsi",
                    "maki", "naki", "makipag", "nakipag", "mang", "pinag", "ipinaki")


def _looks_like_a_verb(word: str) -> bool:
    """Filipino verb shape, by a prefix that only verbs carry."""
    w = word.lower()
    return len(w) >= 6 and any(w.startswith(p) for p in _FIL_VERB_PREFIX)


def extract_names(text: str, lowercase_hints: bool = False,
                  include_first: bool = False) -> list:
    """Capitalized words/sequences that look like names.

    By default the first word of a sentence is skipped, since every sentence
    starts with a capital and "Look" is not a person. That left a hole: an
    invented name as the first word went unseen, so "Michelle lit the lamp"
    passed while "The lamp was lit by Michelle" was caught.

    include_first=True keeps the first word too, relying on _NOT_NAMES to
    discard the ordinary words that open sentences.
    """
    text = _ABBREV.sub(lambda m: m.group(1), text or "")
    # A word inside quote marks is being quoted or defined, not used as a name:
    # 'Behind' means 'at the back of' is the coach explaining a word. Quoted
    # spans have their own rule (check_reply verifies them against the passage),
    # so blanking them here loses no coverage. Spaces keep every offset intact.
    text = _QUOTED.sub(lambda m: " " * len(m.group(0)), text.translate(_QUOTE_MAP))
    names = []
    for m in _CAP_SEQ.finditer(text):
        tokens = m.group(0).split()
        opens_sentence = bool(_SENT_START.search(text[:m.start()]))
        if not include_first and opens_sentence:
            tokens = tokens[1:]
        kept = [_clean_token(t) for t in tokens]
        kept = [t for t in kept if len(t) > 1 and not _not_a_name(t)]
        # The first word of a sentence is capitalised whatever it is, so a word
        # shaped like a verb there is a verb, not a character.
        if include_first and opens_sentence and kept and _looks_like_a_verb(kept[0]):
            kept = kept[1:]
        if kept:
            names.append(" ".join(kept))
    if lowercase_hints:
        for m in _LOWER_NAME.finditer(text):
            word = _clean_token(m.group(1))
            if word.lower() in _TITLES and m.group(2):      # "mang tomas", "aling nena"
                word = f"{word} {_clean_token(m.group(2))}"
            if len(word) > 1 and word.split()[-1].lower() not in _NOT_NAMES:
                names.append(" ".join(w[0].upper() + w[1:] for w in word.split()))
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def find_missing_names(names: list, passage: str) -> list:
    """Names with no whole-word match in the passage. A full name counts as present when
    any part of it is (so "Michelle Cruz" is present if the passage says "Michelle")."""
    missing = []
    for name in names:
        parts = [p for p in name.split() if p.lower().rstrip(".") not in _TITLES and len(p) > 1]
        if not parts:
            continue
        if _has_word(name, passage) or any(_has_word(p, passage) for p in parts):
            continue
        missing.append(name)
    return missing


def earlier_missing_names(earlier: list, passage: str) -> list:
    """Names the student brought up earlier in this thread that the passage never mentions."""
    names = []
    for text in earlier or []:
        names += extract_names(text, lowercase_hints=True, include_first=True)
    return list(dict.fromkeys(find_missing_names(names, passage)))


def names_in(text: str, earlier_missing: list) -> list:
    """Names in this message, plus any earlier missing name it mentions again in any case
    ("is ethan her brother?" after "Who is Ethan?")."""
    names = extract_names(text, lowercase_hints=True, include_first=True)
    have = {n.lower() for n in names}
    for n in earlier_missing or []:
        if n.lower() not in have and _has_word(n, text):
            names.append(n)
    return names


# The words a reading goal is made of once the subject is taken out. A goal with
# nothing left after these is about reading rather than about a topic.
_GOAL_WORDS = set("""
understand understanding understood learn learning learnt learned know knowing knew
read reading find finding want wanting like better more fully properly well
about what why how who where when happen happens happened going
story text passage reading article thing things part parts idea ideas point points
mean means meaning everything something anything
alam alamin matutunan maintindihan intindihin basahin bumasa kuwento kwento teksto
""".split())


def goal_fit(goal: str, passage: str) -> dict:
    """Does the student's reading goal have anything to do with this text?

    Checked when the goal is set, which is the only moment the student can still
    do something about it. A goal about Mary against a text that never mentions
    her is not an error to correct later: it is the student reading the whole
    piece waiting for something that never arrives.

    Three verdicts, and the difference between them is how sure we are.

    "names" is near-certain. A proper name the text never uses is the same signal
    R1 already trusts for questions, so it is reported plainly.

    "no_overlap" is a guess. A goal can fit a text perfectly without sharing a
    word with it ("why people leave home" against a story about a girl moving
    away), so this is worth raising as a question and never as a verdict.

    "ok" is everything else, which will be most goals.
    """
    goal = (goal or "").strip()
    passage = passage or ""
    if not goal or not passage:
        return {"verdict": "ok", "names": [], "shared": []}

    # No lowercase_hints here, unlike R1. In a question "what did michelle do"
    # is worth treating as a name; in a goal the same setting turned
    # "photosynthesis" into a missing person. A lowercase name is still caught,
    # just by the softer check below, which is the right way round.
    names = extract_names(goal, include_first=True)
    missing = find_missing_names(names, passage)
    if missing:
        return {"verdict": "names", "names": missing, "shared": []}

    # "I want to understand it better" is a real goal for any text and names no
    # subject, so it must not be judged against this one.
    terms = [t for t in key_terms(goal) if t not in _GOAL_WORDS]
    shared = [t for t in terms if _term_in(t, passage)]
    if terms and not shared:
        return {"verdict": "no_overlap", "names": [], "shared": []}
    return {"verdict": "ok", "names": [], "shared": shared[:5]}


# ---------------------------------------------------------------------------
# R2 evidence
# ---------------------------------------------------------------------------
def key_terms(message: str) -> list:
    quoted = [norm(q) for q in re.findall(r'["“]([^"”]+)["”]', message or "") if q.strip()]
    words = [re.sub(r"'s$", "", w) for w in _words(message or "")]
    words = [w for w in words if len(w) >= 3 and w not in _STOP and not w.isdigit()]
    return list(dict.fromkeys(quoted + words))


def _term_in(term: str, sentence: str) -> bool:
    if " " in term:
        return _has_word(term, sentence)
    for w in _words(sentence):
        if w == term:
            return True
        if len(term) >= 4 and len(w) >= 4 and (w.startswith(term) or term.startswith(w)):
            return True
    return False


def selection_ids(sentences: list, selection: str) -> list:
    sel = norm(selection)
    if len(sel) < 2:
        return []
    ids = [i for i, s in enumerate(sentences, 1) if sel in norm(s) or norm(s) in sel]
    if ids:
        return ids
    # A selection that spans a sentence break: sentences holding most of its words.
    sel_words = set(_words(selection))
    if not sel_words:
        return []
    return [i for i, s in enumerate(sentences, 1)
            if len(sel_words & set(_words(s))) >= max(2, 0.6 * len(sel_words))]


def candidate_sentences(sentences: list, terms: list, selection: str = "", k: int = 5,
                        neighbours: bool = True) -> list:
    """Sentence ids (1-based) worth showing the model: the selection's sentence(s), the best
    key-term matches, then the selection's neighbours, up to k."""
    chosen = list(dict.fromkeys(selection_ids(sentences, selection)))
    scored = []
    for i, s in enumerate(sentences, 1):
        score = sum(1 for t in terms if _term_in(t, s))
        if score:
            scored.append((-score, i))
    for _, i in sorted(scored):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    for i in list(chosen[:2]) if neighbours else []:
        for j in (i - 1, i + 1):
            if len(chosen) < k and 1 <= j <= len(sentences) and j not in chosen:
                chosen.append(j)
    return sorted(chosen[:k])


def _parse_json(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def _ids(value) -> list:
    if value is None:
        return []
    if isinstance(value, (int, str)):
        value = [value]
    out = []
    for v in value if isinstance(value, list) else []:
        out += [int(x) for x in re.findall(r"\d+", str(v))]
    return out


def _numbered(numbered: list) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in numbered)


LABEL_RULES = """You check whether a reading passage answers a student's question.
You see numbered sentences from the passage. Use ONLY those sentences, never outside knowledge.
Labels:
STATED: a sentence says the answer directly. Words a character says in dialogue count (if Ana says "I am tired", it is stated that Ana is tired).
INFERABLE: no sentence says it directly, but the sentences give clear clues to it.
NOT_STATED: the sentences do not answer the question.
Mentioning a person or thing is not an answer. For "Who is X?", STATED needs a sentence that says who X is (a role, job or relationship such as "his mother").
Output ONLY one JSON object: {"answer": "short answer, or empty", "label": "STATED" or "INFERABLE" or "NOT_STATED", "evidence": [sentence numbers]}
For "answer", use the passage's own words where you can (a few words). As evidence, list only the numbers of the sentences that state or hint at the answer. Use "" and [] for NOT_STATED."""

_RETRY_NOTE = "\n\nYour last output was not valid JSON. Output ONLY the JSON object, nothing else."


def label_with_model(ask: Callable, question: str, numbered: list, selection: str = "") -> Optional[dict]:
    """R2 model step. Returns {"label", "evidence", "attempts"} or None if unparseable twice."""
    prompt = f"Sentences:\n{_numbered(numbered)}\n\n"
    if selection:
        prompt += f"The student highlighted: \"{selection[:300]}\"\n"
    prompt += f"Student's question: \"{question[:300]}\"\n\nJSON:"
    for attempt in range(2):
        raw = ask(LABEL_RULES, prompt + (_RETRY_NOTE if attempt else ""), None,
                  temperature=0.0, max_tokens=150, use_adapter=False)
        data = _parse_json(raw)
        label = str(data.get("label", "")).strip().upper() if isinstance(data, dict) else ""
        if label in (STATED, INFERABLE, NOT_STATED):
            return {"label": label, "evidence": _ids(data.get("evidence")), "attempts": attempt + 1,
                    "answer": str(data.get("answer") or "").strip()}
    return None


# Why/how questions ask for a link between facts; the facts being written down does not make
# the link stated ("Why did Mika hug her brother?" -> "he passed the exam" is written, the reason is inferred).
_REASON_Q = re.compile(r"^\W*(why|how|bakit|paano|paanong)\b", re.I)


def verify(result: Optional[dict], candidate_ids: list, sentences: list, question: str = "") -> GroundingResult:
    """Code-side check of the model's label: cited ids must be candidates we showed it."""
    g = GroundingResult()
    if result is None:
        g.label, g.rule = NOT_STATED, "R2:unparseable"
        g.notes.append("R2: model output was not valid JSON twice -> NOT_STATED")
        return g
    cand = set(candidate_ids)
    cited = list(dict.fromkeys(result["evidence"]))
    valid = [i for i in cited if i in cand]
    dropped = [i for i in cited if i not in cand]
    if dropped:
        g.notes.append(f"R2: dropped evidence ids not shown to the model: {dropped}")
    g.label, g.rule = result["label"], "R2:model"
    if g.label in (STATED, INFERABLE) and not valid:
        g.notes.append(f"R2: model said {g.label} with no valid evidence -> NOT_STATED")
        g.label, g.rule = NOT_STATED, "R2:downgraded"
    answer_words = [w for w in _words(result.get("answer", "")) if w not in _STOP and len(w) > 2]
    if g.label == INFERABLE and answer_words and not _REASON_Q.match(question or ""):
        # The model's own answer is written word for word in one cited sentence: it is stated.
        for i in valid:
            if set(answer_words) <= set(_words(sentences[i - 1])):
                g.notes.append(f"R2: answer {result['answer']!r} is written in sentence {i} -> STATED")
                g.label, g.rule = STATED, "R2:answer_in_text"
                break
    # R2 safety net. The model sometimes answers NOT_STATED for something the
    # passage says outright: asked who the Little Red Hen's friends were, it
    # replied that the text does not say, with the list of friends in front of
    # it. When every content word of the question sits in ONE sentence, that
    # sentence is about the question, so the verdict is overruled.
    #
    # Reason questions are left alone: "why did Mara climb the steps" shares
    # all its words with the sentence that says she climbed them, and the
    # reason is still not written there. Short questions are left alone too,
    # since one or two shared words prove nothing.
    if g.label == NOT_STATED and not _REASON_Q.match(question or ""):
        q_terms = [t for t in key_terms(question) if t not in _META]
        if len(q_terms) >= 3:
            for i in candidate_ids:
                if all(_term_in(t, sentences[i - 1]) for t in q_terms):
                    g.notes.append(f"R2: every word of the question is in sentence {i} "
                                   f"-> NOT_STATED overruled")
                    g.label, g.rule, valid = STATED, "R2:question_in_one_sentence", [i]
                    break

    g.evidence_ids = sorted(valid)
    g.evidence = [sentences[i - 1] for i in g.evidence_ids]
    return g


# ---------------------------------------------------------------------------
# R3 ambiguity
# ---------------------------------------------------------------------------
SENSE_RULES = """You list what a word or phrase could mean in a reading passage.
Use the exact words given, spelled exactly as written. Do not change any letter (for example, "son" is not "sun").
First list EVERY common meaning the exact phrase can have (up to 3), including idioms and figurative meanings.
For each meaning, give the numbers of the sentences whose OTHER words support that meaning in this passage, or [] if no sentence does.
Set "defined" to true only if a sentence directly explains the meaning.
Output ONLY one JSON object: {"senses": [{"sense": "short meaning", "evidence": [sentence numbers], "defined": false}]}"""

# R3: fewer content words than this around the phrase (in the sentences shown) = too little
# context to choose between its meanings, whatever the model says.
MIN_CONTEXT_WORDS = 3

_WHAT_MEANS = re.compile(r"\bwhat\s+(?:does|do|is)\s+(.+?)\s+mean\b|\bmeaning\s+of\s+(.+?)[?.!]*$|"
                         r"\bibig\s+sabihin\s+(?:ng|nang)\s+(.+?)[?.!]*$|\bkahulugan\s+ng\s+(.+?)[?.!]*$", re.I)


def find_verbatim(phrase: str, passage: str) -> str:
    """The passage's own spelling of phrase (case/spacing tolerant), or "" if it isn't there."""
    words = [re.escape(w) for w in (phrase or "").split()]
    if not words:
        return ""
    m = re.search(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", passage or "", re.I)
    return m.group(0) if m else ""


def meaning_target(question: str, selection: str, passage: str) -> str:
    """The word or phrase whose meaning is asked, taken verbatim from the passage."""
    if selection and find_verbatim(selection, passage):
        return find_verbatim(selection, passage)
    candidates = re.findall(r'["“]([^"”]+)["”]', question or "")
    m = _WHAT_MEANS.search(question or "")
    if m:
        candidates.append(next(g for g in m.groups() if g))
    for c in candidates:
        c = re.sub(r"^(the\s+word|the\s+phrase|salitang|ang)\s+", "", c.strip(" \"'“”"), flags=re.I)
        hit = find_verbatim(c, passage)
        if hit:
            return hit
    return selection.strip() if selection else ""


def check_ambiguity(ask: Callable, target: str, numbered: list) -> Optional[list]:
    """R3 model step. Every meaning the model lists, with only the evidence ids that are valid
    (a meaning with no valid ids is unsupported). None if unparseable twice."""
    prompt = (f"Sentences:\n{_numbered(numbered)}\n\n"
              f"Phrase, spelled exactly: \"{target}\"\n\nJSON:")
    valid_ids = {i for i, _ in numbered}
    for attempt in range(2):
        raw = ask(SENSE_RULES, prompt + (_RETRY_NOTE if attempt else ""), None,
                  temperature=0.0, max_tokens=250, use_adapter=False)
        data = _parse_json(raw)
        if isinstance(data, dict) and isinstance(data.get("senses"), list):
            senses, seen = [], set()
            for s in data["senses"]:
                if not isinstance(s, dict):
                    continue
                text = str(s.get("sense", "")).strip()
                ids = sorted({i for i in _ids(s.get("evidence")) if i in valid_ids})
                if text and norm(text) not in seen:
                    seen.add(norm(text))
                    senses.append({"sense": text, "evidence": ids, "defined": bool(s.get("defined"))})
            return senses[:3]
    return None


_WHICH_MEANING_TAILS = ("which one fits the rest of the passage",
                        "alin ang bagay sa natitirang bahagi ng teksto")

def asked_which_meaning(coach_line: str) -> bool:
    """Was the coach's last turn the R3 question offering two meanings?

    R3 reaching AMBIGUOUS is the system recording that the text does not settle
    which meaning is right. One turn later the student picks one, the classifier
    calls that a success, and the coach answers "That's right!" about a thing
    nothing has decided. The two sentences are built from a template here, so the
    turn is recognisable from its ending.
    """
    low = (coach_line or "").lower()
    return any(tail in low for tail in _WHICH_MEANING_TAILS)


def short_sense(text: str, limit: int = 7) -> str:
    """A listed meaning, cut down to what the student needs to choose between.

    The model writes dictionary definitions, and the tail of one carries detail
    the passage has nothing to do with: "a device that produces light, typically
    electric or gas-powered" for a lighthouse lamp with a shutter. The head is
    the meaning; the rest is the dictionary talking, and it reaches the student
    unchecked because this reply is built here rather than written by the model.
    """
    text = (text or "").strip()
    if len(text.split()) <= limit:
        return text
    head = re.split(r"\s*[,;(]\s*", text)[0].strip()
    # Cutting mid-phrase reads worse than a long meaning, so a definition with
    # nothing to cut at is left alone.
    return head if len(head.split()) >= 2 else text


def context_words(target: str, numbered: list) -> list:
    """Content words in the shown sentences other than the phrase's own words."""
    own = set(_words(target))
    return [w for _, s in numbered for w in _words(s) if w not in own and w not in _STOP and len(w) > 2]


# ---------------------------------------------------------------------------
# The tree (R0 -> R3)
# ---------------------------------------------------------------------------
CLAIM_RULES = """You check a student's reply about a reading passage against numbered sentences from it.
Use ONLY those sentences, never outside knowledge. If the student answers yes or no, judge the statement they agree or disagree with in the coach's question.
Verdicts:
SUPPORTED: the sentences say it or clearly show it.
CONTRADICTED: a sentence says the opposite or something different (the student says it rained, but a sentence says it failed to rain).
NOT_MENTIONED: the sentences do not say either way.
NO_CLAIM: the student did not say anything about the story ("I don't know", "ok", a feeling about the task).
Output ONLY one JSON object: {"verdict": "SUPPORTED" or "CONTRADICTED" or "NOT_MENTIONED" or "NO_CLAIM", "evidence": [sentence numbers]}
As evidence, list the numbers of the sentences that support or contradict the reply. Use [] for NOT_MENTIONED and NO_CLAIM."""

_VERDICTS = ("SUPPORTED", "CONTRADICTED", "NOT_MENTIONED", "NO_CLAIM")


def check_claim(ask: Callable, answer: str, coach_line: str, sentences: list,
                selection: str = "") -> tuple:
    """R6 model step: does the text back what the student said? Returns (verdict, valid ids, ids
    shown). A verdict citing no sentence we showed it is NOT_MENTIONED."""
    terms = key_terms(answer)
    if len(terms) < 3:
        terms += key_terms(coach_line)      # "yes" / "the river" only make sense with the question
    cand = candidate_sentences(sentences, terms, selection, k=5, neighbours=False)
    if not cand:
        return "NOT_MENTIONED", [], []
    prompt = f"Sentences:\n{_numbered([(i, sentences[i - 1]) for i in cand])}\n\n"
    if coach_line:
        prompt += f"Coach's question: \"{coach_line[:300]}\"\n"
    prompt += f"Student's reply: \"{answer[:300]}\"\n\nJSON:"
    for attempt in range(2):
        data = _parse_json(ask(CLAIM_RULES, prompt + (_RETRY_NOTE if attempt else ""), None,
                               temperature=0.0, max_tokens=60, use_adapter=False))
        verdict = str(data.get("verdict", "")).strip().upper() if isinstance(data, dict) else ""
        if verdict in _VERDICTS:
            valid = [i for i in dict.fromkeys(_ids(data.get("evidence"))) if i in cand]
            if verdict in ("SUPPORTED", "CONTRADICTED") and not valid:
                verdict = "NOT_MENTIONED"
            return verdict, sorted(valid), cand
    return "NOT_MENTIONED", [], cand


FACT_RULES = """You check a reading coach's reply against numbered sentences from a reading passage.
Use ONLY those sentences, never outside knowledge. For each numbered reply line, choose one verdict:
SUPPORTED: the passage sentences say it, or it follows directly from what they say. Saying what a word means counts if the passage uses the word that way.
GUESS: the passage does not say it, but the sentences give a clue to it. A reason ("because", "so") or a feeling is a GUESS unless a sentence states that exact reason or feeling.
INVENTED: the line states something about the story (an event, a reason, a feeling, a place, what happens next) that the passage sentences do not say or show.
NOT_ABOUT_STORY: the line does not state anything about the story (encouragement, "keep reading", "look at this sentence", "the text doesn't say"), or it only repeats what the student said or asked.
A line about what a WORD means is NOT_ABOUT_STORY, not INVENTED: "'Come out' has several meanings", "a harbour is where boats shelter", "'daang' means street". A word's meaning is not an event in the story, so it needs no sentence to back it.
Repeating the student's own question back to them is NOT_ABOUT_STORY, even when the passage never answers that question.
For a line that is a question, choose one of these instead:
ON_TEXT: the passage sentences answer the question or give clues to its answer.
OFF_TEXT: the question asks about something the passage sentences never mention (for example, what is in front of a building when the passage only says what is behind it).
A question that offers choices ("in front of it or behind it?") is ON_TEXT when the passage shows which choice is right.
Output ONLY one JSON object: {"checks": [{"line": 1, "verdict": "SUPPORTED", "evidence": [sentence numbers]}]}
List every reply line once. As evidence, give the numbers of the passage sentences that support the line or answer the question ([] for INVENTED, OFF_TEXT and NOT_ABOUT_STORY)."""

_FACT_VERDICTS = ("SUPPORTED", "GUESS", "INVENTED", "NOT_ABOUT_STORY", "ON_TEXT", "OFF_TEXT")


# Words a hint question uses about reading itself, not about the story.
_META = set("""find look see read reading tell think mind means picture imagine words clue clues paragraph
sentences show shows describe explain happen happens happened part parts else
hanapin tingnan basahin isipin salita pangungusap talata""".split())


def hedged_throughout(line: str) -> bool:
    """Every clause of a guess sounds like one: "Mandy was nervous because she was alone, and the
    bridge might be dangerous" states the first part as fact."""
    clauses = re.split(r"\s*;\s*|,?\s+(?:and|at|pero)\s+", line)
    return all(is_hedged(c) for c in clauses if len(_words(c)) >= 3)


def reply_statements(reply: str) -> list:
    """The reply's sentences, statements and questions alike."""
    return split_sentences(reply)




def check_facts(ask: Callable, reply: str, sentences: list, selection: str = "",
                student: str = "") -> tuple:
    """R7 model step: which statements in the coach's reply does the passage not back?
    Returns (invented lines, trace note, unhedged guesses). A SUPPORTED or GUESS verdict needs a cited
    sentence we showed the model, and a GUESS must be worded as one; unparseable output
    removes nothing (the other rules still apply)."""
    lines = reply_statements(reply)
    if not lines or not sentences:
        return [], "R7: no statements to check", []
    cand = candidate_sentences(sentences, key_terms(" ".join(lines)), selection, k=8)
    if not cand:
        return [], "R7: no passage sentences share words with the reply", []
    prompt = (f"Passage sentences:\n{_numbered([(i, sentences[i - 1]) for i in cand])}\n\n"
              + (f"The student had said: \"{student[:300]}\"\n\n" if student else "")
              + f"Reply lines ({len(lines)}; give exactly {len(lines)} checks):\n" + "\n".join(f"({n}) {s}" for n, s in enumerate(lines, 1)) + "\n\nJSON:")
    for attempt in range(2):
        raw = ask(FACT_RULES, prompt + (_RETRY_NOTE if attempt else ""), None,
                  temperature=0.0, max_tokens=60 + 30 * len(lines), use_adapter=False)
        data = _parse_json(raw)
        checks = data.get("checks") if isinstance(data, dict) else None
        if not isinstance(checks, list):
            # Output cut off mid-list (the model ran on past the real lines): keep the complete entries.
            checks = [c for c in (_parse_json(m) for m in re.findall(r"\{[^{}]*\}", raw or "")) if c]
            checks = [c for c in checks if "line" in c] or None
        if checks:
            break
    else:
        return [], "R7: model output was not valid JSON twice -> nothing removed", []
    invented, unhedged, trace = [], [], []
    for c in checks:
        if not isinstance(c, dict):
            continue
        n = (_ids(c.get("line")) or [0])[0]
        verdict = str(c.get("verdict", "")).strip().upper()
        if not 1 <= n <= len(lines) or verdict not in _FACT_VERDICTS:
            continue
        valid = [i for i in _ids(c.get("evidence")) if i in cand]
        line = lines[n - 1]
        if line.rstrip().endswith("?"):
            # Questions need no citation. One is removed only when the model calls it off-text AND it
            # brings in a word the passage never uses ("in front of" when it only says "behind"), so a
            # fair hint question ("What is behind something?") isn't lost to a shaky verdict.
            new = [t for t in key_terms(line) if t not in _META and not any(_term_in(t, s) for s in sentences)]
            if verdict in ("OFF_TEXT", "INVENTED") and new:
                invented.append(line)
            trace.append(f"({n}) {verdict} new={new}")
            continue
        if verdict == "INVENTED" or (verdict in ("SUPPORTED", "GUESS") and not valid):
            invented.append(line)
        elif verdict == "GUESS" and not hedged_throughout(line):
            if _IMPERATIVE.match(line):
                pass                   # "Think about what shaking means." asks, it doesn't claim
            elif _CITES_TEXT.search(line):
                invented.append(line)  # "The text says 'hands trembled'": a guess can't be what the text says
            else:
                unhedged.append(line)  # the text gives a clue: keep it, worded as a guess
        trace.append(f"({n}) {verdict} {valid}")
    invented = list(dict.fromkeys(invented))
    unhedged = [u for u in dict.fromkeys(unhedged) if u not in invented]
    return invented, f"R7: {' '.join(trace)} -> removed {len(invented)}, hedged {len(unhedged)}", unhedged


_CITES_TEXT = re.compile(r"\b(the\s+(text|story|passage|sentence|paragraph)|it)\s+(says|said|tells|shows)\b|"
                         r"\bsinasabi\b|\bayon\s+sa\s+teksto\b|[\"“']", re.I)

_IMPERATIVE = re.compile(r"^\W*(think|look|read|try|see|notice|remember|find|check|imagine|picture|consider|"
                         r"let's|lets|go|keep|isipin|tingnan|basahin|hanapin|subukan|alalahanin)\b", re.I)

_LEADING_LINK = re.compile(r"^(but|and|so|also|then|pero|at|kaya|tapos)\b[,\s]*", re.I)


def apply_fact_check(reply: str, invented: list, unhedged: list, lang: str = "en") -> str:
    """The reply without its invented lines, with unhedged guesses marked as guesses, and no
    dangling "But..." where the sentence before it was removed."""
    out, dropped_before = [], False
    for s in split_sentences(reply):
        if s in invented:
            dropped_before = True
            continue
        if s in unhedged:
            s = _LEADING_LINK.sub("", s)                  # not "Maybe so she walked..."
            first = (_words(s) or [""])[0]
            s = ("Siguro " if lang == "fil" else "Maybe ") + (s[0].lower() + s[1:] if first in _STOP else s)
        elif dropped_before or not out:
            s = _LEADING_LINK.sub("", s)
            s = s[:1].upper() + s[1:]
        out.append(s)
        dropped_before = False
    return " ".join(out)


def answer_context(answer: str, passage: str, sentences: list, full_passage: bool = True,
                   earlier_missing: Optional[list] = None, ask: Optional[Callable] = None,
                   coach_line: str = "", selection: str = "") -> GroundingResult:
    """R0 'answer' branch, code only (no model call): the student answered rather than asked.

    Names in the answer that the passage never mentions are reported, and the sentences that
    contain the answer's words are handed to the coach, so a reply to "mandy" can use what
    the passage says about Mandy even when she is outside the highlighted paragraph.
    """
    g = GroundingResult(label=SKIPPED, rule="R0:answer", question=answer)
    if not answer:
        g.rule = "R0:not_a_question"
        return g
    names = names_in(answer, earlier_missing)
    if names and full_passage:
        g.missing_names = find_missing_names(names, passage)
        if g.missing_names:
            g.notes.append(f"R0: the student named {g.missing_names}, which the passage never mentions")
    g.evidence_ids = candidate_sentences(sentences, key_terms(answer) + [norm(n) for n in names], k=3,
                                         neighbours=False)
    g.evidence = [sentences[i - 1] for i in g.evidence_ids]
    if ask is not None and not g.missing_names:
        verdict, ids, shown = check_claim(ask, answer, coach_line, sentences, selection)
        g.claim = verdict
        g.notes.append(f"R6: claim {verdict} evidence={ids} (shown {shown})")
        if verdict == "CONTRADICTED":
            g.label, g.rule = CONTRADICTED, "R6:contradicted"
            g.evidence_ids, g.evidence = ids, [sentences[i - 1] for i in ids]
        elif verdict == "SUPPORTED":
            g.rule = "R6:supported"
    return g



def ground(ask: Callable, *, question: str, passage: str, selection: str = "",
           problem: Optional[str] = None, mode: Optional[str] = None,
           full_passage: bool = True, earlier: Optional[list] = None, coach_line: str = "",
           claim_check: bool = False) -> GroundingResult:
    """Run R0-R3 (and R6 for answers when claim_check is on) and return the decision with its trace.

    full_passage=False means only the paragraph around the selection was available; R1 is
    then skipped, because a name missing from one paragraph may be in another.
    earlier: the student's earlier messages in this thread, so a missing name asked about
    once is still caught when it comes back in lowercase or another phrasing.
    """
    q = (question or "").strip()
    sentences = split_sentences(passage)
    first_vocab_turn = mode == "debug" and problem == "vocab" and not q and bool(selection)
    if not sentences:
        return GroundingResult(label=SKIPPED, rule="R0:no_passage", question=q)
    earlier_missing = earlier_missing_names(earlier, passage) if full_passage else []
    if not (mode == "explain" or is_question(q) or first_vocab_turn):
        return answer_context(q, passage, sentences, full_passage, earlier_missing,
                              ask if claim_check else None, coach_line, selection)

    notes = []
    names = names_in(q, earlier_missing) if q else []
    if names and full_passage:
        missing = find_missing_names(names, passage)
        if missing:
            return GroundingResult(label=NOT_IN_TEXT, rule="R1:name_missing", missing_names=missing,
                                   question=q, notes=[f"R1: {missing} not in the passage; no model call"])
    elif names:
        notes.append("R1 skipped: only the paragraph around the selection was sent")

    terms = key_terms(q) + [norm(n) for n in names]
    cand = candidate_sentences(sentences, terms, selection)
    present = [n for n in names if not find_missing_names([n], passage)]
    term = present[0] if present else (selection if selection and len(selection.split()) <= 6 else "")
    if not cand:
        return GroundingResult(label=NOT_STATED, rule="R2:no_candidates", question=q, term=term,
                               notes=notes + ["R2: no sentence shares a key term with the question"])
    numbered = [(i, sentences[i - 1]) for i in cand]

    target = meaning_target(q, selection, passage) if (first_vocab_turn or is_meaning_question(q)) else ""
    if target and len(target.split()) <= 8:
        senses = check_ambiguity(ask, target, numbered)
        g = GroundingResult(question=q, target=target, term=term, notes=notes)
        if senses is not None:
            g.notes.append(f"R3: model listed {[(s['sense'], s['evidence']) for s in senses]}")
        if senses:
            known = set(_words(passage))
            respelled = [s["sense"] for s in senses if near_misses(s["sense"], target, known)]
            if respelled:
                g.notes.append(f"R3: dropped meanings that change the spelling of \"{target}\": {respelled}")
            senses = [s for s in senses if s["sense"] not in respelled]
        supported = [s for s in senses or [] if s["evidence"]]
        if senses is None:
            g.label, g.rule = AMBIGUOUS, "R3:unparseable"
            g.notes.append("R3: model output was not valid JSON twice -> AMBIGUOUS")
        elif len(context_words(target, numbered)) < MIN_CONTEXT_WORDS:
            g.label, g.rule, g.senses = AMBIGUOUS, "R3:no_context", senses if len(senses) >= 2 else []
            g.notes.append(f"R3: almost no words around \"{target}\" to choose a meaning -> AMBIGUOUS")
        elif len(supported) == 1:
            g.label = STATED if supported[0]["defined"] else INFERABLE
            g.rule, g.senses = "R3:one_sense", supported
            g.evidence_ids = supported[0]["evidence"]
        else:
            g.label, g.rule, g.senses = AMBIGUOUS, f"R3:{len(supported)}_senses", supported
            g.evidence_ids = sorted({i for s in supported for i in s["evidence"]})
        g.evidence = [sentences[i - 1] for i in g.evidence_ids]
        return g

    asked = q or f"What does \"{selection}\" mean or tell us?"
    g = verify(label_with_model(ask, asked, numbered, selection), cand, sentences, asked)
    g.question, g.term, g.notes = q, term, notes + g.notes
    return g


# ---------------------------------------------------------------------------
# R4 hedging, R5 output check
# ---------------------------------------------------------------------------
_HEDGE = re.compile(r"\b(might|may|maybe|perhaps|probably|likely|seems?|suggests?|clues?|hints?|"
                    r"siguro|marahil|baka|tila|mukhang|mukha|pahiwatig|palatandaan)\b", re.I)


def is_hedged(reply: str) -> bool:
    return bool(_HEDGE.search(reply or ""))


# A reply "says not stated" when it tells the student the text does not give the answer. The
# INFERABLE hedge ("doesn't say this directly, but ...") is a clue, not that, so "directly" clauses go first.
_DIRECT = re.compile(r"[^.?!]*\b(directly|direktang|direkta)\b[^.?!]*", re.I)
_NOT_STATED_SAID = re.compile(
    r"\b(doesn't|does not|didn't|did not|never)\s+(say|says|mention|mentions|tell|tells|state|states|explain)\b|"
    r"\bnot\s+(mentioned|stated|in the (text|passage|story|sentence))\b|"
    r"\b(isn't|is not|wasn't|was not)\s+(mentioned|stated|in the)\b|"
    r"\bno\s+(mention|information)\b|"
    r"\bhindi\s+(sinabi|nabanggit|binanggit|sinasabi|nakasaad|binabanggit)\b|"
    r"\bwalang\s+(nabanggit|binanggit|sinabi)\b",
    re.I)


# R6: a reply that tells the student they are right, used when the text contradicts them.
_AGREES = re.compile(r"^\W*(yes|yeah|yep|right|correct|exactly|good job|great job|well done|that's right|"
                     r"thats right|you're right|youre right|you got it|tama|oo|magaling|mismo)\b", re.I)


def agrees(reply: str) -> bool:
    return bool(_AGREES.match(reply or ""))


def says_not_stated(reply: str) -> bool:
    return bool(_NOT_STATED_SAID.search(_DIRECT.sub(" ", (reply or "").replace("’", "'"))))


def _text_of(turn) -> str:
    if isinstance(turn, str):
        return turn
    if isinstance(turn, dict):
        return turn.get("text", "")
    return getattr(turn, "text", "")


# Quoted words right after "means" are the coach's definition, not a quote from the text:
# "'Behind' means 'at the back of'."
_GLOSS_BEFORE = re.compile(r"\b(means?|meaning(\s+of)?|stands\s+for|another\s+word\s+for|or|"
                           r"ibig\s+sabihin(\s+ng)?(\s+ay)?|kahulugan(\s+ay)?|o)\s*[:,]?\s*$", re.I)
_QUOTED = re.compile(r'"([^"]+)"|(?<!\w)\'([^\']+?)\'(?!\w)')


def text_quotes(reply: str) -> list:
    """Quoted spans the reply presents as the text's words (definitions left out)."""
    plain = (reply or "").translate(_QUOTE_MAP)
    return [m.group(1) or m.group(2) for m in _QUOTED.finditer(plain)
            if not _GLOSS_BEFORE.search(plain[:m.start()])]


def unquote(reply: str, pieces: list) -> str:
    """Drop the quotation marks around quotes that aren't the text's words, keeping the words."""
    plain = (reply or "").translate(_QUOTE_MAP)
    bad = {p.strip()[:60] for p in pieces}
    def fix(m):
        inner = m.group(1) or m.group(2)
        return inner if any(b and b in inner for b in bad) else m.group(0)
    return _QUOTED.sub(fix, plain)


# ---------------------------------------------------------------------------
# R8 asking, R9 language, R10 repetition
#
# These three judge the reply as a coaching turn rather than against the
# passage, so they need no model call and run even with the passage rules off.
# ---------------------------------------------------------------------------
def asks_question(reply: str) -> bool:
    """R8. True when the reply actually asks the student something.

    A hint turn that states the answer is the failure the whole system exists
    to prevent, and nothing else catches it: a reply can be perfectly grounded
    in the passage and still just hand over the answer.
    """
    return any(s.rstrip().endswith("?") for s in split_sentences(reply))


_FIL_MARKER = re.compile(
    r"\b(ang|ng|mga|sa|ay|na|hindi|ko|po|ito|siya|nila|kami|tayo|ba|mo|ka|natin|niya|"
    r"kung|dahil|kasi|para|pero|lang|naman|yung|iyon|sina|kay)\b", re.I)


def filipino_markers(text: str) -> int:
    """How many DISTINCT Filipino function words the text uses."""
    return len({m.group(0).lower() for m in _FIL_MARKER.finditer(text or "")})


def wrong_language(reply: str, lang: str) -> bool:
    """R9. True when the reply is clearly not in the language that was chosen.

    Quoted spans are removed first: a coach replying in English about a
    Filipino passage quotes Filipino words, and that is correct, not drift.
    English is judged strictly, because drifting into Filipino on an English
    passage is the failure that actually happened. Filipino is judged loosely,
    because Taglish is allowed and a short reply may use no marker at all.
    """
    body = _QUOTED.sub(" ", (reply or "").translate(_QUOTE_MAP))
    markers = filipino_markers(body)
    if lang == "en":
        return markers >= 3
    if lang == "fil":
        return markers == 0 and len(_words(body)) >= 12
    return False


def repeats_any(reply: str, earlier: list, threshold: float = 0.8) -> bool:
    """R10. True when the reply is nearly the same as something already said.

    Compares word sets, so a reworded repeat still counts. The threshold is
    higher than the one used for re-explanations, because two different
    questions about one passage legitimately share most of their words.
    """
    words = set(_words(reply))
    if not words:
        return False
    for e in earlier or []:
        old = set(_words(e))
        if old and len(words & old) / len(words | old) >= threshold:
            return True
    return False


def check_reply(reply: str, passage: str, message: str = "", history=None, selection: str = "") -> list:
    """R5. Violations in a coach reply (empty list = OK).

    - every capitalized name and every number must appear in the passage, the student's
      message or the conversation history;
    - every quoted span of 2+ words must be the passage's own words in the same order
      (case, spacing, quote style and punctuation ignored);
    - a word one letter away from a word of the selection that the passage never uses is
      flagged as a misspelled target ("sun" for "son").
    """
    history = history or []
    sources = "\n".join([passage or "", message or ""] + [_text_of(t) for t in history])
    violations = []
    # Names may come from the conversation: the coach has to be able to say a
    # name back to the student who used it.
    for name in extract_names(reply, include_first=True):
        if find_missing_names([name], sources):
            violations.append(f"name:{name}")
    # A number the student raised may be echoed, so the coach can ask where the
    # text says it. What it may not do is AGREE with a number the passage never
    # gives: "the label year says 1989" coming back as "you're right, the label
    # is from 1989" turns the student's invention into a fact about the text.
    agreeing = agrees(reply) and not says_not_stated(reply)
    for num in dict.fromkeys(re.findall(r"\d+(?:[.,:]\d+)*", reply or "")):
        pattern = rf"(?<![\d.,]){re.escape(num)}(?!\d)"
        if re.search(pattern, passage or ""):
            continue
        if re.search(pattern, sources) and not agreeing:
            continue
        violations.append(f"number:{num}")
    passage_words = " ".join(_words(passage))
    student_words = " ".join(_words(message or ""))
    for quote in text_quotes(reply):
        if len(_words(quote)) < 2:
            continue
        for piece in re.split(r"\.{3}|…", quote):
            pw = " ".join(_words(piece))
            if pw and not (re.search(rf"(?<!\w){re.escape(pw)}(?!\w)", passage_words)
                           or (student_words and pw in student_words)):
                violations.append(f"quote:{piece.strip()[:60]}")
    for w, sel_word in near_misses(reply, selection, set(_words(sources))):
        violations.append(f"near_miss:{w}~{sel_word}")
    return violations


def off_target(reply: str, selection: str, passage: str) -> list:
    """R11. A short selection is what the turn is about, so the reply has to be
    about it. Flags the reply that never mentions the selected words and instead
    names a different proper noun out of the same sentence: a hint for "daang"
    that talks about "Ilog-Binundok".

    Only a selection of one to four words is judged, because a longer one is a
    stretch of text rather than a target, and only when the reply picks up
    another name: a hint that asks about the sentence without naming anything is
    left alone. The caller decides which turns this applies to.
    """
    if not reply or not selection or not passage:
        return []
    sel_words = _words(selection)
    if not 1 <= len(sel_words) <= 4:
        return []
    # Mentioned at all? _term_in matches "daan" against "daang" and back.
    if any(_term_in(w, reply) for w in sel_words if len(w) >= 3):
        return []
    if norm(selection) in norm(reply):
        return []
    hosts = [s for s in split_sentences(passage) if norm(selection) in norm(s)]
    if not hosts:
        return []
    sel_set = {w.lower() for w in sel_words}
    taken = []
    for host in hosts:
        # include_first, because the sentence's own first word can be the name the
        # reply drifted onto, and the passage's capitalisation is the evidence.
        for name in extract_names(host, include_first=True):
            if name.lower() in sel_set or any(p.lower() in sel_set for p in re.split(r"[-']", name)):
                continue
            if _has_word(name, reply) and name not in taken:
                taken.append(name)
    return [f"off_target:{n}" for n in taken]


def _bare_forms(w: str) -> set:
    """A word with a possessive or plural ending taken off, alongside itself.

    _words() keeps "sun's" and "suns" whole, so comparing lengths against "son"
    never happened and the near miss was missed. Both endings are stripped here
    so the comparison sees "sun".
    """
    forms = {w}
    for suffix in ("'s", "’s", "s'", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            forms.add(w[: -len(suffix)])
    return forms


def near_misses(text: str, selection: str, known: set) -> list:
    """Words of text that are one letter away from a word of a short selection and are not
    known words of the passage: "sun" for "son", and "the sun's light" too."""
    if len(_words(selection)) > 6:
        return []
    known_bare = {f for k in known for f in _bare_forms(k)}
    out = []
    seen = set()
    for sel_word in {w for w in _words(selection) if len(w) >= 3 and w.isalpha() and w not in _STOP}:
        for raw in set(_words(text)):
            for w in _bare_forms(raw):
                if (w not in known_bare and w not in _NOT_NAMES and len(w) == len(sel_word)
                        and w.isalpha() and sum(a != b for a, b in zip(w, sel_word)) == 1
                        and (raw, sel_word) not in seen):
                    seen.add((raw, sel_word))
                    out.append((raw, sel_word))
    return out


# ---------------------------------------------------------------------------
# Reply instructions and templates
# ---------------------------------------------------------------------------
TEMPLATES = {
    "en": {
        NOT_IN_TEXT: "The passage doesn't mention {name}. Let's look at who the passage does talk about — "
                     "who do you see in this part?",
        NOT_STATED: "The passage mentions {term}, but it never says {what}. What clues can you find?",
        "not_stated_plain": "The passage doesn't say {what}. What clues can you find in the text?",
        "hedge": "The text doesn't say this directly, but there's a clue: ",
        INFERABLE: "The text doesn't say this directly, but there's a clue in this part: \"{quote}\" "
                   "What might it tell you?",
        AMBIGUOUS: "\"{target}\" can mean {a} or {b}. Which one fits the rest of the passage?",
        "ambiguous_open": "The text doesn't give enough clues to be sure what \"{target}\" means here. "
                          "What do you think it could mean, and why?",
        STATED: "Look at this part of the text: \"{quote}\" What does it tell you?",
        "meaning": "Look at the words around \"{target}\": \"{quote}\" What do they tell you about what "
                   "\"{target}\" means here?",
        CONTRADICTED: "Let's check that against the text. It says: \"{quote}\" Read it again. "
                      "What does it tell you?",
        "generic": "Let's look at that part of the text again. What does it say?",
    },
    "fil": {
        NOT_IN_TEXT: "Walang nabanggit na {name} sa teksto. Tingnan natin kung sino ang binanggit — "
                     "sino ang nakikita mo sa bahaging ito?",
        NOT_STATED: "Nabanggit ang {term}, pero hindi sinabi ng teksto kung {what}. "
                    "Anong mga palatandaan ang makikita mo?",
        "not_stated_plain": "Hindi sinabi ng teksto kung {what}. Anong mga palatandaan ang makikita mo?",
        "hedge": "Hindi ito direktang sinabi ng teksto, pero may palatandaan: ",
        INFERABLE: "Hindi ito direktang sinabi ng teksto, pero may palatandaan sa bahaging ito: \"{quote}\" "
                   "Ano kaya ang ipinahihiwatig nito?",
        AMBIGUOUS: "Ang \"{target}\" ay maaaring mangahulugang {a} o {b}. "
                   "Alin ang bagay sa natitirang bahagi ng teksto?",
        "ambiguous_open": "Hindi sapat ang palatandaan sa teksto para matiyak ang ibig sabihin ng \"{target}\" "
                          "dito. Ano sa tingin mo ang maaaring kahulugan nito, at bakit?",
        STATED: "Tingnan ang bahaging ito ng teksto: \"{quote}\" Ano ang sinasabi nito?",
        "meaning": "Tingnan ang mga salita sa paligid ng \"{target}\": \"{quote}\" Ano ang sinasabi nito "
                   "tungkol sa ibig sabihin ng \"{target}\" dito?",
        CONTRADICTED: "Balikan natin ang teksto. Sinasabi nito: \"{quote}\" Basahin mo ulit. "
                      "Ano ang sinasabi nito?",
        "generic": "Balikan natin ang bahaging iyon. Ano ang sinasabi ng teksto?",
    },
}


def question_gist(question: str, lang: str) -> str:
    """"Who is Michelle?" -> "who Michelle is" (EN) / "sino si Michelle" (FIL)."""
    s = (question or "").strip().rstrip("?").strip()
    if not s:
        return "iyon" if lang == "fil" else "the answer to that"
    if lang == "fil":
        return s[0].lower() + s[1:]
    m = re.match(r"(?i)^(who|what|where|when|whose)\s+(is|was|are|were)\s+(.+)$", s)
    if m:
        return f"{m[1].lower()} {m[3]} {m[2].lower()}"
    return "the answer to that"


def _short_quote(sentence: str, max_words: int = 14, around: str = "") -> str:
    """At most max_words of the sentence, centred on `around` when the sentence contains it
    (so a quote for "behind" doesn't stop before the word)."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    start = 0
    target = _words(around)
    if target:
        low = [norm(w).strip(".,;:!?\"'()") for w in words]
        hit = next((i for i in range(len(low)) if low[i:i + len(target)] == target), None)
        if hit is not None:
            start = max(0, min(hit - max_words // 2, len(words) - max_words))
    piece = " ".join(words[start:start + max_words])
    return ("…" if start else "") + piece + ("…" if start + max_words < len(words) else "")


def fallback_reply(g: GroundingResult, lang: str) -> str:
    """A reply built only from verified parts, used when the model's reply fails R5."""
    t = TEMPLATES["fil" if lang == "fil" else "en"]
    what = question_gist(g.question, lang)
    if g.label == NOT_IN_TEXT or (g.label == SKIPPED and g.missing_names):
        return t[NOT_IN_TEXT].format(name=", ".join(g.missing_names))
    if g.label == NOT_STATED:
        return (t[NOT_STATED].format(term=g.term, what=what) if g.term
                else t["not_stated_plain"].format(what=what))
    if g.label == AMBIGUOUS:
        if len(g.senses) >= 2:
            return t[AMBIGUOUS].format(target=g.target, a=short_sense(g.senses[0]["sense"]),
                                       b=short_sense(g.senses[1]["sense"]))
        if g.target:
            return t["ambiguous_open"].format(target=g.target)
    if is_meaning(g) and g.evidence:
        return t["meaning"].format(target=g.target, quote=_short_quote(g.evidence[0], around=g.target))
    if g.label in (STATED, INFERABLE, CONTRADICTED) and g.evidence:
        return t[g.label].format(quote=_short_quote(g.evidence[0], around=g.term))
    return t["generic"]


def is_meaning(g: Optional[GroundingResult]) -> bool:
    """R3 found one meaning for a word. Working a word's meaning out from context is the normal
    reading move, not a story fact the text "doesn't say", so it gets no R4 hedge."""
    return g is not None and g.rule.startswith("R3") and g.label in (STATED, INFERABLE) and bool(g.target)


def branch_instruction(g: GroundingResult, explaining: bool) -> str:
    """Extra sentence(s) for the coach's 'Instruction for this turn'."""
    spell = f" Use the exact words \"{g.target}\", spelled exactly as in the text." if g.target else ""
    if is_meaning(g):
        if explaining:
            return f"Explain what \"{g.target}\" means here, using the key sentences below." + spell
        return (f"The meaning of \"{g.target}\" can be worked out from the words around it in the key sentences "
                f"below. Point the student to those words (quote a few exact words) and ask ONE question that "
                f"helps them work out the meaning. Do NOT say that the text doesn't say it." + spell)
    if g.label == CONTRADICTED:
        said = g.question[:200]
        if explaining:
            return (f"The student said \"{said}\", but the key sentences below say something different. "
                    f"Gently correct this in your explanation, based only on those sentences.")
        return (f"The student said \"{said}\", but the key sentences below say something different. Do NOT "
                f"agree or say they are right. Gently say the text says something else, quote a few exact "
                f"words from that sentence, and ask them to read it again.")
    if g.label == SKIPPED:
        if g.missing_names:
            names = ", ".join(g.missing_names)
            return (f"The student mentioned {names}, but the passage never mentions {names}. Say so gently, "
                    f"do NOT invent who or what that is, and point them back to the people or things the "
                    f"passage does mention.")
        if g.evidence:
            return ("Respond to what the student said using the key sentences below, which are where the "
                    "passage mentions it. If you quote, quote only from them, word for word.")
        return ""
    if g.label == STATED:
        if explaining:
            return "Base your explanation only on the key sentences below." + spell
        return ("The text states the answer in the key sentences below. Do not give the answer outright: point the "
                "student to that sentence (quote a few exact words from it) and ask ONE question that helps "
                "them find the answer there." + spell)
    if g.label == INFERABLE:
        if explaining:
            return ("The text does not say this directly. Explain it as a likely inference, not a fact (use words "
                    "like \"probably\" or \"seems\"), based only on the key sentences below." + spell)
        return ("The text does not say this directly, but the key sentences below give a clue. Say that the text "
                "doesn't say it directly, point to the clue (quote a few exact words from it), and ask ONE "
                "question. Use words like \"might\" or \"seems\"." + spell)
    if g.label == NOT_STATED:
        return ("The passage does not answer this question. Say clearly that the text doesn't say it. Do NOT "
                "guess or invent an answer, names or details. Ask what clues the student can find in the text.")
    if g.label == AMBIGUOUS:
        if len(g.senses) >= 2:
            meanings = "; ".join(s["sense"] for s in g.senses)
            return (f"The exact words \"{g.target}\" could mean: {meanings}. The text does not make clear "
                    f"which one. Present these meanings neutrally, do NOT pick one, and ask which one fits "
                    f"the rest of the passage.{spell}")
        return (f"The text does not give enough clues to be sure what \"{g.target}\" means here. Say so, and "
                f"ask the student what it could mean and why.{spell}")
    return ""