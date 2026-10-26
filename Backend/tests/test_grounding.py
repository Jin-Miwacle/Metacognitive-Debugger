"""Grounding rules tests. No GPU or model: the model call is a fake that returns canned JSON.

Run from the project root:  Backend\\.venv\\Scripts\\python.exe -m pytest Backend\\tests -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grounding as gr  # noqa: E402

PASSAGE = (
    "Paolo waited until dinner to tell his family the news. His mother, Aling Rosa, set down the rice "
    "and looked at him. His sister Mika stopped eating.\n\n"
    "\"I passed the scholarship exam,\" Paolo said. \"I will study in Manila next year.\" Aling Rosa "
    "covered her mouth with both hands. Her eyes filled with tears.\n\n"
    "Michelle, who had come with Paolo, smiled and squeezed his hand. Mika ran around the table and "
    "hugged her brother."
)
# Sentence numbers in PASSAGE, for reference:
#  1 Paolo waited ...   2 His mother, Aling Rosa, ...   3 His sister Mika stopped eating.
#  4 "I passed the scholarship exam," Paolo said.   5 "I will study in Manila next year."
#  6 Aling Rosa covered her mouth ...   7 Her eyes filled with tears.
#  8 Michelle, who had come with Paolo, ...   9 Mika ran around the table ...


class FakeModel:
    """Stands in for main.ask_sealion. Returns the queued outputs in order and records calls."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, system, prompt, history=None, *, temperature, max_tokens, use_adapter=True):
        self.calls.append({"system": system, "prompt": prompt, "temperature": temperature,
                           "use_adapter": use_adapter})
        out = self.outputs.pop(0)
        return out if isinstance(out, str) else json.dumps(out)


def test_split_sentences_numbers_and_keeps_quotes():
    s = gr.split_sentences(PASSAGE)
    assert len(s) == 9
    assert s[3] == "\"I passed the scholarship exam,\" Paolo said."
    assert s[7].startswith("Michelle, who had come")
    assert gr.split_sentences("Mr. Cruz left. Gng. Santos stayed.") == ["Mr. Cruz left.", "Gng. Santos stayed."]


# 1. Name not in the passage -> NOT_IN_TEXT via R1, no model call.
def test_name_not_in_text_skips_model():
    fake = FakeModel()
    passage = PASSAGE.replace("Michelle, who had come with Paolo,", "A friend")
    g = gr.ground(fake, question="Who is Michelle?", passage=passage, selection="smiled")
    assert g.label == gr.NOT_IN_TEXT
    assert g.rule == "R1:name_missing"
    assert g.missing_names == ["Michelle"]
    assert fake.calls == []
    assert "doesn't mention Michelle" in gr.fallback_reply(g, "en")


# 2. Michelle is in the text but her role is not; the model claims STATED with a bad id.
def test_stated_with_invalid_evidence_is_downgraded():
    fake = FakeModel({"label": "STATED", "evidence": [42]})
    g = gr.ground(fake, question="Who is Michelle?", passage=PASSAGE, selection="Michelle")
    assert g.label == gr.NOT_STATED
    assert g.rule == "R2:downgraded"
    assert g.evidence_ids == []
    call = fake.calls[0]
    assert call["use_adapter"] is False and call["temperature"] == 0.0
    assert "[8] Michelle, who had come with Paolo" in call["prompt"]
    reply = gr.fallback_reply(g, "en")
    assert reply == "The passage mentions Michelle, but it never says who Michelle is. What clues can you find?"


# 3. Explicitly written answer -> STATED with the verified sentence.
def test_stated_with_valid_evidence():
    fake = FakeModel('Sure: {"label": "STATED", "evidence": ["5"]}')
    g = gr.ground(fake, question="Where will Paolo study?", passage=PASSAGE, selection="")
    assert g.label == gr.STATED and g.rule == "R2:model"
    assert g.evidence_ids == [5]
    assert g.evidence == ["\"I will study in Manila next year.\""]


# 4. Inference question -> INFERABLE; an unhedged reply is caught by R4.
def test_inferable_and_hedging():
    fake = FakeModel({"answer": "happy and proud", "label": "INFERABLE", "evidence": [6, 7]})
    g = gr.ground(fake, question="How does Aling Rosa feel about the news?", passage=PASSAGE,
                  selection="covered her mouth with both hands")
    assert g.label == gr.INFERABLE
    assert g.evidence_ids == [6, 7]
    assert not gr.is_hedged("Aling Rosa is happy and proud of Paolo.")
    assert gr.is_hedged("Aling Rosa might be happy. What clue do you see?")
    assert gr.is_hedged("Mukhang masaya si Aling Rosa.")
    fb = gr.fallback_reply(g, "en")
    assert fb.startswith("The text doesn't say this directly, but there's a clue in this part: ")
    assert gr.check_reply(fb, PASSAGE) == [] and gr.is_hedged(fb)


def test_inferable_upgraded_when_the_answer_is_written_in_the_text():
    # "I will study in Manila" is a separate sentence from "Paolo said", so the model calls it INFERABLE.
    fake = FakeModel({"answer": "in Manila", "label": "INFERABLE", "evidence": [4, 5]})
    g = gr.ground(fake, question="Where will Paolo study?", passage=PASSAGE)
    assert g.label == gr.STATED and g.rule == "R2:answer_in_text"


def test_why_questions_are_not_upgraded():
    # The cause is written down, but that it is the reason for the hug is an inference.
    fake = FakeModel({"answer": "He passed the scholarship exam", "label": "INFERABLE", "evidence": [4, 9]})
    g = gr.ground(fake, question="Why did Mika hug her brother?", passage=PASSAGE, selection="hugged her brother")
    assert g.label == gr.INFERABLE and g.rule == "R2:model"


def test_says_not_stated():
    assert gr.says_not_stated("The passage doesn't say how old Paolo is.")
    assert gr.says_not_stated("Hindi sinabi ng teksto kung ilan sila.")
    assert gr.says_not_stated("Michelle isn’t mentioned here.")
    assert not gr.says_not_stated("The text doesn't say this directly, but there's a clue.")
    assert not gr.says_not_stated("Look at the second sentence. What does it tell you?")
    g = gr.GroundingResult(label=gr.NOT_STATED, question="How old is Paolo?", term="Paolo")
    assert gr.says_not_stated(gr.fallback_reply(g, "en")) and gr.says_not_stated(gr.fallback_reply(g, "fil"))


# 5. "The son comes out" on its own: two senses, not enough context -> AMBIGUOUS; "son" stays "son".
def test_son_comes_out_is_ambiguous():
    passage = "The son comes out."
    fake = FakeModel({"senses": [
        {"sense": "he leaves a room or a house", "evidence": [1], "defined": False},
        {"sense": "he tells people he is gay", "evidence": [1], "defined": False},
        {"sense": "the sun rises", "evidence": [7]},  # changes "son" to "sun" -> dropped
    ]})
    g = gr.ground(fake, question="What does \"The son comes out\" mean?", passage=passage,
                  selection="The son comes out")
    assert g.label == gr.AMBIGUOUS and g.rule == "R3:no_context"
    assert g.target == "The son comes out"
    assert "son" in g.target and "sun" not in g.target.lower()
    assert [s["sense"] for s in g.senses] == ["he leaves a room or a house", "he tells people he is gay"]
    assert 'spelled exactly: "The son comes out"' in fake.calls[0]["prompt"]
    assert "The son comes out" in gr.branch_instruction(g, explaining=False)
    # R5 flags a reply that turns "son" into "sun".
    assert "near_miss:sun~son" in gr.check_reply("When the sun comes out, it is morning.", passage,
                                                 selection="The son comes out")


def test_context_picks_one_meaning():
    passage = ("The son comes out of the house with his school bag. He waves goodbye to his mother "
               "and walks to the bus stop.")
    fake = FakeModel({"senses": [
        {"sense": "he leaves the house", "evidence": [1, 2]},
        {"sense": "he tells people he is gay", "evidence": []},
    ]})
    g = gr.ground(fake, question="What does this mean?", passage=passage, selection="The son comes out")
    assert g.label == gr.INFERABLE and g.rule == "R3:one_sense"
    assert [s["sense"] for s in g.senses] == ["he leaves the house"]


def test_first_vocab_turn_runs_r3_without_a_typed_question():
    fake = FakeModel({"senses": [{"sense": "the time after sunset", "evidence": [1], "defined": False}]})
    g = gr.ground(fake, question="", passage="Umuwi kami nang takipsilim, nang madilim na ang langit.",
                  selection="takipsilim", problem="vocab", mode="debug")
    assert g.label == gr.INFERABLE and g.rule == "R3:one_sense"
    assert g.target == "takipsilim"


# 6. Filipino person markers.
def test_filipino_markers():
    assert gr.extract_names("Sino si Michelle?", lowercase_hints=True) == ["Michelle"]
    assert gr.extract_names("Ano ang sinabi niya kay Michelle?", lowercase_hints=True) == ["Michelle"]
    assert gr.extract_names("sino si michelle", lowercase_hints=True) == ["Michelle"]
    assert gr.extract_names("who is michelle", lowercase_hints=True) == ["Michelle"]
    assert gr.extract_names("who is she?", lowercase_hints=True) == []
    assert gr.find_missing_names(gr.extract_names("who is mang tomas?", lowercase_hints=True),
                                 PASSAGE) == ["Mang Tomas"]
    assert gr.find_missing_names(["Michelle"], PASSAGE) == []
    assert gr.find_missing_names(["Michelle Cruz"], PASSAGE) == []   # any part of the name counts
    assert gr.find_missing_names(["Aling Nena"], PASSAGE) == ["Aling Nena"]
    fake = FakeModel()
    g = gr.ground(fake, question="Sino si Bea?", passage=PASSAGE)
    assert g.label == gr.NOT_IN_TEXT and g.missing_names == ["Bea"] and fake.calls == []
    assert gr.fallback_reply(g, "fil").startswith("Walang nabanggit na Bea sa teksto.")


# 7. R5 output check.
def test_check_reply_flags_unsupported_names_numbers_and_quotes():
    ok = "Look at the part where \"Aling Rosa covered her mouth\". What might she feel?"
    assert gr.check_reply(ok, PASSAGE) == []
    assert "name:Michelle" not in gr.check_reply("Think about how Michelle reacts.", PASSAGE)
    v = gr.check_reply("Think about how Carlo reacts.", PASSAGE)
    assert v == ["name:Carlo"]
    assert gr.check_reply("He will study for 4 years.", PASSAGE) == ["number:4"]
    assert gr.check_reply("He will study for 4 years.", PASSAGE, message="is it 4 years?") == []
    bad_quote = gr.check_reply("The text says \"Paolo cried at the table\". Why?", PASSAGE)
    assert bad_quote == ["quote:Paolo cried at the table"]
    # single quotes count too; apostrophes do not
    assert gr.check_reply("The next sentence says 'Paolo cried at the table'.", PASSAGE) == \
        ["quote:Paolo cried at the table"]
    assert gr.check_reply("Look at Paolo's words: 'I will study in Manila'. What's next?", PASSAGE) == []
    # case, curly quotes and punctuation are ignored; the words and their order are not
    assert gr.check_reply("It says “i passed the scholarship exam” — what does that tell you?", PASSAGE) == []
    # names from the conversation history are allowed
    assert gr.check_reply("What did Carlo say?", PASSAGE, history=[{"role": "student", "text": "Carlo said hi"}]) == []


# 8. R0: a plain answer skips R1-R3.
def test_non_questions_skip_the_tree():
    fake = FakeModel()
    for msg in ("takipsilim", "i dont know", "Michelle is his girlfriend"):
        g = gr.ground(fake, question=msg, passage=PASSAGE, selection="Michelle", mode="followup")
        assert g.label == gr.SKIPPED and g.rule == "R0:answer"
    assert fake.calls == []
    assert gr.is_question("sino si Michelle")
    assert gr.is_question("Anong ibig sabihin nito")
    assert not gr.is_question("i dont know")
    assert not gr.is_question("her grandmother?")        # hesitant answer
    assert gr.is_question("is she his sister or his friend?")
    assert gr.is_question("si Mika ba kung sino?")


def test_answers_get_the_sentences_they_refer_to():
    # "who is ethan?" -> "...who do you see in the passage?" -> "mika": Mika is outside the
    # highlighted paragraph, so the coach needs the sentences about her.
    fake = FakeModel()
    g = gr.ground(fake, question="mika", passage=PASSAGE, selection="smiled", mode="followup")
    assert g.label == gr.SKIPPED and g.rule == "R0:answer" and fake.calls == []
    assert g.evidence_ids == [3, 9]
    assert all("Mika" in s for s in g.evidence)
    assert "key sentences" in gr.branch_instruction(g, explaining=False)


def test_answers_naming_someone_not_in_the_text():
    g = gr.ground(FakeModel(), question="I think it is his friend Carlo", passage=PASSAGE, mode="followup")
    assert g.missing_names == ["Carlo"]
    assert "never mentions Carlo" in gr.branch_instruction(g, explaining=False)
    assert gr.fallback_reply(g, "en").startswith("The passage doesn't mention Carlo.")
    # With only one paragraph sent, a missing name may be elsewhere: not reported.
    g = gr.ground(FakeModel(), question="his friend Carlo", passage=PASSAGE, mode="followup", full_passage=False)
    assert g.missing_names == []


# 9. Malformed JSON -> one retry -> NOT_STATED.
def test_malformed_json_retries_once_then_not_stated():
    fake = FakeModel("label: STATED", "{oops")
    g = gr.ground(fake, question="Where will Paolo study?", passage=PASSAGE)
    assert len(fake.calls) == 2
    assert "not valid JSON" in fake.calls[1]["prompt"]
    assert g.label == gr.NOT_STATED and g.rule == "R2:unparseable"

    fake = FakeModel("nope", {"label": "STATED", "evidence": [5]})
    g = gr.ground(fake, question="Where will Paolo study?", passage=PASSAGE)
    assert len(fake.calls) == 2 and g.label == gr.STATED


def test_r1_is_skipped_without_the_full_passage():
    fake = FakeModel({"label": "NOT_STATED", "evidence": []})
    g = gr.ground(fake, question="Who is Michelle?", passage="Mika ran around the table.",
                  selection="Mika ran", full_passage=False)
    assert g.label == gr.NOT_STATED and len(fake.calls) == 1
    assert any("R1 skipped" in n for n in g.notes)


# Yes/no questions are questions; a short hesitant answer still is not.
def test_yes_no_question_detection():
    assert gr.is_question("is ethan her brother?")
    assert gr.is_question("did it rain?")
    assert not gr.is_question("her grandmother?")


# A missing name stays caught for the rest of the thread, in any case and phrasing.
def test_missing_name_remembered_across_turns():
    fake = FakeModel()
    g = gr.ground(fake, question="is ethan her brother?", passage=PASSAGE, selection="hugged her brother",
                  earlier=["Who is Ethan?"])
    assert g.label == gr.NOT_IN_TEXT and g.missing_names == ["Ethan"]
    g = gr.ground(fake, question="where is ethan?", passage=PASSAGE, selection="hugged her brother")
    assert g.label == gr.NOT_IN_TEXT and g.missing_names == ["Ethan"]
    g = gr.answer_context("ethan was there", PASSAGE, gr.split_sentences(PASSAGE),
                          earlier_missing=["Ethan"])
    assert g.missing_names == ["Ethan"]
    assert fake.calls == []


DRY = ("The farmers looked at the sky every day. It failed to rain for three months. "
       "The rice fields turned brown and dry.")


# R6: an answer the text contradicts is labelled CONTRADICTED, with the cited sentence.
def test_contradicted_answer():
    fake = FakeModel({"verdict": "CONTRADICTED", "evidence": [2]})
    g = gr.ground(fake, question="it's raining", passage=DRY, selection="turned brown",
                  coach_line="What is the weather like?", claim_check=True)
    assert g.label == gr.CONTRADICTED and g.rule == "R6:contradicted"
    assert g.evidence == ["It failed to rain for three months."]
    assert "failed to rain" in fake.calls[0]["prompt"]
    assert fake.calls[0]["use_adapter"] is False and fake.calls[0]["temperature"] == 0.0
    assert "failed to rain" in gr.fallback_reply(g, "en")
    assert "Do NOT agree" in gr.branch_instruction(g, explaining=False)
    assert gr.agrees("Yes, it is raining!") and not gr.agrees("Let's check that against the text.")


# R6: CONTRADICTED citing a sentence that wasn't shown is not trusted.
def test_contradicted_without_valid_evidence_is_ignored():
    fake = FakeModel({"verdict": "CONTRADICTED", "evidence": [42]})
    g = gr.ground(fake, question="it's raining", passage=DRY, claim_check=True)
    assert g.label == gr.SKIPPED and g.rule == "R0:answer"


# R6 is off unless asked for, and is skipped when the answer names someone not in the text.
def test_claim_check_only_when_asked():
    fake = FakeModel()
    gr.ground(fake, question="it's raining", passage=DRY)
    gr.ground(fake, question="maybe Ethan made it rain", passage=DRY, claim_check=True)
    assert fake.calls == []


# R7: statements in the coach's reply that the text doesn't back are removed; questions aren't checked.
def test_check_facts_removes_invented_lines():
    reply = ("Mika hugged her brother. She was proud because he will be a doctor. "
             "Maybe she was happy for him. Keep reading! What did Aling Rosa do?")
    fake = FakeModel({"checks": [
        {"line": 1, "verdict": "SUPPORTED", "evidence": [9]},
        {"line": 2, "verdict": "INVENTED", "evidence": []},
        {"line": 3, "verdict": "GUESS", "evidence": [9]},
        {"line": 4, "verdict": "NOT_ABOUT_STORY", "evidence": []},
        {"line": 5, "verdict": "ON_TEXT", "evidence": [6]}]})
    invented, note, _ = gr.check_facts(fake, reply, gr.split_sentences(PASSAGE), "hugged her brother")
    assert invented == ["She was proud because he will be a doctor."]
    assert "(5) What did Aling Rosa do?" in fake.calls[0]["prompt"]
    assert fake.calls[0]["use_adapter"] is False and fake.calls[0]["temperature"] == 0.0


# R7: a question about something the text never mentions is removed too.
def test_check_facts_removes_off_text_question():
    reply = "Paolo told his family the news. What did Paolo eat for breakfast?"
    fake = FakeModel({"checks": [
        {"line": 1, "verdict": "SUPPORTED", "evidence": [1]},
        {"line": 2, "verdict": "OFF_TEXT", "evidence": []}]})
    invented, _, unhedged = gr.check_facts(fake, reply, gr.split_sentences(PASSAGE), "the news")
    assert invented == ["What did Paolo eat for breakfast?"]


# R5: quoted definitions are not quotes from the text; a bad quote can be fixed by unquoting it.
def test_quoted_definition_is_not_a_text_quote():
    reply = "'Behind' means 'at the back of'. Look at \"his sister Mika stopped\"."
    assert gr.check_reply(reply, PASSAGE) == []
    bad = "The text says \"Mika was very proud of him\"."
    assert gr.check_reply(bad, PASSAGE) == ["quote:Mika was very proud of him"]
    assert gr.unquote(bad, ["Mika was very proud of him"]) == "The text says Mika was very proud of him."


# R7: SUPPORTED needs a sentence we showed it, and a GUESS must be worded as a guess.
def test_check_facts_verifies_evidence_and_hedging():
    reply = "Mika hugged her brother. She was happy for him."
    fake = FakeModel({"checks": [
        {"line": 1, "verdict": "SUPPORTED", "evidence": [42]},
        {"line": 2, "verdict": "GUESS", "evidence": [9]}]})
    invented, _, unhedged = gr.check_facts(fake, reply, gr.split_sentences(PASSAGE), "hugged her brother")
    assert invented == ["Mika hugged her brother."]
    assert unhedged == ["She was happy for him."]
    fixed = gr.apply_fact_check("Mika hugged her brother. She was happy for him. But she sat down.",
                                invented, unhedged)
    assert fixed == "Maybe she was happy for him. But she sat down."
    assert gr.apply_fact_check("Paolo left. But he came back.", ["Paolo left."], []) == "He came back."
    assert gr.apply_fact_check("Mika was proud.", [], ["Mika was proud."]) == "Maybe Mika was proud."


# R7: unparseable output removes nothing.
def test_check_facts_bad_json_removes_nothing():
    fake = FakeModel("not json", "still not json")
    invented, note, _ = gr.check_facts(fake, "Mika hugged her brother.", gr.split_sentences(PASSAGE))
    assert invented == [] and "not valid JSON" in note


# R3 meanings get no R4 hedge, and the fallback quotes the words around the target.
def test_meaning_result_has_no_hedge_and_quotes_around_target():
    sentence = ("When the rains failed for the second year, the community garden shrank to a single "
                "raised bed behind the old schoolhouse.")
    g = gr.GroundingResult(label=gr.INFERABLE, rule="R3:one_sense", target="behind", evidence=[sentence])
    assert gr.is_meaning(g)
    assert "doesn't say" not in gr.branch_instruction(g, explaining=False).replace("Do NOT say that the text doesn't say it", "")
    reply = gr.fallback_reply(g, "en")
    assert "raised bed behind the old schoolhouse" in reply and "doesn't say" not in reply
    assert not gr.is_meaning(gr.GroundingResult(label=gr.INFERABLE, rule="R2:model"))


# R7: output cut off mid-list still yields the complete entries (the model ran on past the real lines).
def test_check_facts_salvages_truncated_output():
    raw = ('```json\n{"checks": [{"line": 1, "verdict": "INVENTED", "evidence": []}, '
           '{"line": 2, "verdict": "SUPPORTED", "evidence": [3]}, {"line": 3, "')
    fake = FakeModel(raw)
    invented, _, unhedged = gr.check_facts(fake, "Mika was angry at Paolo.", gr.split_sentences(PASSAGE), "stopped eating")
    assert invented == ["Mika was angry at Paolo."]
    assert len(fake.calls) == 1


# R7: a guess that claims to be what the text says is removed, not hedged; "Maybe" drops a leading "So".
def test_guess_citing_the_text_is_removed_and_hedge_drops_link_word():
    reply = "The text says Mika 'felt proud'. So she hugged her brother tightly."
    fake = FakeModel({"checks": [
        {"line": 1, "verdict": "GUESS", "evidence": [9]},
        {"line": 2, "verdict": "GUESS", "evidence": [9]}]})
    invented, _, unhedged = gr.check_facts(fake, reply, gr.split_sentences(PASSAGE), "hugged her brother")
    assert invented == ["The text says Mika 'felt proud'."]
    assert gr.apply_fact_check(reply, invented, unhedged) == "Maybe she hugged her brother tightly."
    assert gr.check_reply("The text says 'felt proud'.", PASSAGE) == ["quote:felt proud"]


# R7: an instruction to the student is not a claim, so it is never turned into "Maybe Think...".
def test_instruction_is_not_hedged():
    reply = "Think about what hugging means. Mika was proud."
    fake = FakeModel({"checks": [
        {"line": 1, "verdict": "GUESS", "evidence": [9]},
        {"line": 2, "verdict": "GUESS", "evidence": [9]}]})
    invented, _, unhedged = gr.check_facts(fake, reply, gr.split_sentences(PASSAGE), "hugged her brother")
    assert invented == [] and unhedged == ["Mika was proud."]
    assert gr.apply_fact_check(reply, invented, unhedged) == "Think about what hugging means. Maybe Mika was proud."
