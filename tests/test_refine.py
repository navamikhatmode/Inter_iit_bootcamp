from meeting_ai.refine import check_refinement, refine_transcript, strip_preamble
from meeting_ai.textutils import chunk_sentences, collapse_repetitions


class FakeLLM:
    def __init__(self, fn):
        self.fn = fn

    def generate(self, system, user, max_new_tokens=1024):
        return self.fn(user)


RAW = "We will not deploy on Friday. The cube flow cluster needs 8 nodes. Maybe we upgrade later."


def test_good_refinement_accepted():
    assert check_refinement(RAW, RAW.replace("cube flow", "Kubeflow")) == []


def test_dropped_negation_rejected():
    assert any("negation" in p for p in check_refinement(RAW, RAW.replace("not ", "")))


def test_changed_number_rejected():
    assert any("number" in p for p in check_refinement(RAW, RAW.replace("8", "9")))


def test_summarising_rejected():
    assert check_refinement(RAW, "Deploy later.")


def test_fallback_to_raw_when_model_misbehaves():
    refined, report = refine_transcript(FakeLLM(lambda u: "Summary: nothing."), RAW)
    assert refined == RAW and report[0]["fallback"]


def test_preamble_stripped_and_text_used():
    llm = FakeLLM(lambda u: "Here is the refined transcript:\n\n" + RAW.replace("cube flow", "Kubeflow"))
    refined, report = refine_transcript(llm, RAW)
    assert "Kubeflow" in refined and not report[0]["fallback"]
    assert strip_preamble("Sure, here is the text:\nHello there") == "Hello there"


def test_chunking_covers_everything_and_terminates():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_sentences(text, 50)
    assert " ".join(chunks).split() == text.split()
    overlapped = chunk_sentences(text, 50, overlap_words=15)
    assert len(overlapped) > len(chunks) and overlapped[-1].endswith("199 is here.")


def test_long_run_on_sentence_is_cut():
    assert all(len(c.split()) <= 20 for c in chunk_sentences("word " * 95, 20))


def test_whisper_loop_collapsed():
    assert collapse_repetitions("Thank you. " * 6 + "Next topic.") == "Thank you. Thank you. Next topic."
