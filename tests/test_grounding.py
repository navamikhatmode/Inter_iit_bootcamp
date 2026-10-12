from meeting_ai.grounding import Indexed, find_quote, evidence_text, value_supported

T = "Tom, can you write the migration script? Sure, I'll have it done by Friday. We should maybe move the cache to Redis."


def test_exact_quote_found_and_returned_verbatim():
    idx = Indexed(T)
    score, s, e = find_quote("sure i'll have it done by friday", idx)
    assert score == 1.0 and evidence_text(idx, s, e) == "Sure, I'll have it done by Friday."


def test_slightly_wrong_quote_still_found():
    assert find_quote("Tom can you write the migration scripts", Indexed(T)) is not None


def test_fabricated_quote_rejected():
    assert find_quote("Everyone agreed to adopt GraphQL next quarter", Indexed(T)) is None


def test_too_short_quote_rejected():
    assert find_quote("Friday", Indexed(T)) is None


def test_owner_and_deadline_must_appear():
    idx = Indexed(T)
    assert value_supported("Tom", idx) and value_supported("Friday", idx)
    assert not value_supported("Priya", idx) and not value_supported("next Monday", idx)


def test_ordinals_and_possessives_normalised():
    idx = Indexed("Deadline is October 7th. Tom's team owns it.")
    assert value_supported("October 7", idx) and value_supported("Tom", idx)
