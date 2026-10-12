from meeting_ai.schema import UNSPEC, extract_json, validate_meeting_record


def test_action_key_aliases_are_mapped_not_wiped():
    """Regression: the old prompt said "action" but the validator read "task" -> every task became 'Unspecified'."""
    rec = validate_meeting_record({"action_items": [{"action": "Send the report", "owner": "Priya", "deadline": "Friday"}]})
    a = rec["action_items"][0]
    assert a["task"] == "Send the report" and a["owner"] == "Priya" and a["deadline"] == "Friday"


def test_missing_owner_and_deadline_become_unspecified():
    rec = validate_meeting_record({"action_items": [{"task": "Do X", "owner": "None", "deadline": ""}]})
    assert rec["action_items"][0]["owner"] == UNSPEC and rec["action_items"][0]["deadline"] == UNSPEC


def test_action_without_task_is_dropped_not_invented():
    repairs = []
    rec = validate_meeting_record({"action_items": [{"owner": "Tom"}]}, repairs)
    assert rec["action_items"] == [] and repairs


def test_string_minutes_get_a_topic():
    rec = validate_meeting_record({"minutes": ["Budget: costs were reviewed", "We talked about hiring plans"]})
    assert rec["minutes"][0] == {"topic": "Budget", "discussion": "costs were reviewed"}
    assert rec["minutes"][1]["topic"].startswith("Discussion point")


def test_none_strings_become_empty_lists():
    rec = validate_meeting_record({"decisions": "None", "action_items": "N/A", "minutes": None, "summary": None})
    assert rec == {"summary": "", "minutes": [], "decisions": [], "action_items": []}


def test_idempotent():
    once = validate_meeting_record({"decisions": ["We agreed on X"], "action_items": [{"action": "A", "owner": "Bo"}]})
    assert validate_meeting_record(once) == once


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    assert extract_json("no json here") is None
