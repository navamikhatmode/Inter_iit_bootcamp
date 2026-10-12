import json

from meeting_ai.document import generate_record

TRANSCRIPT = ("Tom, can you write the migration script? Sure, I'll have it done by Friday. "
              "We should maybe switch the cache to Redis. Let's talk about that next week. "
              "After the review the team agreed to go with Vendor B.")


class FakeLLM:
    """Scripted stand-in for the documentation model. Routes on the 'TASK:' header of each prompt."""

    def __init__(self):
        self.calls = {}

    def generate(self, system, user, max_new_tokens=1024):
        task = user.split("\n", 1)[0].replace("TASK:", "").strip()
        self.calls[task] = self.calls.get(task, 0) + 1
        if task == "MINUTES":
            if self.calls[task] == 1:
                return "Sorry, here are the minutes (not json)"            # forces a retry
            return json.dumps({"minutes": ["Migration: script assigned to Tom"]})   # plain string -> repaired
        if task == "DECISIONS":
            return json.dumps({"decisions": [
                {"decision": "The team agreed to go with Vendor B.", "evidence": "the team agreed to go with Vendor B."},
                {"decision": "Switch the cache to Redis.", "evidence": "We should maybe switch the cache to Redis."},
                {"decision": "Adopt GraphQL.", "evidence": "Everyone agreed to adopt GraphQL next quarter"}]})
        if task == "VERIFY_DECISION":
            return json.dumps({"verdict": "proposed" if "Redis" in user.split("Candidate decision:")[1] else "agreed"})
        if task == "ACTIONS":
            return json.dumps({"action_items": [
                {"action": "Write the migration script", "owner": "Tom", "deadline": "Friday",
                 "evidence": "Tom, can you write the migration script? Sure, I'll have it done by Friday."},
                {"action": "Review dashboards", "owner": "Zed", "deadline": "Monday",
                 "evidence": "Tom, can you write the migration script?"}]})
        if task == "VERIFY_ACTION":
            return json.dumps({"verdict": "committed"})
        if task == "SUMMARY":
            return json.dumps({"summary": "The team assigned a migration script and chose Vendor B."})
        raise AssertionError(task)


def test_full_documentation_stage():
    llm = FakeLLM()
    record, audit = generate_record(llm, TRANSCRIPT, verify=True)

    # JSON retry worked (first reply was not JSON)
    assert llm.calls["MINUTES"] == 2
    # proposal and fabricated quote are NOT decisions; the real one is
    assert [d["decision"] for d in record["decisions"]] == ["The team agreed to go with Vendor B."]
    reasons = {r["text"]: r["reason"] for r in audit["rejected_candidates"]}
    assert "proposed" in reasons["Switch the cache to Redis."]
    assert "not found" in reasons["Adopt GraphQL."]
    # "action" key mapped to task; real owner/deadline kept
    first = record["action_items"][0]
    assert (first["task"], first["owner"], first["deadline"]) == ("Write the migration script", "Tom", "Friday")
    # owner and deadline that never occur in the transcript are NOT invented
    second = record["action_items"][1]
    assert second["owner"] == "Unspecified" and second["deadline"] == "Unspecified"
    # string minutes were repaired, with a topic
    assert record["minutes"][0]["topic"] == "Migration"
    assert record["summary"]


def test_without_verifier_proposal_still_needs_real_evidence():
    record, audit = generate_record(FakeLLM(), TRANSCRIPT, verify=False)
    assert "Adopt GraphQL." not in [d["decision"] for d in record["decisions"]]
