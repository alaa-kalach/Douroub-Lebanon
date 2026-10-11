"""Douroub Lebanon test suite. Tests app.py, which is exported from business_agent.ipynb.

    python -m pytest tests.py -v                         # offline tests, no API calls
    $env:RUN_LIVE=1; python -m pytest tests.py -v        # also run live tests (PowerShell)

Live tests call the real Gemini API; they run only when RUN_LIVE=1 and GEMINI_API_KEY is set.
"""
import os
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
ME = ROOT / "me"


# ---------------------------------------------------------------- Phase 1: documents
def _pdf_text():
    return " ".join(p.extract_text() for p in PdfReader(ME / "about_business.pdf").pages)


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("–", "-").replace("—", "-"))


def test_required_files_exist():
    for f in ["about_business.pdf", "business_summary.txt", "experiences.csv"]:
        assert (ME / f).exists(), f


def test_catalog_schema_and_values():
    df = pd.read_csv(ME / "experiences.csv")
    assert list(df.id) == [f"D{i:02d}" for i in range(1, 21)]
    assert df.id.is_unique and df.name.is_unique
    assert set(df.day_slots) <= {1, 2}
    assert df.notna().all().all(), "catalog has empty cells"
    assert (df.price_usd > 0).all() and set(df.price_basis) == {"per_person"}
    assert (df.min_group <= df.max_group).all()
    assert set(df.availability_status) == {"inquiry_only"}
    assert set(df.transport_included) == {"unknown"}


def test_summary_and_pdf_agree_with_catalog():
    txt = _norm((ME / "business_summary.txt").read_text(encoding="utf-8"))
    pdf = _norm(_pdf_text())
    for r in pd.read_csv(ME / "experiences.csv").itertuples():
        line = f"{r.id} {r.name} ({r.region}, {r.category}) - {r.duration_label.lower()} - ${r.price_usd} per person"
        assert line in txt, f"summary mismatch for {r.id}: expected '{line}'"
        assert r.name in pdf and f"${r.price_usd} per person" in pdf, f"PDF mismatch for {r.id}"


def test_no_course_or_assignment_references():
    pattern = re.compile(r"(?i)assignment|course|EECE|academic|fictional|illustrative")
    for f in ["business_summary.txt", "experiences.csv"]:
        assert not pattern.search((ME / f).read_text(encoding="utf-8")), f
    assert not pattern.search(_pdf_text())


def test_identity_elements_present_in_both_docs():
    txt, pdf = (ME / "business_summary.txt").read_text(encoding="utf-8"), _pdf_text()
    for needle in ["Douroub Lebanon", "Find Your Way to the Authentic Lebanon", "Sara",
                   "Ahmad", "Khaled", "Alaa", "Arabic, English, and French"]:
        assert needle in txt and needle in _norm(pdf), needle
    for section in ["MISSION", "SERVICES", "TEAM", "WHAT MAKES US DIFFERENT", "HOW BOOKING WORKS",
                    "DETAILS CONFIRMED BY OUR TEAM ON REQUEST"]:
        assert section in txt
    for section in ["Mission", "Services", "Team", "What makes us different", "How booking works",
                    "Details confirmed by our team on request"]:
        assert section in pdf


# ---------------------------------------------------------------- app.py is the notebook's code
import app  # noqa: E402

LIVE = pytest.mark.skipif(not (os.getenv("RUN_LIVE") == "1" and app.has_api_key()),
                          reason="live test: set RUN_LIVE=1 and GEMINI_API_KEY")


def test_app_py_matches_notebook():
    """app.py must be an exact export of the notebook's '# EXPORT' cells (re-run the Export cell if this fails)."""
    nb = nbformat.read(ROOT / "business_agent.ipynb", as_version=4)
    blocks = [c.source.split("\n", 1)[1].strip() for c in nb.cells
              if c.cell_type == "code" and c.source.startswith("# EXPORT")]
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    for block in blocks:
        assert block in app_src, f"app.py is out of date; re-run the Export cell. Missing:\n{block[:200]}"
    assert len(blocks) >= 8


# ---------------------------------------------------------------- Phase 2: persona, instructions, router
def test_persona_and_instructions_are_separate_and_labeled():
    assert app.PERSONA.startswith("You are Alaa") and "Douroub Lebanon" in app.PERSONA
    assert app.INSTRUCTIONS.startswith("1. ") and app.PERSONA not in app.INSTRUCTIONS
    for rule in ["Never invent", "record_customer_interest", "record_feedback", "Never present an inquiry",
                 "confirmed reservation", "Privacy", "Security"]:
        assert rule in app.INSTRUCTIONS, rule
    assert set(app.ROUTE_GUIDANCE) == set(app.INTENTS)


def test_instructions_cover_locals_and_typo_ids():
    assert "preferred date(s)" in app.INSTRUCTIONS and "never assume a person is" in app.INSTRUCTIONS
    assert "travel dates" not in app.INSTRUCTIONS.lower()
    assert "preferred date or dates" in app.ROUTE_GUIDANCE["LEAD_COLLECTION"]
    assert "ask which experience they meant instead of logging it" in app.INSTRUCTIONS
    assert "not \"travel dates\"" in app.REFLECTION_PROMPT


def test_system_prompt_contains_all_parts():
    sp = app.build_system_prompt(app.Route("LEAD_COLLECTION", ["FEEDBACK_UNKNOWN"]), today=date(2026, 10, 11))
    order = [sp.index(h) for h in ["# PERSONA", "# INSTRUCTIONS", "# CONTEXT", "# ROUTING", "# BUSINESS KNOWLEDGE"]]
    assert order == sorted(order)
    assert "Sunday, 11 October 2026" in sp
    assert "LEAD_COLLECTION" in sp and "FEEDBACK_UNKNOWN" in sp
    assert "HOW BOOKING WORKS" in sp                         # summary text loaded
    assert "Details confirmed by our team on request" in sp  # PDF text loaded
    assert "# ROUTING" not in app.build_system_prompt()      # no route -> no routing block


def test_router_eval_set_is_valid():
    cases = app.ROUTER_EVAL_CASES
    assert len(cases) >= 30
    assert {c[3] for c in cases} == set(app.INTENTS)
    assert {c[4] for c in cases} <= set(app.INTENTS) | {""}
    categories = {c[0] for c in cases}
    for cat in ["general", "lead", "unknown", "feedback", "mixed", "ambiguous", "adversarial", "multilingual"]:
        assert cat in categories, cat


def test_rules_router_accuracy():
    results = app.evaluate_router("rules")
    assert results.correct.mean() >= 0.85, results[~results.correct][["message", "expected", "predicted"]]


@pytest.mark.parametrize("msg,last,intent,secondary", [
    ("my email is a.b@example.com", None, "LEAD_COLLECTION", []),
    ("Is pickup from the airport included?", None, "FEEDBACK_UNKNOWN", []),
    ("Karim", "Could you share your name and email?", "LEAD_COLLECTION", []),
    ("Karim", None, "GENERAL_QUESTION", []),
    ("Book D03 for 2, is it vegan?", None, "LEAD_COLLECTION", ["FEEDBACK_UNKNOWN"]),
    ("What's in D09?", None, "GENERAL_QUESTION", []),
])
def test_rules_router_key_cases(msg, last, intent, secondary):
    r = app.route_with_rules(msg, last)
    assert (r.intent, r.secondary, r.method) == (intent, secondary, "rules")


def test_parse_router_json_accepts_fenced_and_rejects_invalid():
    raw = ('```json\n{"intent": "lead_collection", "secondary": ["FEEDBACK_UNKNOWN", "LEAD_COLLECTION", "X"], '
           '"reason": "wants to book"}\n```')
    assert app._parse_router_json(raw) == ("LEAD_COLLECTION", ["FEEDBACK_UNKNOWN"], "wants to book")
    for bad in ["", "LEAD", '{"intent": "BOOKING"}']:
        with pytest.raises(ValueError):
            app._parse_router_json(bad)


class FakeClient:
    """Stands in for the OpenAI client: returns a fixed reply, or raises."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_llm_router_uses_reply_and_context():
    fake = FakeClient('{"intent": "LEAD_COLLECTION", "secondary": [], "reason": "gives name"}')
    r = app.route("Lina", history=[{"role": "assistant", "content": "What\'s your name?"}], client=fake)
    assert (r.intent, r.method) == ("LEAD_COLLECTION", "llm")
    sent = fake.calls[0]["messages"][1]["content"]
    assert "What\'s your name?" in sent and "<visitor_message>\nLina\n</visitor_message>" in sent
    assert fake.calls[0]["temperature"] == 0


@pytest.mark.parametrize("fake", [FakeClient(error=TimeoutError("api down")),
                                  FakeClient('{"intent": "BOOK_NOW"}'), FakeClient("sorry, I can\'t")])
def test_router_falls_back_to_rules(fake):
    r = app.route("Is transport included?", client=fake)
    assert r.intent == "FEEDBACK_UNKNOWN" and r.method == "rules-fallback" and "LLM router failed" in r.reason


@LIVE
def test_live_llm_router_accuracy():
    results = app.evaluate_router("llm", delay=0.5)
    answered = results[results.predicted != "ERROR"]
    print(f"\nLLM router accuracy: {results.correct.mean():.1%} overall, "
          f"{answered.correct.mean():.1%} on the {len(answered)} answered messages")
    print(results[~results.correct][["case", "message", "expected", "predicted", "reason"]].to_string())
    if len(answered) < 0.75 * len(results):
        pytest.skip(f"only {len(answered)}/{len(results)} router calls completed (API quota)")
    assert answered.correct.mean() >= 0.85


# ---------------------------------------------------------------- Phase 3: tools
import itertools  # noqa: E402
import json  # noqa: E402

CATALOG = pd.read_csv(ME / "experiences.csv").set_index("id")


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Every test writes logs to a temporary folder and starts with an empty de-duplication cache."""
    monkeypatch.setattr(app, "LOG_DIR", tmp_path / "logs")
    app._RECENT.clear()
    app.MODEL_COOLDOWN.clear()  # no model starts a test "resting" after an earlier test's rate limit
    return tmp_path / "logs"


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# --- record_customer_interest
def test_lead_saved_with_id_timestamp_and_push(isolated_logs, capsys):
    r = app.record_customer_interest("Lina@Example.com ", " Lina  Haddad", "D03 for 2 people on Sunday")
    assert r["status"] == "success" and re.fullmatch(r"INQ-\d{8}-[0-9A-F]{6}", r["inquiry_id"])
    assert "not a confirmed reservation" in r["next_step"]
    saved = _lines(isolated_logs / "inquiries.jsonl")
    assert saved == [{"inquiry_id": r["inquiry_id"], "timestamp": r["saved_at"], "name": "Lina Haddad",
                      "email": "lina@example.com", "message": "D03 for 2 people on Sunday", "status": "new"}]
    out = capsys.readouterr().out
    assert "[PUSH] New inquiry" in out and "l***@example.com" in out and "lina@example.com" not in out


@pytest.mark.parametrize("email,name,message,bad_field", [
    ("lina@", "Lina", "D03 for two people", "email"),
    ("lina.example.com", "Lina", "D03 for two people", "email"),
    ("lina@example", "Lina", "D03 for two people", "email"),
    ("", "Lina", "D03 for two people", "email"),
    ("lina@example.com", "unknown", "D03 for two people", "name"),
    ("lina@example.com", "", "D03 for two people", "name"),
    ("lina@example.com", "L", "D03 for two people", "name"),
    ("lina@example.com", "Lina", "hi", "message"),
    ("lina@example.com", "Lina", "", "message"),
])
def test_lead_validation_rejects_and_saves_nothing(isolated_logs, email, name, message, bad_field):
    r = app.record_customer_interest(email, name, message)
    assert r["status"] == "error" and r["error"] == "validation_failed" and bad_field in r["fields"]
    assert "NOT saved" in r["message"] and "inquiry_id" not in r
    assert not (isolated_logs / "inquiries.jsonl").exists()


def test_duplicate_lead_is_saved_once(isolated_logs):
    a = app.record_customer_interest("lina@example.com", "Lina", "D03 for two people")
    b = app.record_customer_interest("LINA@example.com", "Lina", "D03 for two  people")
    assert a["inquiry_id"] == b["inquiry_id"] and b["duplicate"] is True
    assert len(_lines(isolated_logs / "inquiries.jsonl")) == 1


def test_unique_ids_for_different_leads():
    ids = {app.record_customer_interest("a@example.com", "Ana", f"Interested in D0{i} for two")["inquiry_id"] for i in range(1, 8)}
    assert len(ids) == 7


def test_storage_failure_is_reported_not_hidden(monkeypatch):
    def broken(*_):
        raise PermissionError("disk is read-only")
    monkeypatch.setattr(app, "_append_jsonl", broken)
    for r in [app.record_customer_interest("lina@example.com", "Lina", "D03 for two people"),
              app.record_feedback("Is pickup included?")]:
        assert r["status"] == "error" and r["error"] == "storage_failed" and "Do not tell" in r["message"]


# --- record_feedback
def test_feedback_saved_with_kind(isolated_logs, capsys):
    r = app.record_feedback("Is the Qadisha hike wheelchair accessible?")
    c = app.record_feedback("The guide was late.", kind="complaint")
    assert r["status"] == c["status"] == "success" and re.fullmatch(r"FB-\d{8}-[0-9A-F]{6}", r["feedback_id"])
    saved = _lines(isolated_logs / "feedback.jsonl")
    assert [(x["kind"], x["question"]) for x in saved] == [
        ("unanswered_question", "Is the Qadisha hike wheelchair accessible?"), ("complaint", "The guide was late.")]
    assert "[PUSH] New complaint" in capsys.readouterr().out


@pytest.mark.parametrize("question,kind", [("", "unanswered_question"), ("hi", "unanswered_question"),
                                           ("Is pickup included?", "rant")])
def test_feedback_validation(isolated_logs, question, kind):
    r = app.record_feedback(question, kind)
    assert r["status"] == "error" and not (isolated_logs / "feedback.jsonl").exists()


# --- search_experiences
def test_search_results_are_grounded_in_the_catalog():
    r = app.search_experiences(limit=20)
    assert r["total_matches"] == 20
    for card in r["results"]:
        row = CATALOG.loc[card["id"]]
        assert (card["name"], card["region"], card["price_usd_per_person"], card["season"]) == \
               (row["name"], row["region"], row["price_usd"], row["season"])
        assert card["transport"].startswith("Not documented") and card["availability"].startswith("Inquiry only")


@pytest.mark.parametrize("filters", [
    {"max_price_per_person": 30}, {"region": "Chouf"}, {"category": "nature"}, {"group_size": 14},
    {"month": "July"}, {"month": 1}, {"max_duration_hours": 4}, {"region": "south"},
    {"interests": ["food"], "max_price_per_person": 35, "month": "August", "group_size": 2},
])
def test_search_respects_hard_filters(filters):
    r = app.search_experiences(limit=20, **filters)
    assert r["status"] == "success" and r["total_matches"] >= 1
    month = app._parse_month(filters.get("month"))
    for card in r["results"]:
        row = CATALOG.loc[card["id"]]
        if "max_price_per_person" in filters:
            assert row.price_usd <= filters["max_price_per_person"]
        if filters.get("region") == "south":
            assert row.region in {"Sidon", "Tyre", "Jezzine"}
        elif "region" in filters:
            assert row.region == filters["region"]
        if "category" in filters:
            assert row.category.lower() == filters["category"]
        if "group_size" in filters:
            assert row.min_group <= filters["group_size"] <= row.max_group
        if month:
            assert month in app.season_months(row.season)
        if "max_duration_hours" in filters:
            assert row.duration_hours <= filters["max_duration_hours"]


def test_search_ranks_by_keywords_and_reports_no_match():
    r = app.search_experiences(query="kayak sunset")
    assert r["results"][0]["id"] == "D20"
    none = app.search_experiences(query="scuba diving")
    assert none["total_matches"] == 0 and "record_feedback" in none["note"] and none["excluded_by_filter"]["keywords"] == 20
    assert app.search_experiences(month="Smarch")["error"] == "invalid_month"


def test_season_parsing():
    assert app.season_months("Winter only (December-March, snow-dependent)") == {12, 1, 2, 3}
    assert app.season_months("Spring to autumn (April-November)") == set(range(4, 12))
    assert app.season_months("Year-round; harvest season is September-October") == set(range(1, 13))


# --- compare_experiences
def test_compare_normalizes_ids_and_summarizes():
    r = app.compare_experiences(["d4", "D08", "D08", "D99"], group_size=4)
    assert [c["id"] for c in r["compared"]] == ["D04", "D08"] and r["not_found"] == ["D99"]
    assert r["summary"]["cheapest"] == ["D04"] and r["summary"]["price_difference_per_person_usd"] == 5
    assert [c["group_total_usd"] for c in r["compared"]] == [220, 240]
    assert "transport" in r["unknown_for_all_options"]


def test_compare_flags_group_size_and_needs_two_ids():
    r = app.compare_experiences(["D03", "D04"], group_size=12)
    assert r["summary"]["not_suitable_for_group_size"] == ["D03"]
    assert app.compare_experiences(["D01"])["error"] == "need_two_valid_ids"
    assert app.compare_experiences(["D01", "X1"])["error"] == "need_two_valid_ids"


# --- create_itinerary
def _placed(plan):
    return [a for d in plan["days"] for a in d["activities"] if not a.get("continued")]


def test_itinerary_respects_budget_and_group_total():
    r = app.create_itinerary(days=2, budget_per_person=150, interests=["nature", "food"], month="July", group_size=2)
    assert r["feasible"] and r["estimated_total_per_person_usd"] <= 150
    assert r["estimated_total_per_person_usd"] == sum(a["price_usd_per_person"] for a in _placed(r))
    assert r["estimated_group_total_usd"] == 2 * r["estimated_total_per_person_usd"]


def test_itinerary_revises_requested_plan_over_budget():
    r = app.create_itinerary(days=2, budget_per_person=100, experience_ids=["D08", "D06", "D03"])
    assert r["feasible"] and r["estimated_total_per_person_usd"] <= 100
    assert any("Removed D06" in x and "budget" in x for x in r["revisions"])
    assert {"D08", "D03"} <= {a["id"] for a in _placed(r)}


def test_itinerary_removes_out_of_season_and_impossible_overnight():
    r = app.create_itinerary(days=1, experience_ids=["D19", "D06", "D05"], month="July")
    ids = {a["id"] for a in _placed(r)}
    assert "D19" not in ids and "D06" not in ids and "D05" in ids
    assert any("D19" in x and "July" in x for x in r["revisions"])
    assert any("D06" in x and "at least 2 days" in x for x in r["revisions"])


def test_itinerary_overnight_spans_two_days():
    r = app.create_itinerary(days=2, experience_ids=["D06"])
    slots = [(d["day"], a["slot"]) for d in r["days"] for a in d["activities"] if a["id"] == "D06"]
    assert slots == [(1, "Afternoon, overnight stay"), (2, "Morning, end of overnight stay")]
    assert r["estimated_total_per_person_usd"] == 85


def test_itinerary_infeasible_explains_why():
    r = app.create_itinerary(days=1, budget_per_person=20)
    assert r["status"] == "success" and r["feasible"] is False and "$25" in r["infeasible_reason"]
    nothing = app.create_itinerary(days=2, interests=["scuba"])
    assert nothing["feasible"] is False and "no catalog experience" in nothing["infeasible_reason"]


def test_itinerary_total_budget_needs_group_size():
    assert app.create_itinerary(days=2, total_budget=200)["error"] == "group_size_required"
    r = app.create_itinerary(days=2, total_budget=200, group_size=4)
    assert r["estimated_total_per_person_usd"] <= 50 and r["estimated_group_total_usd"] <= 200


def test_itinerary_invariants_over_many_constraint_combinations():
    """Property check over 180 combinations: totals add up, budgets hold, slots are never overbooked,
    nothing repeats, and every scheduled experience is in season and accepts the group."""
    combos = itertools.product([1, 2, 3], [None, 40, 90, 200], [None, ["nature"], ["food", "history"]],
                               [None, "January", "July"], [None, 2, 9], [1, 2])
    count = 0
    for days, budget, interests, month, group, per_day in itertools.islice(combos, 0, None, 2):
        r = app.create_itinerary(days=days, budget_per_person=budget, interests=interests, month=month,
                                 group_size=group, max_activities_per_day=per_day)
        placed = _placed(r)
        ids = [a["id"] for a in placed]
        assert len(ids) == len(set(ids)), r
        assert r["estimated_total_per_person_usd"] == sum(a["price_usd_per_person"] for a in placed)
        if budget is not None:
            assert r["estimated_total_per_person_usd"] <= budget
        for day in r["days"]:
            assert len({a["id"] for a in day["activities"]}) <= per_day
        m = app._parse_month(month)
        for i in ids:
            row = CATALOG.loc[i]
            assert not m or m in app.season_months(row.season)
            assert not group or row.min_group <= group <= row.max_group
        assert r["feasible"] == bool(placed)
        count += 1
    assert count >= 100


# --- schemas and dispatcher
def test_tool_specs_are_complete_mcp_style():
    assert [s["name"] for s in app.TOOL_SPECS] == ["record_customer_interest", "record_feedback",
                                                   "search_experiences", "compare_experiences", "create_itinerary"]
    for s in app.TOOL_SPECS:
        assert {"name", "description", "inputSchema", "outputSchema"} <= set(s)
        assert set(s["inputSchema"].get("required", [])) <= set(s["inputSchema"]["properties"])
        assert "status" in s["outputSchema"]["properties"]
    assert app.SPEC_BY_NAME["record_customer_interest"]["inputSchema"]["required"] == ["email", "name", "message"]
    assert app.SPEC_BY_NAME["record_feedback"]["inputSchema"]["required"] == ["question"]


def test_api_tool_schemas_are_gemini_safe():
    text = json.dumps(app.OPENAI_TOOLS)
    assert len(app.OPENAI_TOOLS) == 5 and "additionalProperties" not in text and '"format"' not in text


@pytest.mark.parametrize("name,args,error", [
    ("book_now", {}, "unknown_tool"),
    ("record_feedback", "{not json", "invalid_json"),
    ("record_feedback", "[1, 2]", "invalid_json"),
    ("record_feedback", {}, "invalid_arguments"),
    ("record_feedback", {"question": "Is pickup included?", "priority": "high"}, "invalid_arguments"),
    ("create_itinerary", {"days": "two"}, "invalid_arguments"),
    ("create_itinerary", {"days": 9}, "invalid_arguments"),
    ("compare_experiences", {"experience_ids": ["D01"]}, "invalid_arguments"),
    ("search_experiences", {"category": "Shopping"}, "invalid_arguments"),
])
def test_dispatcher_rejects_bad_calls(name, args, error):
    r = app.execute_tool(name, args)
    assert r["status"] == "error" and r["error"] == error
    assert app.TOOL_CALL_LOG[-1]["tool"] == name and app.TOOL_CALL_LOG[-1]["status"] == "error"


def test_dispatcher_coerces_light_type_mistakes():
    r = app.execute_tool("create_itinerary", '{"days": "2", "group_size": "4", "budget_per_person": "120"}')
    assert r["status"] == "success" and r["estimated_group_total_usd"] == 4 * r["estimated_total_per_person_usd"]
    assert app.execute_tool("search_experiences", {"category": "nature"})["status"] == "success"


def test_dispatcher_catches_tool_crashes_and_masks_emails(monkeypatch):
    monkeypatch.setitem(app.TOOL_FUNCTIONS, "search_experiences", lambda **_: 1 / 0)
    assert app.execute_tool("search_experiences", {})["error"] == "tool_exception"
    app.execute_tool("record_customer_interest", {"email": "lina@example.com", "name": "Lina", "message": "D03 for two"})
    assert app.TOOL_CALL_LOG[-1]["arguments"]["email"] == "l***@example.com"


# ---------------------------------------------------------------- Phase 4: agent loop and memory
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall  # noqa: E402
from openai.types.chat.chat_completion_message_function_tool_call import Function  # noqa: E402


def tool_msg(*calls, content=None):
    """A model message that asks for one or more tools: calls are (name, arguments_dict) pairs."""
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=[
        ChatCompletionMessageFunctionToolCall(id=f"call_{i}", type="function",
                                              function=Function(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)])


def text_msg(text):
    return ChatCompletionMessage(role="assistant", content=text)


class ScriptedClient:
    """Plays back a fixed sequence of model messages and records every request it receives."""

    def __init__(self, *messages, error=None):
        self.queue, self.error, self.requests = list(messages), error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=self.queue.pop(0))])


def test_agent_plain_answer_sends_prompt_history_and_tools():
    client = ScriptedClient(text_msg("Ahla w sahla! We have 20 experiences."))
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"},
               {"role": "tool", "content": "ignored"}, {"role": "user", "content": {"file": "x.png"}}]
    turn = app.run_agent("What do you offer?", history, client=client, use_llm_router=False)
    assert turn.reply == "Ahla w sahla! We have 20 experiences." and turn.tool_events == [] and turn.rounds == 1
    req = client.requests[0]
    assert req["messages"][0]["role"] == "system" and "# PERSONA" in req["messages"][0]["content"]
    assert [m["role"] for m in req["messages"][1:]] == ["user", "assistant", "user"]
    assert req["messages"][-1]["content"] == "What do you offer?"
    assert [t["function"]["name"] for t in req["tools"]] == [s["name"] for s in app.TOOL_SPECS]


def test_agent_executes_tool_and_returns_real_result_to_model(isolated_logs):
    client = ScriptedClient(
        tool_msg(("record_customer_interest", {"email": "rami@example.com", "name": "Rami", "message": "D12 for 3 on 25 Oct"})),
        text_msg("Done! Your reference is {id}."))
    session = app.Session()
    turn = app.run_agent("Book D12 for 3, Rami, rami@example.com", session=session, client=client, use_llm_router=False)
    saved = _lines(isolated_logs / "inquiries.jsonl")
    assert len(saved) == 1 and turn.tool_events[0]["status"] == "success" and turn.rounds == 2
    second = client.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_0"
    tool_reply = second[-1]
    assert tool_reply["role"] == "tool" and tool_reply["tool_call_id"] == "call_0"
    assert json.loads(tool_reply["content"])["inquiry_id"] == saved[0]["inquiry_id"]
    assert session.saved_ids() == {saved[0]["inquiry_id"]} and session.visitor_name == "Rami"
    assert turn.tool_events[0]["arguments"]["email"] == "r***@example.com"


def test_agent_chains_tools_across_rounds():
    client = ScriptedClient(
        tool_msg(("search_experiences", {"interests": ["history"], "limit": 3})),
        tool_msg(("compare_experiences", {"experience_ids": ["D04", "D08"]}),
                 ("create_itinerary", {"days": 2, "interests": ["history"], "budget_per_person": 120})),
        text_msg("Here is your plan."))
    session = app.Session()
    turn = app.run_agent("Plan 2 history days under $120", session=session, client=client, use_llm_router=False)
    assert [e["tool"] for e in turn.tool_events] == ["search_experiences", "compare_experiences", "create_itinerary"]
    assert [e["round"] for e in turn.tool_events] == [1, 2, 2] and turn.rounds == 3
    assert session.preferences == {"interests": ["history"], "days": 2, "budget_per_person": 120}
    assert "budget per person (USD): 120" in session.summary()


def test_agent_passes_tool_errors_back_without_recording_success(isolated_logs):
    client = ScriptedClient(tool_msg(("record_customer_interest", {"email": "nope", "name": "Rami", "message": "D12 for 3"}),
                                     ("teleport", {})),
                            text_msg("Could you share a valid email?"))
    session = app.Session()
    turn = app.run_agent("Book D12", session=session, client=client, use_llm_router=False)
    assert [e["status"] for e in turn.tool_events] == ["error", "error"]
    results = [json.loads(m["content"]) for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [r["error"] for r in results] == ["validation_failed", "unknown_tool"]
    assert session.saved_records == [] and not (isolated_logs / "inquiries.jsonl").exists()


def test_agent_caps_runaway_tool_loops():
    loops = [tool_msg(("search_experiences", {"query": "nature"})) for _ in range(app.MAX_TOOL_ROUNDS)]
    client = ScriptedClient(*loops, text_msg("Here are some nature options."))
    turn = app.run_agent("nature?", client=client, use_llm_router=False)
    assert turn.error == "max_tool_rounds" and len(turn.tool_events) == app.MAX_TOOL_ROUNDS
    assert turn.reply == "Here are some nature options." and "tools" not in client.requests[-1]


def test_agent_api_failure_returns_fallback():
    turn = app.run_agent("hello", client=ScriptedClient(error=ConnectionError("offline")))
    assert turn.reply == app.FALLBACK_REPLY and "ConnectionError" in turn.error
    assert turn.route.method == "rules-fallback"


class RateLimit(Exception):
    status_code = 429


def test_llm_call_waits_as_suggested_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(app.time, "sleep", waits.append)
    replies = [RateLimit("429 RESOURCE_EXHAUSTED ... 'retryDelay': '7s'"), RateLimit("Please retry in 3.5s."), "ok"]

    def create(**_):
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert app.llm_call(client, model="m", messages=[]) == "ok"
    assert waits == [8.0, 4.5]


def test_llm_call_fails_fast_on_daily_quota_and_non_rate_errors(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda s: pytest.fail("should not wait"))
    for error, expected in [(RateLimit("429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
                             app.DailyQuotaExceeded), (ValueError("bad request"), ValueError)]:
        client = ScriptedClient(error=error)
        with pytest.raises(expected):
            app.llm_call(client, model="m", messages=[])
    turn = app.run_agent("hi", client=ScriptedClient(error=RateLimit("429 PerDay quota")), use_llm_router=False)
    assert turn.reply == app.BUSY_REPLY


def test_agent_routing_guidance_and_memory_reach_the_prompt():
    session = app.Session(preferences={"group_size": 3, "interests": ["history"]})
    client = ScriptedClient(text_msg('{"intent": "LEAD_COLLECTION", "secondary": ["FEEDBACK_UNKNOWN"], "reason": "x"}'),
                            text_msg("Sure!"))
    turn = app.run_agent("Book D03, is it vegan?", session=session, client=client)
    assert turn.route.intent == "LEAD_COLLECTION" and turn.route.method == "llm"
    system = client.requests[1]["messages"][0]["content"]
    assert "classified this message as LEAD_COLLECTION" in system and "FEEDBACK_UNKNOWN" in system
    assert "# SESSION MEMORY" in system and "group size: 3" in system and "No inquiry or feedback has been saved" in system


def test_trim_history_keeps_last_text_messages():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
    trimmed = app.trim_history(history)
    assert len(trimmed) == app.HISTORY_LIMIT and trimmed[-1]["content"] == "m29"


def test_session_ignores_failed_and_duplicate_records():
    s = app.Session()
    s.update_from_tool("record_feedback", {"question": "q?"}, {"status": "error"})
    s.update_from_tool("record_feedback", {"question": "q?"}, {"status": "success", "feedback_id": "FB-1"})
    s.update_from_tool("record_feedback", {"question": "q?"}, {"status": "success", "feedback_id": "FB-1", "duplicate": True})
    s.update_from_tool("search_experiences", {"interests": ["Food"]}, {"status": "success"})
    s.update_from_tool("create_itinerary", {"interests": ["food", "nature"], "group_size": 2}, {"status": "success"})
    assert s.saved_ids() == {"FB-1"} and s.preferences == {"interests": ["food", "nature"], "group_size": 2}


def test_agent_eval_scenarios_cover_all_tools():
    expected = set().union(*(sc["expect"] for sc in app.AGENT_EVAL_SCENARIOS))
    assert expected == set(app.TOOL_FUNCTIONS)
    assert len(app.AGENT_EVAL_SCENARIOS) >= 10


@LIVE
def test_live_agent_end_to_end(tmp_path):
    """Set AGENT_SCENARIOS="mixed,memory" to run only scenarios whose names contain those words."""
    only = [x.strip() for x in os.getenv("AGENT_SCENARIOS", "").split(",") if x.strip()] or None
    results, _ = app.evaluate_agent(log_dir=tmp_path / "eval", only=only)
    print("\n" + results[["scenario", "outcome", "tools", "problems"]].to_string())
    completed = results[results.outcome != "rate_limited"]
    if len(completed) < len(results):
        print(f"{len(results) - len(completed)} scenario(s) stopped by the API quota; re-run them later with "
              f'AGENT_SCENARIOS="{",".join(results[results.outcome == "rate_limited"].scenario.str.split(":").str[0])}"')
    if len(completed) == 0:
        pytest.skip("every scenario was stopped by the API quota")
    assert completed.passed.mean() >= 0.8, completed[~completed.passed][["scenario", "problems"]].to_string()
    if only:
        return
    assert {"search_experiences", "compare_experiences", "create_itinerary", "record_customer_interest",
            "record_feedback"} <= {t.split(":")[0] for t in ", ".join(results.tools).split(", ") if t.endswith(":success")}


# ---------------------------------------------------------------- Phase 5: reflection (bonus)
def _turn(intent="GENERAL_QUESTION", draft="", events=None):
    return app._fake_turn(intent, draft, events)


@pytest.mark.parametrize("text", [
    "Your booking for D03 Taste of the Village is confirmed for Sunday!",
    "Your reservation is confirmed!",
    "Good news, your spot on D05 has been reserved.",
    "You're all booked for Saturday.",
    "I've booked it for you.",
])
def test_booking_claims_are_flagged(text):
    assert "booking_claim" in {f["category"] for f in app.deterministic_checks(text, _turn(), app.Session())}


@pytest.mark.parametrize("text", [
    "Your booking is not confirmed until Khaled emails you.",
    "Tours are confirmed by the team by email.",
    "I've noted that you're 3 people.",
    "D04 Heritage Through Time ($55 per person) and D08 Baalbek ($60 per person) are both full days.",
    "For 4 people, D04 is $55 per person, so $220 in total.",
    "Ahla w sahla! How can I help you plan your trip?",
])
def test_good_sentences_raise_no_flags(text):
    assert app.deterministic_checks(text, _turn(), app.Session(), "we are 4 people") == []


def test_saved_ids_and_claims_are_allowed_after_a_real_save():
    s = app.Session(saved_records=[{"type": "inquiry", "id": "INQ-20261011-ABC123"}])
    event = {"tool": "record_customer_interest", "status": "success", "arguments": {}, "result": {"status": "success"}}
    text = "I've recorded your inquiry; your reference is INQ-20261011-ABC123."
    assert app.deterministic_checks(text, _turn("LEAD_COLLECTION", events=[event]), s) == []


def test_rule_layer_alone_fixes_planted_problems(isolated_logs):
    results = app.evaluate_reflection(use_llm=False)
    by_case = results.set_index("case")
    assert by_case.loc["control: correct price", "passed"] and by_case.loc["control: honest unknown", "passed"]
    assert results.passed.sum() >= 7, results[~results.passed][["case", "detected", "final"]]
    assert "$55 per person" in by_case.loc["wrong price", "final"]
    assert "INQ-20261011-ABC123" not in by_case.loc["fake inquiry ID", "final"]
    assert "is confirmed" not in by_case.loc["invented booking confirmation", "final"]
    log = _lines(isolated_logs / "reflection_log.jsonl")
    assert len(log) == len(results)
    assert {"timestamp", "draft", "final", "changed", "failed_checks", "method"} <= set(log[0])


def _review(verdict, revised="", issues=()):
    return text_msg(json.dumps({"verdict": verdict, "issues": list(issues), "revised_reply": revised}))


def test_llm_revision_is_used_and_logged(isolated_logs, monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    turn = _turn("FEEDBACK_UNKNOWN", "Yes, transport from Beirut is included in D08.")
    fixed = "Transport for D08 isn't documented, so the team will confirm it."
    client = ScriptedClient(_review("revise", fixed, [{"category": "invented_information", "detail": "transport"}]))
    final, report = app.reflect_and_revise("Is transport included for Baalbek?", turn, app.Session(), client=client)
    assert final == fixed and report["method"] == "llm" and report["changed"] is True
    assert report["failed_checks"] == ["invented_information"]
    prompt = client.requests[0]["messages"][1]["content"]
    assert "DRAFT:\nYes, transport from Beirut is included in D08." in prompt and "D08 Baalbek: City of the Sun" in prompt
    assert client.requests[0]["model"] == app.FAST_MODEL
    entry = _lines(isolated_logs / "reflection_log.jsonl")[0]
    assert entry["draft"] == turn.draft and entry["final"] == fixed


def test_llm_ok_keeps_draft(monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    draft = "D05 Coastal Charms of Batroun is $28 per person."
    final, report = app.reflect_and_revise("price of D05?", _turn(draft=draft), app.Session(), client=ScriptedClient(_review("ok")))
    assert final == draft and report["changed"] is False and report["llm_verdict"] == "ok"


@pytest.mark.parametrize("client", [ScriptedClient(text_msg("I think it's fine")), ScriptedClient(error=TimeoutError())])
def test_reflection_failure_falls_back_to_safe_patches(client, monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    turn = _turn("LEAD_COLLECTION", "Done! Your reservation is confirmed, reference INQ-20261011-FAKE00.")
    final, report = app.reflect_and_revise("book D05", turn, app.Session(), client=client)
    assert report["method"] == "deterministic-fallback" and "llm_error" in report
    assert "is confirmed" not in final and "INQ-20261011-FAKE00" not in final
    assert "not a confirmed reservation" in final and "email" in final


def test_safety_net_patches_a_bad_llm_revision(monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    bad_revision = "Thanks! Your inquiry reference is INQ-20261011-ZZZ999. D04 is $45 per person."
    final, report = app.reflect_and_revise("D04?", _turn(draft="D04 is great."), app.Session(),
                                           client=ScriptedClient(_review("revise", bad_revision)))
    assert "INQ-20261011-ZZZ999" not in final and "$55 per person" in final and report["patched"]


def test_flags_mode_skips_the_llm_when_nothing_is_suspicious(monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "flags")
    client = ScriptedClient()
    final, report = app.reflect_and_revise("hi", _turn(draft="Ahla w sahla!"), app.Session(), client=client)
    assert client.requests == [] and report["method"] == "none" and final == "Ahla w sahla!"


def test_run_agent_with_reflection_returns_revised_reply_and_keeps_draft(monkeypatch):
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    client = ScriptedClient(text_msg("Lovely! D20 is $40 per person."),
                            _review("revise", "Lovely! D20 is $40 per person. Could you share your name and email?",
                                    [{"category": "forgot_contact_request", "detail": "no email asked"}]))
    turn = app.run_agent("Book D20 for 2 please", client=client, use_llm_router=False, reflect=True)
    assert turn.draft == "Lovely! D20 is $40 per person."
    assert turn.reply.endswith("name and email?") and turn.reflection["changed"]


@LIVE
def test_live_reflection_fixes_planted_problems():
    results = app.evaluate_reflection(use_llm=True)
    print("\n" + results[["case", "detected", "method", "changed", "fixed", "passed"]].to_string())
    assert results.passed.sum() >= 7


# ---------------------------------------------------------------- Phase 6: Gradio app
GRADIO_HISTORY = [  # the exact shape Gradio 6 passes to ChatInterface functions
    {"role": "user", "content": [{"type": "text", "text": "We are 3 people."}]},
    {"role": "assistant", "metadata": {"title": "🧭 Route: GENERAL_QUESTION"}, "content": [{"type": "text", "text": "trace"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Lovely, a group of 3!"}]},
    {"role": "user", "content": [{"type": "file", "file": {"path": "photo.png"}}]},
]


def test_trim_history_reads_gradio_format_and_skips_trace_panels():
    assert app.trim_history(GRADIO_HISTORY) == [{"role": "user", "content": "We are 3 people."},
                                                {"role": "assistant", "content": "Lovely, a group of 3!"}]
    assert app.message_text([{"type": "text", "text": "a"}, "b", {"type": "file"}]) == "a\nb"


def test_agent_sends_gradio_history_as_plain_text():
    client = ScriptedClient(text_msg("ok"))
    app.run_agent("Plan 2 days", GRADIO_HISTORY, client=client, use_llm_router=False)
    sent = client.requests[0]["messages"]
    assert sent[1] == {"role": "user", "content": "We are 3 people."}
    assert sent[2] == {"role": "assistant", "content": "Lovely, a group of 3!"} and len(sent) == 4


def test_sessions_are_per_visitor_and_bounded(monkeypatch):
    monkeypatch.setattr(app, "SESSIONS", app.OrderedDict())
    monkeypatch.setattr(app, "MAX_SESSIONS", 3)
    a1, a2 = app.get_session(SimpleNamespace(session_hash="a")), app.get_session(SimpleNamespace(session_hash="a"))
    b = app.get_session(SimpleNamespace(session_hash="b"))
    assert a1 is a2 and a1 is not b
    assert app.get_session(None) is not app.get_session(None)
    for key in "cde":
        app.get_session(SimpleNamespace(session_hash=key))
    assert list(app.SESSIONS) == ["c", "d", "e"]


def test_respond_guards(monkeypatch):
    assert "What would you like" in app.respond("   ", [])
    assert "shorten" in app.respond("x" * (app.MAX_MESSAGE_CHARS + 1), [])
    monkeypatch.setattr(app, "API_KEY", "")
    assert "not configured" in app.respond("hello", [])


def test_respond_returns_trace_panels_and_final_reply(monkeypatch, isolated_logs):
    monkeypatch.setattr(app, "API_KEY", "test")
    monkeypatch.setattr(app, "REFLECTION_MODE", "always")
    client = ScriptedClient(
        text_msg('{"intent": "LEAD_COLLECTION", "secondary": [], "reason": "booking"}'),
        tool_msg(("record_customer_interest", {"email": "lina@example.com", "name": "Lina", "message": "D05 for 3 people"})),
        text_msg("Saved! Reference {id}."), _review("ok"))
    monkeypatch.setattr(app, "get_client", lambda: client)
    out = app.respond("Book D05 for 3, Lina, lina@example.com", [], True, True, SimpleNamespace(session_hash="t1"))
    titles = [m.metadata["title"] for m in out[:-1]]
    assert titles[0] == "🧭 Understood as: booking request"
    assert titles[1] == "🛠️ Tool: record_customer_interest (success)"
    assert titles[2].startswith("🔍 Reflection")
    assert out[-1].content.startswith("Saved!") and not out[-1].metadata
    assert "l***@example.com" in out[1].content and "lina@example.com" not in out[1].content


def test_respond_without_trace_returns_plain_text_and_survives_crashes(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "test")
    monkeypatch.setattr(app, "get_client", lambda: ScriptedClient(text_msg('{"intent": "GENERAL_QUESTION"}'), text_msg("Hello!")))
    assert app.respond("hi", [], False, False) == "Hello!"
    monkeypatch.setattr(app, "run_agent", lambda *a, **k: 1 / 0)
    assert app.respond("hi", [], False, False) == app.FALLBACK_REPLY


def test_build_demo_constructs_the_branded_page():
    demo = app.build_demo()
    assert isinstance(demo, app.gr.Blocks) and isinstance(demo.chat, app.gr.ChatInterface)
    reflect_box, trace_box = demo.chat.additional_inputs
    assert reflect_box.value is True and trace_box.value is True  # settings rendered on top, both on
    assert "Douroub Lebanon" in app.SIDE_HTML and "دروب لبنان" in app.SIDE_HTML and 'class="db-map"' in app.SIDE_HTML
    assert app.LEBANON_MAP_SVG.count('class="db-place"') == 14 and "db-flag" in app.SIDE_HTML
    assert set(app.REGION_CHIPS) == set(app.load_catalog().region)  # one chip per catalog region
    assert "fill-opacity" in app.CSS and "data:image/svg+xml" in app.CSS  # faint cedar watermark
    assert "doesn't reserve a spot" in app.FOOTNOTE_HTML and "passport" in app.FOOTNOTE_HTML
    assert set(app.LAUNCH_OPTIONS) == {"theme", "css", "head", "footer_links"}
    assert len(app.EXAMPLES) >= 5 and app._in_notebook() is False


def test_duplicate_inquiry_still_counts_as_saved_in_a_new_session(isolated_logs):
    first, second = app.Session(), app.Session()
    args = {"email": "lina@example.com", "name": "Lina", "message": "D05 for 3 people"}
    for session in (first, second):  # the same request sent from two browser tabs
        result = app.execute_tool("record_customer_interest", args)
        session.update_from_tool("record_customer_interest", args, result)
    assert first.saved_ids() == second.saved_ids() and len(_lines(isolated_logs / "inquiries.jsonl")) == 1
    reply = f"Saved! Your reference is {next(iter(second.saved_ids()))}."
    turn = app._fake_turn("LEAD_COLLECTION", reply,
                          [{"tool": "record_customer_interest", "status": "success", "arguments": {}, "result": result}])
    assert app.deterministic_checks(reply, turn, second) == []



# ---------------------------------------------------------------- model fallback chains
class ModelClient:
    """Fake client whose behaviour depends on the model asked for: an exception, or a reply message."""

    def __init__(self, behaviour: dict, default=None):
        self.behaviour, self.default, self.models = behaviour, default, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.models.append(kwargs["model"])
        outcome = self.behaviour.get(kwargs["model"], self.default)
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=outcome)])


class Missing(Exception):
    status_code = 404


class BadRequest(Exception):
    status_code = 400


class ServerDown(Exception):
    status_code = 503


def test_chains_are_configured_and_start_with_the_preferred_model():
    assert len(app.AGENT_MODELS) >= 2 and len(app.FAST_MODELS) >= 2
    assert app.AGENT_MODELS[0] == app.AGENT_MODEL and app.FAST_MODELS[0] == app.FAST_MODEL


def test_fallback_switches_at_once_and_rests_the_exhausted_model(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda s: pytest.fail("should switch models, not wait"))
    client = ModelClient({"a": RateLimit("429 quotaId GenerateRequestsPerDayPerProjectPerModel"),
                          "b": text_msg("from b")})
    resp, used = app.llm_call_with_fallback(client, ["a", "b", "c"], messages=[])
    assert used == "b" and resp.choices[0].message.content == "from b" and client.models == ["a", "b"]
    assert app.MODEL_COOLDOWN["a"] - app.time.time() > 3000  # daily limit: rests about an hour
    app.llm_call_with_fallback(client, ["a", "b", "c"], messages=[])
    assert client.models[-1] == "b" and client.models.count("a") == 1  # 'a' is skipped while resting


def test_fallback_skips_unknown_models_and_server_errors():
    client = ModelClient({"typo-model": Missing("model not found"), "b": ServerDown("503"), "c": text_msg("ok")})
    assert app.llm_call_with_fallback(client, ["typo-model", "b", "c"], messages=[])[1] == "c"


def test_fallback_does_not_hide_bad_requests():
    client = ModelClient({"a": BadRequest("invalid argument")}, default=text_msg("never"))
    with pytest.raises(BadRequest):
        app.llm_call_with_fallback(client, ["a", "b"], messages=[])
    assert client.models == ["a"]


def test_when_every_model_is_busy_it_waits_instead_of_failing(monkeypatch):
    waits = []
    monkeypatch.setattr(app.time, "sleep", waits.append)
    client = ModelClient({"a": [RateLimit("429 retry in 4s"), RateLimit("429 retry in 4s"), text_msg("ok")],
                          "b": RateLimit("429 retry in 9s")})
    resp, used = app.llm_call_with_fallback(client, ["a", "b"], messages=[])
    assert used == "a" and waits == [5.0] and resp.choices[0].message.content == "ok"


def test_agent_falls_back_mid_conversation_and_reports_it(monkeypatch, isolated_logs):
    first, backup = app.AGENT_MODELS[0], app.AGENT_MODELS[1]
    client = ModelClient({first: RateLimit("429 PerDay quota"),
                          backup: [tool_msg(("search_experiences", {"region": "Batroun"})), text_msg("D05 and D20.")]})
    turn = app.run_agent("What is in Batroun?", client=client, use_llm_router=False)
    assert turn.reply == "D05 and D20." and turn.model == backup and not turn.error
    assert client.models == [first, backup, backup]  # the backup keeps the whole reply once it starts
    titles = [m.metadata["title"] for m in app.trace_messages(turn)]
    assert f"🔁 Answered by backup model {backup}" in titles


def test_quota_message_only_when_every_model_is_out(monkeypatch):
    client = ModelClient({}, default=RateLimit("429 PerDay quota"))
    turn = app.run_agent("hi", client=client, use_llm_router=False)
    assert turn.reply == app.BUSY_REPLY and set(client.models) == set(app.AGENT_MODELS)
