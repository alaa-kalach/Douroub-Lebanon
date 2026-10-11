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
    errors = results[~results.correct]
    print(f"\nLLM router accuracy: {results.correct.mean():.1%}")
    print(errors[["case", "message", "expected", "predicted"]].to_string())
    assert results.correct.mean() >= 0.85
