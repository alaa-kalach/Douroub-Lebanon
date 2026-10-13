---
title: Douroub Lebanon
emoji: 🌲
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: Alaa, the AI concierge of Douroub Lebanon
---

# Douroub Lebanon — Alaa, the AI concierge

**Douroub Lebanon** (دروب لبنان, "paths of Lebanon") is a small Lebanese experiences company: locally hosted
village walks, food and craft workshops, heritage days, nature hikes, coastal outings, and mountain stays across
13 regions, from Tripoli and the Qadisha Valley to Tyre and Jezzine. It serves residents of Lebanon as well as
visitors.

**Alaa** is its AI concierge. Alaa answers questions about the company and its 20 experiences, compares options,
plans multi-day itineraries within a budget, records booking inquiries for the team, and logs questions it
cannot answer — without inventing prices, availability, or confirmations.

> Live demo: _[Hugging Face Space link]_

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![Alaa in the chat interface](docs/screenshot.png) -->

---

## What Alaa can do

| Visitor asks… | What happens |
|---|---|
| "What can I do in the Chouf under $35?" | Searches the catalog with hard filters and explains why each option fits |
| "Compare D04 and D08 for 4 people" | Side-by-side comparison with group totals and what is still unknown |
| "Plan 2 days, nature and food, $120 each" | Builds a geographically sensible itinerary and revises it if it breaks the budget or season |
| "I'd like to book D12 for 3 on 25 October" | Collects name and email, records the inquiry, returns an `INQ-…` reference |
| "Is pickup from Beirut included?" | Says honestly that the team confirms this, logs it, returns an `FB-…` reference |
| "What's the weather like in late October?" | General seasonal background, clearly not a forecast, nothing logged |

Alaa replies in the visitor's language (Arabic, English, or French) and never presents an inquiry as a confirmed
booking: Khaled from the team confirms everything by email.

---

## Project structure

```
Douroub Lebanon/
├── me/
│   ├── about_business.pdf     business profile, policies, and catalog table (4 pages)
│   ├── business_summary.txt   the same knowledge as plain text
│   └── experiences.csv        the experience catalog: 20 experiences, the source of truth for catalog facts
├── business_agent.ipynb       main code: every component, explained and demonstrated, plus the evaluations
├── app.py                     the Gradio app, exported from the notebook (run this, or deploy it)
├── tests.py                   offline test suite (+ optional live tests)
├── requirements.txt
├── .env.example               copy to .env and add your key
└── logs/                      created at run time, git-ignored (inquiries, feedback, reflection log)
```

**The notebook is the source of truth.** Every code cell that starts with `# EXPORT` is copied, in order, into
`app.py` by the notebook's export cell. Edit the notebook, rerun the export cell, and `tests.py` checks that the two
stay in sync. `app.py` never reads the notebook at run time.

---

## Quick start

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

copy .env.example .env            # Windows (cp on macOS / Linux), then put your key in .env
python app.py                     # opens http://127.0.0.1:7860
```

To explore how it works, open `business_agent.ipynb` and run it top to bottom.

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `LLM_BASE_URL` | Gemini's OpenAI-compatible endpoint | any OpenAI-compatible endpoint works |
| `AGENT_MODELS` | `gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite` | fallback chain for the agent |
| `FAST_MODELS` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash` | fallback chain for the router and reflection |
| `REFLECTION_MODE` | `always` | `always`, `flags` (LLM review only when needed), or `off` |

The key lives only in `.env`, which is git-ignored and never committed.

---

## How it works

```
visitor message
      │
      ▼
 ┌─────────┐   GENERAL_QUESTION / LEAD_COLLECTION / FEEDBACK_UNKNOWN (+ secondary intent)
 │ Router  │── LLM classifier (JSON), with a rule-based fallback
 └────┬────┘
      ▼
 ┌──────────────────────────────┐
 │ System prompt                │  persona + instructions + business knowledge (summary + PDF)
 │                              │  + routing guidance + session memory + today's date
 └────┬─────────────────────────┘
      ▼
 ┌──────────────────────────────┐      ┌──────────────────────────────────────┐
 │ Agent loop (up to 6 rounds)  │◄────►│ Tools, validated and run in Python    │
 │ model decides which tools    │      │ search · compare · itinerary ·        │
 │ to call; real results return │      │ record_customer_interest · feedback   │
 └────┬─────────────────────────┘      └──────────────────────────────────────┘
      ▼
 ┌──────────────────────────────┐
 │ Reflection                   │  rule checks + LLM review → approve or revise;
 │                              │  draft and final are logged
 └────┬─────────────────────────┘
      ▼
   reply (+ optional trace panels in the UI)
```

### 1. Business knowledge
`business_summary.txt` and the text of `about_business.pdf` go into the system prompt. Experience details always
come from `experiences.csv` through the tools, so the model is never asked to remember catalog facts. Details the
company does not publish (transport, availability, accessibility, dietary suitability, safety, exact schedules,
payment, refunds) are listed explicitly as "confirmed by the team on request".

### 2. Persona and instructions
Kept separate and labelled in the prompt: `# PERSONA` (who Alaa is, its tone) and `# INSTRUCTIONS` (13 numbered
rules: sources of truth, never invent, prices and dates, leads, unknown questions, tool honesty, bookings,
privacy, scope, security, style, neutral wording for locals and visitors).

### 3. Router
Each message is classified into one of three intents, plus an optional secondary intent for mixed messages
("book D03 — is it vegan?" is a lead **and** an unanswered question). The intent selects extra guidance for the
agent. The LLM router runs on the fast model chain; if it fails, a rule-based router takes over.

### 4. Tools
All five tools are defined in MCP style (`name`, `description`, `inputSchema`, `outputSchema`), validated against
their schema before running, and always return a `status`. Every call is printed as a `[TOOL]` line; saves are
also printed as `[PUSH]` notifications and written to JSON Lines files in `logs/`.

| Tool | Required | Optional | Returns |
|---|---|---|---|
| `record_customer_interest` | `email`, `name`, `message` | — | `inquiry_id`, `saved_at`, `next_step` |
| `record_feedback` | `question` | `kind` (unanswered_question, complaint, correction, suggestion, praise, other) | `feedback_id`, `saved_at`, `next_step` |
| `search_experiences` | — | `query`, `interests`, `category`, `region`, `max_price_per_person`, `max_duration_hours`, `group_size`, `month`, `limit` | ranked `results`, `filters_applied`, what was excluded and why |
| `compare_experiences` | `experience_ids` (2–6) | `group_size` | side-by-side rows, group totals, cheapest / longest, what is unknown for all |
| `create_itinerary` | `days` | `budget_per_person`, `total_budget`, `group_size`, `interests`, `region`, `month`, `experience_ids`, `max_activities_per_day` | day-by-day plan, total per person, `feasible`, revisions and warnings |

`create_itinerary` keeps each day within one broad area, respects seasons and group sizes, and when requested
experiences do not fit (over budget, out of season, an overnight stay on a one-day trip) it removes or replaces
them and says why.

### 5. Agent loop and memory
The model receives the system prompt, the last 12 messages, and the tool schemas. Tool calls are executed in
Python and their real results are sent back, for up to 6 rounds. A per-conversation `Session` remembers the
visitor's name, preferences given to tools (group size, budget, month…), and the IDs actually saved — the only
reference numbers Alaa is allowed to quote.

### 6. Reflection
Every draft is checked before it is sent:

- **Rule checks (no LLM):** invented or unsaved reference numbers, claims of a confirmed booking, "saved" claims
  with nothing saved, unknown experience IDs, per-person prices that don't match the catalog, itinerary totals
  that don't match the tool, a booking request with no request for an email, team promises with nothing
  recorded, and wording that assumes the visitor is travelling.
- **LLM review:** the draft is compared with the catalog facts, the tool results, and the saved IDs; the reviewer
  approves it or returns a minimally corrected version.
- **Safety net:** critical problems are re-checked on the final text and patched deterministically, even if the
  LLM review failed or missed them.

Draft, flags, verdict, and final reply are written to `logs/reflection_log.jsonl`.

### 7. Interface
A Gradio `ChatInterface` inside a branded layout: a side panel with the company, a map of Lebanon with its
regions, and region shortcuts; the chat with example prompts; and two settings — reflection on/off and
"show how Alaa answered", which adds collapsible panels for the route, each tool call, the model used, and the
reflection verdict.

---

## Safety and honesty measures

| Risk | Measure |
|---|---|
| Invented prices, inclusions, availability | Catalog facts only via tools; reflection checks every price against the catalog |
| Fake confirmations | "Inquiry, not reservation" rule; booking-claim detection and patching |
| Invented reference numbers | Only IDs returned by a successful tool call may be quoted; anything else is removed |
| Invented contact details | **Python refuses** an inquiry whose email or name the visitor never typed, whatever the model sends |
| Duplicate records | The same inquiry or question sent twice while the app is running returns the existing ID |
| Prompt injection | Visitor messages are treated as data; rule 11 and an injection test scenario |
| Privacy | Only name, email, and request are collected; emails are masked in all printed logs; `logs/` is git-ignored |
| API limits | Model fallback chains; a model that hits its daily quota rests for an hour; short limits are waited out |

---

## Evaluation

All evaluations are in the notebook and can be rerun. The live results below come from a full run of
`business_agent.ipynb` on 13 October 2026.

| Evaluation | Cases | Result |
|---|---|---|
| Router, rule-based | 43 labelled messages (8 categories, incl. mixed, adversarial, Arabic and French) | 41/43 (95.3%) |
| Router, LLM | same 43 | **43/43 (100%)**, mixed-intent secondary detected 2/3 |
| Agent, end to end | 19 multi-turn scenarios (every tool, leads, unknowns, memory, typos, scope, injection) | **18/19** — the remaining one was stopped by a Gemini service error, not an agent mistake |
| Reflection, rules only | 10 problematic and control drafts | 9/10 (inventing transport needs the LLM review) |
| Reflection, rules + LLM | same 10 | 9/10 |
| Offline test suite | `pytest tests.py` | 154 passed, 3 live tests skipped by default |

Notes on this run:

- **Models.** The preferred agent model (`gemini-3.8-flash`) was overloaded (HTTP 503) for the whole run, so the
  fallback chain answered: `gemini-3.6-flash` for the first scenarios, then `gemini-3.5-flash-lite` once that
  hit its daily quota. The router ran on `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`. These are the
  lighter models, so the results are a conservative estimate.
- **The one agent miss** ("over budget, then revised") got no answer: the model request failed after 72 seconds (the service was slow or overloaded). After
  this run the client was changed to hand a slow or overloaded model over to the next one within 50 seconds, and
  the evaluation now scores such cases as `api_error` instead of a failure.
- **The one LLM-reflection miss** was a test that was too strict: the reviewer correctly removed an unrecorded
  promise but also reworded "seafood", which the test required word for word. The test now checks the substance.
- No scenario saved an invented contact detail; the Python check refused none because the model never tried.

```bash
python -m pytest tests.py -q                       # offline: no API calls
RUN_LIVE=1 python -m pytest tests.py -q -s         # also runs the live router / agent / reflection checks
```
(PowerShell: `$env:RUN_LIVE="1"; python -m pytest tests.py -q -s`)

---

## Deploying to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `me/` folder (the notebook and tests are optional).
3. In the Space's **Settings → Variables and secrets**, add a secret named `GEMINI_API_KEY`. Never upload `.env`.
4. The Space builds and starts `app.py` automatically. The header at the top of this README configures it.

On Spaces, `logs/` lives on the Space's temporary disk and is cleared when the Space restarts.

---

## Limitations

- **Inquiries, not bookings.** There is no real availability calendar or payment; the team confirms by email.
- **Logs are local files.** Fine for a demo; a production version would send inquiries to a CRM or email.
- **The catalog is the limit of Alaa's knowledge.** Questions about transport, accessibility, or exact
  schedules are logged for the team rather than answered.
- **Free-tier quotas.** On the free Gemini tier, heavy use can exhaust the main model for the day; Alaa then
  answers with lighter backup models, which follow instructions slightly less precisely.
- **Itinerary travel times are not modelled**, only kept sensible by grouping each day within one area.

---

## Team

Sara (founder and experience curator), Ahmad (partnerships and operations), Khaled (guest experience: reviews
inquiries and confirms bookings), and Alaa, the AI concierge.
