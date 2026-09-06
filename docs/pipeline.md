# docs/pipeline.md — Downstream pipeline (full detail)

Project: **Hiredra**. Read this before building enrichment, resume generation, autofill/gate,
outreach, tracking, and the Naukri utility (build steps 8–12). Everything here obeys the hard
rules in `CLAUDE.md` section 1 — especially: nothing auto-submits/auto-sends, free stack only,
interview-defensible.

The expensive stages below run in priority order on the ranked survivor list, paced to the token
budget (see `CLAUDE.md` 3.5). They do not run on a fixed "top 10" — 10 is only the review batch.

---

## 1. Enrichment (agents; promoted jobs only)

Run only on jobs that scored high enough to be worth the tokens.

### Research agent
- Web search for company context (what they do, product, tech, recent news) so cover letters and
  outreach are specific, not generic.
- Free web search only. No paid tools.

### Contact agent
- Find a hiring manager / recruiter / relevant person for the role.
- Free web search only. No paid people-finders.

---

## 2. Resume generator (agents + deterministic renderer + owner's novel loop)

Built from scratch — no existing resume format is reused (the prior system never generated
resumes). The renderer is reused from career-ops (Node) and adjusted; everything else is ours.

Flow:
1. **Tailoring agent** — reads the JD + the master profile; emits the tailored CV content as a
   JSON payload (NOT HTML). May only assert facts present in the master profile
   (interview-defensible rule). `interview-ready`-level skills must not be presented as production
   experience.
2. **Renderer (deterministic, reused Node, called via subprocess)** — merges the JSON into a FIXED
   one-page ATS-safe HTML template → Playwright → PDF. The template owns all markup/spacing so the
   format is identical every time. Base the template's STRUCTURE on the owner's known-good resume
   layout (single column, standard fonts, one page) so it stays ATS-parseable.
3. **Critic agent (adversarial, separate from the tailoring agent)** — checks for: skills/claims
   not in the master profile (hallucination), generic filler, missed JD keywords. Routes failures
   back to the tailoring agent.
4. **Page-fit loop (owner's piece)** — render → measure page count / overflow → revise content →
   re-render until it fits one page. career-ops does NOT do this; it is genuinely ours.
5. Optional: also emit a plain-text version for ATS boxes that mangle PDFs.

Cover letter: generated from tailoring + research output, through the same render pipeline. Held as
a draft (never sent by the system).

---

## 3. Autofill + notify-and-resolve gate

### Autofill agent (Playwright)
- Opens the application form, fills every field it can FROM VERIFIED FACTS ONLY (master profile +
  structured profile), attaches the tailored CV, drafts the free-text answers.
- ATS APIs are read-only (they give postings, not a submit endpoint), so the actual application
  goes through the company's web form — hence Playwright here.
- Reimplement the saved-browser-session pattern from scratch (do NOT fork wellfound's unlicensed
  code; do NOT store a plaintext password — log in once, reuse the session).

### Pause list (design copied from job-application-agent)
Pause and hand to the human on any of: CAPTCHA, login/SSO/MFA, legal attestations, demographic
questions, work-authorization/sponsorship questions, salary questions, government IDs, unverifiable
claims, or any field NOT derivable from verified facts.

### The gate (mandatory)
- Applications queue. When 10 are ready, the human reviews EACH (drafted answers + filled form),
  approves the batch together, and clicks Submit personally.
- The system NEVER clicks Submit. Record a job as "submitted" ONLY after a visible success
  confirmation.

### Notify-and-resolve
- The system runs on its own and pings the human only when: a batch of 10 is ready, OR it hits a
  pause (CAPTCHA / blocked field / error).
- Notification channel: DEFERRED (Telegram / email / desktop). Decide when building this stage.
  The review surface (CLI / local page) is decided here too — start simplest (CLI) unless the owner
  wants the local page for demo value.

---

## 4. Outreach (agent; held at send-gate)
- Drafts a recruiter/hiring-manager message from the research context (e.g. a short LinkedIn note).
- Held at the human send-gate. The human sends. The system never auto-sends.

---

## 5. Track + learn

### Ledger (single source of truth)
- One record per job: scoring features + drafted artifacts + application status + outcome.
- Start with a simple file (JSON / markdown / xlsx). No database unless asked.
- This is also the training-data store for the future weight-learning v2 (docs/scoring.md 7).

### Analysis
- Per-channel yield (which sources/ATSes convert), rejection patterns.

### Learning loop (human-approved)
- Every ~10 applications, review outcomes and PROPOSE changes: targeting tweaks, prompt
  improvements, block-list additions.
- The owner approves; only then does shared memory update. The loop never changes preferences
  on its own.

---

## 6. Memory model
- **Shared memory** (durable, read by all agents, written ONLY via the approved learning loop):
  master profile / graph, canonical skills + aliases, calibration weights + thresholds,
  block-list, prompt versions.
- **Specific memory** (per-job working record, passed down the pipeline, FLUSHED on batch
  approval): one job's extracted evidence → score → drafts → status.

---

## 7. Naukri refresher (standalone utility — NOT part of the pipeline)
- Purpose: keep the owner's Naukri profile "recently updated" so recruiters surface it (India
  market recruiters sort by recency).
- Mechanism: periodically toggle a trivial change (e.g. a trailing dot on the headline) so it
  counts as a profile update. Saved browser session (log in once); off-screen browser (Naukri
  blocks headless); scheduled (e.g. Windows Task Scheduler).
- Clean rewrite of naukri_update. NEVER store a plaintext password in any file (the original does;
  do not copy that). Session-based only.
- Note: this may touch Naukri ToS — the owner runs it at their own discretion; it is isolated from
  the main system.
