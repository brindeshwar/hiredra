# CLAUDE.md — Hiredra

**Hiredra** (styled "Hire-dra" / "HireDra" in prose) — one system, many agents. Each head handles
a piece — sourcing, scoring, tailoring, outreach — and hands you a batch of ready-to-send
applications for your approval.

This file is the project constitution. Read it fully at the start of every session and treat it
as the source of truth. If an instruction in a chat conflicts with this file, ask before acting.
Detailed designs live in `/docs` — read the relevant one before building that part.

---

## 0. What this project is

A pure-Python system that takes the owner from "no jobs" to tailored job applications queued for
human approval. It sources jobs widely, filters them cheaply, scores them against the owner's
profile, tailors a CV + cover letter per job, autofills the application form, drafts outreach,
and tracks everything.

**The system automates everything up to the click. It never submits or sends anything itself.**
The human reviews applications in batches of 10 and clicks Submit/Send personally.

**Important: "10" is the human review-batch size, NOT a cap on how many jobs are processed.**
The intent is that pre-scoring filters are strong enough that every job reaching scoring is
apply-worthy; scoring then RANKS them into priority order (it is not a cut). The expensive stages
work through the FULL ranked list in priority order over time, paced to the free token budget —
best jobs first — never discarding survivors. See section 3.5 (Scale).

This project's purpose is to be a **best-in-class portfolio artifact** that demonstrates agentic
AI engineering. It is NOT optimized for speed-to-apply (a separate working system already handles
day-to-day applying). Optimize for correctness, clarity, and defensibility.

**Definition of "best" for this project:** every component is the right tool for its job, each
choice is defensible in one sentence, and complexity appears only where it earns its place.
Cutting unnecessary complexity is the goal, not adding sophistication. When in doubt, choose the
simpler option and write down why.

---

## 1. Hard rules (never violate; do not "improve" around these)

1. **Nothing auto-submits or auto-sends.** No code path may click Submit, Send, Apply-confirm,
   or post anything to a third party. The system prepares; the human commits. This is the core
   design principle, not a limitation.
2. **Free runtime stack only.** All services the running system depends on must be free-tier
   (Groq, Gemini free tier, public ATS APIs, web search). No paid APIs, no paid data tools, no
   paid people-finders. (The build tool itself running on a subscription is fine — that's not the
   runtime.)
3. **LLM reads, code decides.** The LLM is only used to read and normalize messy text (extract
   skills, classify domain, flag legitimacy). Every number, score, threshold, and decision is
   computed by deterministic code. The model must never emit a score or make a ranking decision.
4. **Every claim must be interview-defensible.** No fabricated experience, skills, or metrics
   anywhere — in scoring, tailoring, autofill, or outreach. The tailoring/critic agents must only
   assert facts present in the owner's master profile.
5. **No secrets in code or git.** All credentials/keys go in `.env` (git-ignored). Never hardcode
   or commit a secret. Never store a plaintext password in any file.
6. **Human-in-the-loop gate is mandatory** before any irreversible or external action
   (submit, send, publish, delete, form-submit, accept-terms).

---

## 2. How to work in this repo

- **Start in Ask mode.** Propose diffs and commands; wait for approval. Do not batch large
  autonomous changes.
- **Build one slice at a time**, in the order given in section 6. Do not scaffold the whole
  system at once. Finish, review, and confirm a slice before starting the next.
- **Prefer the simplest thing that works.** If you're about to add a library, a graph, a model,
  or an abstraction, first check whether a simpler mechanism already specified here suffices. If
  you believe added complexity is justified, stop and ask.
- **Do not re-introduce rejected approaches.** See `docs/decisions.md` for the list of things we
  deliberately did NOT do and why. If you find yourself proposing one, that's a signal to stop.
- **When a design question arises that isn't answered here or in `/docs`, ask — do not guess.**
  Architecture decisions are made by the owner, not improvised during building.
- **Keep this file and `/docs` current.** When a new decision is made, record it before moving on.

---

## 3. Architecture overview (the pipeline)

Deterministic stages use code only. Agent stages use an LLM for reading/normalizing, never for
scoring.

```
Profile & verified facts   (source of truth: master profile file)
        │
Source & filter            CODE  — JobSpy service + Python ATS providers; dedup; title filter;
        │                          seniority gate (reads JD body); liveness check
Score                      CODE+LLM — LLM extracts skills+domain+legitimacy; CODE computes score
        │                          (see docs/scoring.md)
Enrich (promoted only)     AGENT — research (company context) + contact (find HM/recruiter)
        │
Tailor CV + cover          AGENT — tailoring agent → JSON → deterministic renderer → PDF;
        │                          critic agent; page-fit loop  (see docs/pipeline.md)
Autofill + draft answers   AGENT — Playwright fills verified facts only; pause list
        │
HUMAN GATE                 — batch of 10; human reviews each, approves together, clicks submit
        │
Track + learn              CODE — ledger (single source of truth) + per-channel yield +
                                  learning loop (proposes changes, human approves)
```

Standalone, NOT part of the pipeline: a Naukri profile-refresher utility (see docs/pipeline.md).

---

## 3.5 Scale (design for this from the start)

Two numbers in this system are different and must not be conflated:

- **Jobs processed per run: hundreds to thousands.** Sourcing pulls wide. The cheap stages
  (source → filter → extract → score) MUST be built to handle this volume:
  - Stream/batch through the funnel; do not load the entire job pool into memory at once.
  - Filter hard and early (seniority, title, role) BEFORE any LLM call, so the LLM only ever
    sees the survivors.
  - Make all LLM calls rate-limit aware (respect Groq/Gemini RPM/TPM/RPD; back off and resume).
  - Persist progress so a run can stop and resume without redoing work.
- **Jobs taken through the expensive stages (tailor, autofill): the full ranked survivor list,
  worked in priority order, paced to the token budget.** There is NO top-N cut. Scoring ranks;
  the expensive stages drain the ranked list highest-priority-first until the list is done or the
  daily budget is spent, resuming next run. Batches of 10 are only the human review unit.

Countermeasure to budget limits is **pacing + priority order**, never capping. If survivors
consistently exceed daily tailoring budget, the levers (in order of preference) are: tighten
filters, accept the paced drain, then the provider/token split.

PARKED idea (do not implement without owner sign-off): cycling multiple API keys/accounts to
multiply free-tier limits. FLAG when this is reached: rotating keys/accounts to bypass free-tier
limits is commonly against provider Terms of Service (Groq/Gemini) and risks account bans, and is
hard to defend in an interview. Weigh against the cleaner countermeasures above before adopting.

The graph is built once and queried at scale (frozen during a run); it is not rebuilt per job.

---

## 4. Scoring engine (summary — full detail in `docs/scoring.md`)

The funnel removes different noise cheaply before spending any LLM tokens:

1. **Seniority gate** — CODE. Reads the JD body (not just the title): regex for years +
   anti-signal phrases ("mentor junior engineers", "lead a team"). Drops anything above 0–2 yr.
   Target: fresher / 0–1 yr / roles open to 0–2 yr; interns and trainees are INCLUDED. Seniority
   is a gate only — it is NOT a scoring factor.
2. **Role-type filter** — CODE. Keyword/alias sets per bucket (backend / agent / LLM / data).
   Flat membership test. Not a graph.
3. **Legitimacy** — LLM, bundled into extraction. Flags ghost/spam/vague postings.
4. **Skills extraction** — LLM. Extracts required skills, normalizes to canonical names, tags
   each skill required vs good-to-have, and classifies the job into one domain bucket. Extraction
   only — no scoring. Use strict/structured output (JSON) via direct HTTP.
5. **Score** — CODE.

**Formula:**
```
score = w_domain * domain_fit
      + w_skills * coverage
      - w_gap   * gap_distance
      + legitimacy_flag
```
- `domain_fit` = trinary step {0, 0.5, 1} from the LLM's domain bucket, mapped via a config the
  owner controls (agent/LLM = 1, backend = 0.5, data/ML = 0.5, generic/other = 0).
- `coverage` = threshold-then-average of the per-skill fit vector (count of skills you clear).
- `gap_distance` = zero-the-wins-then-magnitude of the shortfalls (size of the holes; keeps
  per-skill distance so a near gap hurts less than a far gap).
- good-to-have skills are down-weighted by factor `x` in both coverage and gap.

**Per-skill fit** (needs the graph): for each required skill, find its single nearest `mine`
skill by cosine similarity `s`, then map:
- `s ≥ HAVE` → fit = 1.0
- `ADJ ≤ s < HAVE` → fit = s
- `s < ADJ` → fit = 0.0
Embed the JD skill fresh; never require it to already be a graph node.

**Two calibrations, both required, in this order:**
- **Threshold calibration** — set `HAVE` and `ADJ` for the chosen embedding model via an
  F1-maximizing sweep over ~50 hand-labeled skill pairs. Makes matching accurate.
- **Weight calibration** — set `w_domain`, `w_skills`, `w_gap`, and the apply/skip line by
  hand-rating ~15 real jobs (apply/maybe/skip) and tuning until the ranking matches the owner's
  judgment. Makes the score trustworthy.
Weights/thresholds are frozen between calibrations; re-calibrated deliberately, not every run.

**Data logging for future learning:** the ledger stores the scoring features (domain, coverage,
gap_distance, fit vector, extracted skills) AND the eventual outcome (applied / reply / interview
/ rejection) per job, from day one. A future v2 may learn the weights from real outcomes
(logistic regression first — NOT a neural net by default), gated on ~150–200 labeled outcomes.
Model choice is deferred; only the data collection is committed now.

---

## 5. Knowledge graph (summary — full detail in `docs/graph.md`)

A capability graph BOUNDED to the owner's skills + skills actually seen in real JDs. Never a full
tech ontology.

- Each skill is an object: `name`, `aliases`, `mine` (bool), `level`
  (production/built-with/learned/interview-ready, only if mine), `vector`, `edges`.
- **Target = embeddings + edges, built in two phases:**
  - **Phase A (build now):** everything except edges; the `edges` field exists but stays empty.
    Scoring reads only `vector` and `mine`.
  - **Phase B (build last):** an LLM pass fills `edges`; unlocks "what to learn next", eligibility,
    portfolio. Gated on Phase A + weight calibration proving out.
- **ONE embedded index, not a graph plus a separate vector store.** Node embeddings ARE the
  vectors used for similarity and gap distance.
- **Build by backfill:** one wide UNFILTERED scrape → extract skills → normalize/dedup to
  canonical → add as `market` nodes → embed and place → overlay the master profile as `mine`
  nodes. Backfill is refreshed periodically (new skills are rare right after a backfill).
- **Overlay match** (profile skill → market skill): exact canonical-name match first, then a
  high-threshold vector check (≥0.9) for synonyms. Never auto-merge merely-similar skills.
- **Source-of-truth rule:** the owner edits the MASTER PROFILE FILE; the graph is rebuilt from it.
  Never edit capability directly in the graph.
- **Freeze rule:** the graph is frozen during a scoring run for score stability. New skills seen
  mid-run are scored via a fresh embedding and queued; nodes are added between runs, not mid-run.

---

## 5.5 Source repos — what to harvest (verdicts fixed; detail in `docs/repos.md`)

We reuse aggressively rather than rebuild, with credit when published. The VERDICT per repo is
fixed by the owner and must not be changed. The DETAIL of what each contains is to be produced by
reading the actual cloned repos (see task note below) into `docs/repos.md`.

1. **career-ops** (github.com/santifer/career-ops, MIT) — the main resource.
   - PORT to Python: the ATS provider modules (public ATS JSON APIs, no keys) and the sourcing/
     scan + liveness logic.
   - REUSE (as Node, via subprocess) and adjust: the CV renderer (JSON → HTML template → Playwright
     PDF) and its tracker/repost-detection helpers.
   - REBUILD as our own agents: its "agents" are prompt-mode files run by an AI CLI, NOT callable
     scripts. Do not try to invoke them. Only the deterministic `.mjs` helpers are reusable.
2. **OpenJobs** (github.com/outscal/OpenJobs, MIT fork) — the wide net.
   - PORT: the harvester and the ATS adapters and the ATS-probe/discovery scripts.
   - Its company dataset is gaming-skewed; use it FILTERED to India + tech only, and rely on our
     role filter (not the dataset) to find AI/backend roles.
3. **job-application-agent** (github.com/vaibhavarora14/job-application-agent, MIT) — DESIGN ONLY.
   - Do NOT run it (it's an agent-skill). COPY its design into our autofill/gate: the safety pause
     list, OS-secret-store handling, "record submitted only after visible success", and the
     every-10-applications learning-loop.
4. **wellfound_autoApply** (github.com/ankitbaghel01/wellfound_autoApply) — STUDY ONLY, NO CODE.
   - UNLICENSED — do not copy or adapt any of its code (a modified copy is still a derivative).
   - It auto-submits (forbidden here) and hits a DataDome CAPTCHA. Read it only as confirmation
     that browser auto-apply is a dead end. Reimplement the saved-session idea from scratch if
     needed.
5. **naukri_update** (github.com/ankitbaghel01/naukri_update) — REWRITE as a standalone utility.
   - It's a Naukri profile-refresher, not a job applier. Rewrite cleanly with a saved browser
     session. NEVER copy its plaintext-password-in-.env approach; no password in any file.

TASK NOTE (a good early Claude Code task): when starting repo work, FIRST set up the
codebase-graph MCP (see tech note in section 7) so the large repos are read token-efficiently.
THEN clone repos 1–3, read the actual code, and WRITE `docs/repos.md` mapping exactly what each
reusable piece does and how we'd port/call it — WITHIN the verdicts above. This is reading/
reporting, not deciding; do not change the verdicts, and flag anything that contradicts them for
the owner.

---

## 6. Build order

Build in this sequence. Each item is a slice to finish and review before the next.

1. **Project skeleton** — folders, `.env` + `.gitignore`, config files, `git init`.
2. **Sourcing & filtering** — JobSpy service integration + Python ATS providers (ported), dedup,
   title filter, seniority gate, liveness. (Does not need the graph.)
3. **Skill extraction (LLM)** — canonical skills + required/good-to-have + domain bucket +
   legitimacy, structured JSON output.
4. **Graph Phase A** — Skill object, backfill scrape, normalize, embed, profile overlay, the
   per-skill fit mechanic. Empty edges.
5. **Threshold calibration** — the F1 sweep to set HAVE/ADJ for the chosen embedding model.
6. **Scoring** — the formula, coverage, gap_distance, domain step, legitimacy, ledger with
   feature+outcome logging.
7. **Weight calibration** — rate ~15 jobs, tune weights.
8. **Resume generator** — tailoring agent → JSON → renderer → PDF; critic agent; page-fit loop.
9. **Autofill + notify-and-resolve gate** — Playwright fill, pause list, notifications, batch-10.
10. **Outreach** — draft-only, held at send-gate.
11. **Track + learn** — ledger analysis, per-channel yield, learning loop (human-approved).
12. **Naukri refresher** — standalone utility.
13. **Phase B (graph edges)** — after Phase A + calibration prove out.

Deferred decisions (do not block on these; ask the owner when reached): streaming vs buffered
apply gate (decide by experiment), provider/token split (Groq vs Gemini per agent), embedding
model choice, learner model choice for v2.

---

## 7. Tech constraints

- Language: Python for the orchestrator and all new components.
- The CV renderer is reused from an existing Node.js tool (career-ops) and ADJUSTED, not
  rewritten. Call it from Python via subprocess. Node is allowed only for this renderer.
- Browser automation / PDF rendering: Playwright.
- LLMs: Groq (gpt-oss, strict structured output via direct HTTP) and Gemini free tier. Exact
  split deferred.
- Storage: start with simple files (JSON/markdown/xlsx) as already used elsewhere; do not
  introduce a database unless asked.
- Secrets in `.env`, git-ignored, from the first commit.

**Tooling note — codebase-graph MCP (set up at the repo-porting stage, before reading the big
repos):** when repo work begins (the section 5.5 task, and any later reading of the large external
repos), set up a codebase-graph MCP server FIRST so Claude Code navigates those big repos by
querying a structural graph instead of re-reading files every session (large token savings on
500+ file repos). Candidates: `code-review-graph` or `Claude Context` (zilliztech). This is
PRIMARILY for the external repos and for when THIS repo grows large; it is NOT needed for the
initial small skeleton, so do not set it up on day one. NOTE: this "codebase graph" is unrelated
to the project's skills knowledge graph (section 5) — different thing, same word.

---

## 8. Detail docs (read before building the relevant slice)

- `docs/scoring.md` — full scoring engine: funnel, formula, fit mechanic, both calibrations.
- `docs/graph.md` — Skill object, phases, backfill, overlay, freeze rule, fit thresholds.
- `docs/pipeline.md` — enrichment, resume generator, autofill + gate, outreach, track+learn,
  Naukri utility.
- `docs/repos.md` — the source repos: exactly what to port/reuse/rebuild from each, within the
  fixed verdicts in section 5.5. (To be generated by reading the actual cloned repos.)
- `docs/decisions.md` — rejected approaches and why (do not re-introduce these).

If any of these files is missing, ask the owner before improvising that part.
