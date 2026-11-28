# Hiredra architecture

What exists today: sourcing, the filter funnel, LLM skill extraction, the skill graph, and scoring with
calibrated thresholds and weights. The later stages (tailoring, autofill, the human gate, outreach,
tracking) are designed; first versions of tailoring, the desktop app and form preparation are being
built on a branch, and are marked so. The pipeline is Python; the desktop app will be TypeScript
(Electron + React) and talks to the pipeline through a local API.

## Stages and data flow

```
boards table ──> Source ──> jobs (fetched) ──> Filter ──> filtered | dropped
                                                  │
                                                  v
                          Extract (LLM) ──> job_skills, domain, legitimacy (extracted)
                                                  │
skill_vocab (graph) <── Graph sync <──────────────┤
        │                                         v
        └──────────── frozen during ──────> Score (CODE) ──> score + features (scored)
                                                  │
                                                  v
                     ranked list, review list ──> [Tailor -> Autofill -> human gate]   (planned)
```

- **One database** (Postgres 17 + pgvector, local) holds everything: jobs and each job's stage, the
  extracted skills, the skill graph with its vectors, board health, runs, cooldowns and daily API usage.
  Files are for config and exports only, never for state.
- **A job's status machine:** `fetched -> filtered -> extracted -> scored -> tailored -> queued ->
  applied -> outcome`, or `dropped` at any point, with the reason. Every change is a `job_events` row,
  so a job's history can be replayed.

## Source

- **Providers.** One small module per job platform: Workday, Greenhouse, Lever, Ashby, Oracle,
  Eightfold, Phenom, SmartRecruiters, iCIMS and about 20 more. Each returns the same `Job` object. JobSpy
  adds Indeed and LinkedIn searches.
- **Boards.** The `boards` table lists what to crawl. Each board has a cadence: daily, or weekly once it
  has had no India or remote job in 3 crawls. It also has a health record: 3 failures in a row rest it,
  for 6 h doubling to at most a week. One clean crawl heals it.
- **The conveyor belt** (`hiredra/sourcing/belt.py`). Every due board's first task is laid out before
  the run: each platform's boards longest first, the platforms interleaved so each starts at minute 0.
  - Big boards are cut into chunks of at most 50 requests. Workday's 2,000-result cap is split by facet
    slices.
  - A free worker takes the first task the rules allow:
    - a platform below its share is served first;
    - a platform held to a rate has at most `at_once` tasks in flight, and every request waits on its
      one pace, set at 60% of the measured refusal rate;
    - a host has at most its HTTP slots (2 by default).
  - Tasks are kept per host, so a full host is skipped in one step.
- **Errors.**
  - Can wait, and the task goes back on the belt after the server's Retry-After or a cooldown, up to 4
    times: a refusal, a time-out, a 5xx, Workday's weekly maintenance page.
  - Can't be done, and the board fails at once: gone (404), needs a login (401), other 4xx, a bug.
- **One writer.** A writer thread does every database write, in order: jobs, board results, cooldowns,
  progress. The dispatcher only hands out tasks, so it never waits on the database.
- **Dedup.** Within a run, then against every stored job, by exact keys only (no fuzzy matching): the
  URL, then company + role. The second only merges copies from different sources in the same countries:
  a false merge would lose a real job. A company's own posting wins over an aggregator's copy.
- **Closing.** A job not listed for 48 hours after a crawl that saw the whole board is soft-closed. It is
  never deleted, and reopens if listed again.

## Filter (no LLM)

1. **Location:** keep India, or remote with nothing excluding India. Unknown is kept, never guessed.
2. **Language:** a description not in English is dropped.
3. **Seniority:** reads the title and the job text: senior title words and level numbers ("III", "3"),
   years asked for, a senior opening heading, and phrases like "lead a team". Keeps fresher to 2-year
   roles, interns and trainees.
4. **Role type:** keyword and alias sets per role bucket (backend, AI agent, LLM, data).
5. **Liveness:** the board is listed again to confirm the job is still open, or the job is flagged
   "unchecked".

The funnel runs before any LLM call, so the LLM reads about 0.1% of what is fetched.

## Extract (the LLM reads)

- One schema-forced call per surviving job (`gemini-3.5-flash-lite`, free tier) returns its skills
  (required or nice-to-have), its domain bucket and a legitimacy flag. Code then grounds every skill in
  the job's text and drops what isn't there.
- When the day's Gemini budget is spent, the same run continues on an NVIDIA-hosted Nemotron model,
  chosen by a measured comparison of 33 models.
- **Quota model:** a per-minute pace, a daily cap with headroom, counters in the database, and a lock so
  two processes never share the budget unknowingly. A daily-quota stop is expected, not an error.

## Graph (bounded skill graph)

- Nodes are skills: the owner's own (`mine`) plus skills seen in real jobs. It is never a full tech
  ontology.
- **One embedded index:** each node's vector (Qwen3-Embedding-0.6B, run locally) is the vector used for
  similarity. There is no separate vector store.
- **Skill names are cleaned** as they arrive: versions and same-spelling variants become aliases, lists
  are split, junk is dropped. Each skill gets a one-line description.
- **Merges** (synonyms) are proposed by code and approved by the human. Vector similarity alone never
  merges two skills.
- **Freeze rule:** scoring holds the graph lock for its whole run, so the graph can't change mid-run.
  Skills new to the graph are scored by a fresh embedding and added between runs.

## Score (code decides)

```
score = w_domain * domain_fit + w_skills * coverage - w_gap * gap_distance
```

- **Per-skill fit:** a job's skill is matched to its nearest owner skill by cosine similarity `s`. At or
  above `HAVE` it counts fully; between `ADJ` and `HAVE` it counts as `s`; below `ADJ` it doesn't count.
- **Thresholds:** `HAVE` and `ADJ` are calibrated by an F1 sweep over hand-labelled skill pairs, scored
  once on a holdout.
- **Weights:** set from the owner's blind ratings of real jobs (pairwise rank agreement on a
  6,300-combination grid, scored once on held-out jobs), then frozen. Version 1 is in
  `config/weights.yml`; see `calibration/reports/step7-summary.md`.
- **The review gate:** a job flagged suspect or ghost by the legitimacy check, or with too few skills,
  goes to a review list, not the ranked list.
- **The apply line** is a quality floor, not a cap: every job above it is worked through in priority
  order.
- Every score stores its features, so a future model can learn the weights from real outcomes.

## Planned stages

- **Tailor:** a CV and cover letter that may only state facts from the owner's profile, checked by a
  code claim gate and a critic, written as an editable Word file plus a PDF.
- **Autofill:** reads the form, fills only verified facts, leaves a pause list for the human, and may
  click Next but never Submit.
- **The human gate:** batches of 10 in the desktop app's Apply workspace, reviewed and submitted by
  the human.
- **Outreach:** drafts only, held at a send gate.
- **Track and learn:** outcomes, yield per channel, and a learning loop whose changes the human approves.

## Tool choices, one line each

| Tool | Why |
|---|---|
| Python | One language for the pipeline and every agent. |
| Electron + React (desktop app) | One workspace to run and approve everything, with a controllable embedded browser. |
| Postgres + pgvector | One store for state and vectors; real locks and transactions. |
| psycopg 3, hand-written SQL | Small, explicit queries; no ORM to learn or fight. |
| Gemini free tier, NVIDIA fallback | Free structured output, with a measured fallback for quota days. |
| Qwen3-Embedding-0.6B, local | Free, no quota, and no network once cached. |
| httpx + tenacity | One shared client for retries, cooldowns and per-host slots. |
| YAML config | Owner-editable settings, loaded strictly: an unknown key is an error. |
| pytest, ruff, gitleaks in CI | Tests on every PR, with and without Postgres, plus a full-history secret scan. |
