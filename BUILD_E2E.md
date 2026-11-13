# Hiredra end-to-end build: instructions for any agent

This file is the complete brief for finishing Hiredra's end-to-end build: the CV builder, the
application filler and the desktop app. It is written so that an agent or model with no earlier
context can do the work. Read all of it before writing code. When something here conflicts with your
own habits, this file wins. When it conflicts with a hard rule (section 2), the hard rule wins.

Owner's brief (2026-11-13): "build it properly but don't cover edge cases". The goal is a **fully
functional** system that works end to end on about 50 real jobs. Ideal and perfect come later.
Refinement, platform coverage, and weight tuning are explicitly NOT part of this build. The owner
will run it themselves later.

---

## 1. Merge rule (read this first)

- All work lands on the branch **`build/e2e`**. `main` is never touched during this build.
- Each piece (section 6) gets its own branch, `build/pN-<short-name>`, created from the latest
  `build/e2e`, and its own pull request **into `build/e2e`**.
- A piece PR may be squash-merged into `build/e2e` once its CI is green (the owner allowed this:
  "merge pieces"). Check CI once; never poll it in a loop.
- **`build/e2e` is merged into `main` ONLY when the owner writes that they are satisfied** after
  reviewing the work (planned for the evening of 2026-11-14). Not before, not on a guess, not because
  CI is green. "Looks good", "continue" or "merge pieces" do not count. Wait for an explicit statement
  of satisfaction with the build, then open one PR `build/e2e` -> `main` and merge it on the owner's
  word.
- Never force-push `main`. Never rewrite history on `build/e2e`.

---

## 2. Hard rules (never violate, never "improve" around them)

1. **Nothing submits or sends.** No code path may click Submit, Send or Apply-confirm, press Enter
   in a form field, call `form.submit()`, dispatch a submit event, or post anything to a third party.
   The system prepares; the human clicks Submit. This is the design, not a limitation.
2. **Free runtime only.** Free tiers and local tools only. No paid API, data tool or people-finder.
3. **The LLM reads, code decides.** A model may write text (bullets, a cover letter, a drafted
   answer) and read text. Every number, score, threshold, decision and gate is code.
4. **No invented claims.** Nothing generated may claim a skill, fact, number or experience that is
   not in the owner's master profile. The claim gate (piece 2) enforces this in code.
5. **No secrets in code or git.** Keys live in `.env` (git-ignored). Never print, log, commit or put
   a key in an error message. Never store a password anywhere.
6. **Human gate** before any irreversible or external action.
7. **The repo is public.** No personal data in tracked files, commit messages, PR text or issue
   text: no name, email, phone, profile content, ratings or application details. Personal data lives
   only under `data/` (git-ignored).
8. **Commits:** never add a `Co-Authored-By` trailer or any AI attribution to commits or PR text in
   this repo. Commits are authored by the owner only.
9. **No live form filling during this build.** The filler runs only against local fixture forms
   (HTML files in the repo). Reading public job pages and public question APIs with GET is fine.
   The first fill on a real site happens later, with the owner watching.
10. **Do not spend Gemini quota.** All new model calls in this build use the NVIDIA hosted model
    (section 4.4). Gemini stays for extraction and description only.

---

## 3. Decisions already made (do not reopen)

| Topic | Decision |
|---|---|
| Résumé file | An editable **Word .docx** is the main output, plus a **PDF** made from it. It replaces the earlier HTML-to-PDF plan. |
| Résumé layout | Follows the owner's example file (private: `data/profile/resume-example/example.docx`); the renderer already does (piece 1). |
| Text writer | NVIDIA hosted model `nvidia/nemotron-3.5-super-vl-120b-a12b` through `hiredra/nvidia.py`. Switchable later in config. |
| Desktop app | Electron + TypeScript + React, with the Python pipeline as a local FastAPI service (the "sidecar"). Built for adding features later. |
| Weights | Placeholders (`config/weights.yml`, `apply_line: null`). No tuning in this build. |
| Platform coverage | Generic, not tuned per site. Greenhouse and Lever form reading first. Ashby, Workday and the rest come later. |
| Edge cases | Out of scope. Works on about 50 jobs = done. |
| Landing | One branch `build/e2e`, a PR per piece, merge to `main` only on the owner's satisfaction (section 1). |

---

## 4. The system as it is today

### 4.1 Pipeline (built, on `main` and `build/e2e`)

```
sourcing crawl -> filter (location, seniority, role, liveness) -> extraction (LLM skills)
  -> graph sync (embeddings) -> graph.describe (descriptions, same-skill folds)
  -> scoring (code) -> graph audit (health checks)
```

Commands (from the repo root):

```bash
python -m hiredra.sourcing crawl
python -m hiredra.filter
python -m hiredra.extraction
python -m hiredra.graph sync
python -m hiredra.graph.describe
python -m hiredra.scoring run
python -m hiredra.scoring list --all
python -m hiredra.graph audit
```

### 4.2 Storage

- Postgres 17 + pgvector in Docker (`docker compose up -d`, localhost only, password in `.env`).
- psycopg 3 and hand-written SQL in `hiredra/storage/db.py`; no ORM.
- Schema changes are numbered files in `migrations/` (the last one is `0007_skill_description_source.sql`).
  A new migration is a new numbered file; never edit an applied one (their checksums are enforced).
- `init_db(conn)` applies pending migrations.
- `jobs.status` values (a CHECK constraint, do not add new ones): `fetched`, `filtered`, `extracted`,
  `scored`, `tailored`, `queued`, `applied`, `outcome`, `dropped`. Change status only through
  `hiredra.storage.set_status(conn, job_id, status, detail=...)`, which also writes a `job_events`
  row.
- Useful functions: `connect`, `init_db`, `get_job`, `select_jobs`, `set_status`, `add_note`,
  `status_counts` in `hiredra.storage`; `promoted`, `all_scored`, `review_list` in
  `hiredra.scoring.run`.
- Locks: `hiredra.lock.RunLock(...)` (Postgres advisory lock). Model-calling steps take the `llm`
  lock; a held lock means exit code 4 and "NOT RUN".

### 4.3 Piece 1, already merged into `build/e2e` (PR #177)

`hiredra/resume/`:

| File | What it does |
|---|---|
| `payload.py` | `CvPayload`, `CoverLetter` and their parts (pydantic, `extra="forbid"`). `Experience.about`, `Project.tagline`, `Project.tech`. `**term**` in text = bold. |
| `word.py` | `render_cv(payload, path, layout)` and `render_cover(letter, path)`: the owner's example layout (US Letter, Arial 9.5 pt, sections Summary / Technical Skills / Experience / Projects / Education). `Layout` holds every size; `Layout.tighter()` steps it down. |
| `pdf.py` | `to_pdf(docx)`: LibreOffice if installed, else Microsoft Word (Windows, PowerShell automation). `page_count(pdf)`. `PdfUnavailable` if neither. |
| `build.py` | `write_cv(payload, out_dir)`: render, convert, count; tighten the layout, then drop whole bullets until it fits one page. `write_cover(letter, out_dir)`. |
| `source.py` | `read_profile()` reads the master profile's entries (Experience, Flagship project, Projects), bullets, `Stack:` lines, links, education, the "For tailoring only" guidance and the standing rules. `load_candidate()` reads `data/autofill_facts.yml` (`contact:` block). `baseline_payload()` makes a résumé from the profile's own words with no model. `format_dates()`, `first_sentence()`. |
| `__main__.py` | `python -m hiredra.resume sample` and `python -m hiredra.resume render PAYLOAD.json`. |

Tests: `tests/test_resume_core.py` (uses `tests/fixtures/profile_resume.md`, a fake profile).

### 4.4 Models and quotas

- Gemini flash-lite: 450 requests/day shared by extraction and describe. **Do not use in this build.**
- NVIDIA hosted: `hiredra.nvidia.NvidiaChain(model, schema)` with `.invoke(prompt) -> schema | None`.
  The key is `NVIDIA_API_KEY` in `.env`. Pace under 40 requests/minute. Count requests with
  `hiredra.quota.DailyUsage(10000, conn, "nvidia", model, count_errors=False)` (the owner's guard:
  10,000 accepted requests a day).
- Retries follow the owner's rule: on 429, 5xx or a network error keep retrying with
  `hiredra.quota.jittered_wait` (1, 2, 4, 8 s doubling plus jitter, capped at 120 s) and
  `stop=stop_never`; only the daily budget stops a run. Copy the pattern in
  `hiredra/graph/describe.py` (`Retrying(retry=retry_if_exception(lambda e: classify_failure(e) == "transient"), ...)`).

### 4.5 Private files the code reads (all under `data/`, git-ignored)

| Path | Content |
|---|---|
| `data/master_profile.md` | The owner's master profile: the single source of truth for every claim. Sections: "For tailoring only" (never printed), Education, Skills (Strong / Working / Familiar, strict grammar), Experience, Flagship project, Projects, standing rules. |
| `data/autofill_facts.yml` | Contact facts (`contact:` block). Shape: `config/autofill_facts.example.yml`. |
| `data/profile/resume-example/example.docx` | The owner's example résumé (layout reference only). |
| `data/packets/` | Output folder for packets. |

Only Strong and Working skills may be asserted in generated text. Familiar skills are for matching
only. Read the profile's "Standing rules" section and obey it (one page, no em dashes, no Oxford
commas, important terms bold, never "full-stack" framing).

---

## 5. Working environment

- Windows. Use Git Bash for POSIX commands, PowerShell where noted. Python is the repo's `.venv`
  (Python 3.12, managed by `uv`). Add a dependency with `uv add <pkg>`; commit `pyproject.toml` and
  `uv.lock`.
- Node 24 and npm are installed (for the app in `app/`).
- Microsoft Word is installed (used for PDF conversion). LibreOffice is not.
- **Shell gotcha:** a bash heredoc with non-ASCII characters (é, —, –) breaks with
  "unexpected EOF". Write such files with a file-writing tool, not a heredoc.
- Before every commit:

```bash
.venv/Scripts/ruff.exe format .
.venv/Scripts/ruff.exe check .
.venv/Scripts/python.exe scripts/ci_guards.py
.venv/Scripts/python.exe -m pytest -q
```

- Database tests run only when `HIREDRA_TEST_DATABASE_URL` points at a Postgres server that may
  create and drop databases (the same server as `.env`'s `HIREDRA_DATABASE_URL`, database
  `postgres`). Build that URL in a small script; never print it (it holds the password). Run pytest
  with `--tb=short` so a failure never echoes the URL.
- CI (`.github/workflows/ci.yml`) runs on every pull request: ruff, ci_guards, tests, tests on
  Postgres, gitleaks. CI guards: G-a only `hiredra/autofill/driver.py` may import Playwright inside
  `hiredra/autofill/` and `hiredra/gate/`; G-b nothing there sends POST/PUT; G-c no mail or
  messaging client anywhere in `hiredra/`. Keep them passing.
- Style: lean code, functions over classes, no speculative abstraction, a docstring that says why.
  Tests next to every piece. Write user-facing text and docs in plain English.
- Back up before any destructive database step: `python scripts/backup_private.py --with-db`.

---

## 6. The pieces

Do them in order. Each ends with: tests pass, a PR into `build/e2e`, CI green, squash-merge into
`build/e2e`, a line in the progress log (section 9).

### Piece 2: tailoring, the claim gate, the cover letter, packets

**Goal:** for a scored job, write a tailored résumé and cover letter (Word + PDF) into a packet
folder, using only facts from the master profile, and move the job to `tailored`.

Files to create:

- `config/skill_display.yml` (tracked; public-safe because it only maps skill tokens): every Strong
  and Working token of the profile -> `{name: "<display name>", category: "<one of the categories>"}`.
  Categories, in this order: `Languages`, `AI / LLM`, `ML / NLP`, `Backend`, `Infra & Tools`.
  Tokens missing from the file go to `Infra & Tools` with the token as the name.
- `config/resume.yml`: `provider: nvidia`, `model: nvidia/nemotron-3.5-super-vl-120b-a12b`,
  `requests_per_minute: 35`, `max_projects: 4`, `max_bullets_per_experience: 5`,
  `max_bullets_per_project: 3`, `max_jd_chars: 6000`.
- `hiredra/resume/skills.py`: **code builds the skills section** (no model): Strong and Working
  tokens only, grouped by `skill_display.yml` categories, inside each group the job's own skills
  first (from `get_job(...)["skills"]`, resolved through the alias table), then the rest in profile
  order, at most about 12 items per group.
- `hiredra/resume/tailor.py`:
  - Model output schema (pydantic): `summary: str`; `experience: list[{ref: str, bullets: list[str]}]`;
    `projects: list[{ref: str, tagline: str, tech: str, bullets: list[str]}]`. `ref` is the profile
    entry's heading text exactly as written after `### `.
  - The prompt holds: the job title, company and description (cut to `max_jd_chars`); every profile
    entry as `### heading` + intro + bullets + `Stack:`; the guidance lines; the standing rules; the
    output rules (rephrase and reorder only what the entry says; bold the terms that match the job
    with `**term**`; no number that is not in the entry; no skill that is not in the entry or in the
    Strong/Working list; past tense for finished work; one line about the company comes from code,
    not from you).
  - Code assembles the `CvPayload`: `candidate` from `load_candidate`; company, role, dates and
    `about` from the profile entry named by `ref` (never from the model); education from
    `education_items`; skills from `skills.py`; summary and bullets from the model.
- `hiredra/resume/claims.py`, **the claim gate** (pure code, used by the CV, the cover letter and
  drafted form answers). Input: generated text pieces with where they belong (`"experience[0].bullets[2]"`)
  and the profile. Checks:
  - `bad_ref`: a `ref` that is not a real profile heading -> fail.
  - `new_number`: every number in a piece (`\d[\d,.]*%?`, also `x` multipliers like `50x`) must
    appear in the referenced entry's text (for the summary and the cover letter: anywhere in the
    profile) -> otherwise fail.
  - `unsupported_skill`: every skill-vocabulary name found in the text (match `skill_vocab` names and
    aliases on word boundaries, case-insensitive, names of 3+ characters) must be a Strong/Working
    token, or appear in the referenced entry's text -> otherwise fail. Familiar skills -> fail.
  - `style`: an em dash or en dash -> fix it in code (`word.clean` already does), not a failure.
  - Result shape: `{"hits": [...], "pass": bool}` with one entry per finding (`rule`, `where`, `detail`).
- Flow per job: call the model -> claim gate -> if it fails, call the model once more with the
  findings listed -> claim gate again -> any piece that still fails is replaced by the profile's own
  bullets for that entry (`baseline_payload` logic) and the packet is flagged `critic_unresolved`.
  Never ship a failing piece.
- `hiredra/resume/cover.py`: a `CoverLetter` with 3 short paragraphs (why this role, two pieces of
  evidence from the profile, a close), same claim gate, same fallback (a plain letter built by code
  from the summary and two profile bullets).
- `migrations/0008_artifacts.sql` (Postgres):

```sql
CREATE TABLE artifacts (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id         bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    kind           text NOT NULL,          -- cv_docx|cv_pdf|cover_docx|cover_pdf|payload|claims|form_scan|answers
    path           text NOT NULL,          -- relative to the repo, under data/
    sha256         text NOT NULL,
    model          text,                   -- NULL for code-only artifacts
    prompt_version text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    used_at        timestamptz             -- the human used it (uploaded with an application)
);
CREATE INDEX artifacts_job ON artifacts (job_id, kind);

CREATE TABLE llm_calls (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    at          timestamptz NOT NULL DEFAULT now(),
    model       text NOT NULL,
    stage       text NOT NULL,             -- tailor|cover|answers
    job_id      bigint REFERENCES jobs (id) ON DELETE SET NULL,
    ok          boolean NOT NULL,
    error_class text,
    in_chars    integer, out_chars integer, latency_ms integer
);
```

  Add storage helpers `add_artifact(...)`, `artifacts_for(job_id)`, `record_llm_call(...)` in
  `hiredra/storage/db.py` and export them. Update the expected migration list in `tests/test_pg.py`.
- `hiredra/resume/packet.py`: `build_packet(conn, job_id)` writes into
  `data/packets/<job folder>/` (folder name: the job key with every character outside
  `[A-Za-z0-9._-]` replaced by `_`, cut to 80 characters): `cv.docx`, `cv.pdf`, `cover.docx`,
  `cover.pdf`, `payload.json`, `claims.json`. Records artifacts, then `set_status(..., "tailored")`.
  Adds a note when the packet is flagged.
- CLI additions to `python -m hiredra.resume`:
  - `packets --top N` builds packets for the N best `scored` jobs with no `review_reason` (ordered
    by score; weights are placeholders, so the PR text says the order is provisional) that have no
    packet yet. Takes the `llm` lock. Prints one line per job and a summary.
  - `packets --job JOB_KEY` builds one.
  - `--dry-run` prints which jobs would get packets and makes no model call.
- Tests (all with a fake chain, no network): skills section building, the claim gate (each rule, a
  pass case, the Familiar-skill case), the revise-then-fallback flow, the packet folder and
  artifacts, `packets --dry-run`, the migration list.
- **Done when:** `python -m hiredra.resume packets --top 5` on the real database makes 5 packets
  with NVIDIA, each `cv.pdf` is one page, the claim gate passed or the packet is flagged, and the
  five jobs are `tailored`. Look at two of the PDFs before calling it done.

### Piece 3: the sidecar API

**Goal:** a local HTTP service the desktop app talks to. It is the only way the app reaches the
pipeline.

- Dependencies: `uv add fastapi uvicorn`.
- `hiredra/api/__init__.py`, `hiredra/api/__main__.py`, `hiredra/api/app.py`.
- Start: `python -m hiredra.api [--port 0]`. Binds **127.0.0.1 only**. Port 0 = an ephemeral port.
  The bearer token comes from the environment variable `HIREDRA_API_TOKEN` (the app generates one
  per launch). On startup print exactly one JSON line to stdout:
  `{"ready": true, "port": <port>}`. Every request needs `Authorization: Bearer <token>`; anything
  else is 401. No CORS for other origins.
- Endpoints (all JSON):

| Method + path | Purpose |
|---|---|
| `GET /health` | `{"ok": true}` (no token needed, nothing sensitive) |
| `GET /status` | job counts by status, today's quota use per model (`api_usage`), the latest graph audit file name |
| `GET /jobs?view=ranked\|review\|tailored\|queued\|all&limit=&offset=` | paged lists: id, title, company, score, status, review_reason |
| `GET /jobs/{job_key}` | the full job incl. description, skills, `score_features` |
| `POST /jobs/{job_key}/release`, `POST /jobs/{job_key}/reject` | the owner's triage: `hiredra.storage.release_job(conn, job_key)`; reject = `set_status(conn, job_key, "dropped", detail="review: rejected")` |
| `POST /runs/{stage}` | start a stage as a subprocess: `filter`, `extraction`, `sync`, `describe`, `scoring`, `audit`, `packets` (with `{"top": N}`). 409 if a run of that stage is active. Returns a run id |
| `GET /runs/{id}` | state (`running`/`ok`/`failed`), exit code, the last 200 log lines |
| `GET /packets/{job_key}` | artifact list; `GET /packets/{job_key}/files/{name}` streams one file from that packet folder (refuse any name with a path separator) |
| `GET /batches`, `POST /batches` (make the next batch, piece 5), `GET /batches/{id}` | batches |
| `POST /apply/{job_key}/scan`, `GET /apply/{job_key}` | form scan and answers (piece 5) |
| `POST /apply/{job_key}/decision` | `{"decision": "applied"\|"not_yet"\|"skipped"\|"deferred"\|"closed", "confirmation": "..."}`; `applied` requires a non-empty confirmation text from the owner |

- Runs: keep an in-memory table of runs (id, stage, process, log buffer). A stage that is already
  locked elsewhere fails with the stage's own "NOT RUN" exit code 4; report it as `failed` with the
  log. Never start a second run of a stage that is running.
- Tests with FastAPI's `TestClient` against `pg_url`: auth (401 without token), each list view,
  release/reject round trip with a `job_events` row, the file endpoint refusing `../x`, a fake
  stage run.
- **Done when:** `python -m hiredra.api` starts, `/status` answers with the token, and every
  endpoint has a test.

### Piece 4: the Electron desktop app

**Goal:** a desktop window that shows the pipeline, the jobs, the packets and the batches, and that
is easy to extend.

- Location: `app/`. TypeScript `strict`, React, `electron-vite`. Keep dependencies minimal:
  `electron`, `electron-vite`, `vite`, `@vitejs/plugin-react`, `react`, `react-dom`, `typescript`,
  `vitest`, plus their types. Commit `package-lock.json`. Styling: one small CSS file; no UI kit.
- Processes:
  - **main** (`app/src/main/`): creates the window; generates a random token; spawns the sidecar
    (`<repo>/.venv/Scripts/python.exe -m hiredra.api --port 0` on Windows, `python3` elsewhere, cwd
    = repo root, `HIREDRA_API_TOKEN` in its environment); reads the `{"ready": true, "port": N}`
    line; polls `/health`; restarts the sidecar once if it dies; kills it when the app quits. Owns
    all HTTP calls to the sidecar.
  - **preload** (`app/src/preload/`): exposes one typed bridge,
    `window.hiredra.call(method, path, body?) -> Promise<json>`, plus `openPacketFile(jobKey, name)`
    and `openExternal(url)`. No raw `ipcRenderer`.
  - **renderer** (`app/src/renderer/`): the React UI.
- Security (all on from day one): `contextIsolation: true`, `sandbox: true`,
  `nodeIntegration: false`, a strict Content-Security-Policy, deny all permission requests,
  block navigation away from the app and open external links only through `openExternal` (http/https
  only), validate the IPC sender, `openPacketFile` only opens files inside `data/packets/` (resolve
  the path and check the prefix).
- Extensible structure: a single screen registry, `app/src/renderer/screens/index.ts`, exporting a
  list of `{id, title, component}`; the sidebar and router read only this list. Adding a feature =
  one new screen file + one line in the registry. A typed API client module
  (`app/src/renderer/api.ts`) wraps every endpoint.
- Screens:
  - **Pipeline:** one row per stage with a Run button, live state and the log tail (poll
    `/runs/{id}` every 2 s); the status counts and quota use.
  - **Jobs:** tabs Ranked / Review / Tailored / Queued / All; a table (title, company, score,
    status); a detail panel (description, skills with have/adjacent/missing from `score_features`);
    Release and Reject buttons on the Review tab.
  - **Packets:** tailored jobs; buttons to open `cv.docx`, `cv.pdf`, `cover.docx`, `cover.pdf`;
    a "Build packets" button (`POST /runs/packets` with a number).
  - **Batches:** the batches and their 10 jobs with state; a "Make batch" button.
  - **Apply:** for one queued job: the packet links, the copy panel (every prepared field label and
    value from `answers.json`, each with a Copy button), the pause list, an "Open job page" button
    (`openExternal(apply_url)`), and the decision buttons (Applied needs a typed confirmation text).
    The embedded browser and in-app filling are NOT part of this build.
- npm scripts in `app/package.json`: `dev`, `build`, `test`, `typecheck`.
- Tests (vitest): the API client builds the right requests; the screen registry has unique ids; a
  test that reads the `BrowserWindow` options object the main process builds and checks every
  security flag. Keep these tests pure (no Electron runtime in CI).
- CI: add a job to `.github/workflows/ci.yml` that runs `npm ci`, `npm run typecheck` and
  `npm test` in `app/` on Windows.
- **Done when:** `npm run dev` in `app/` opens the window, the sidecar starts by itself, and every
  screen shows real data from the local database.

### Piece 5: the application filler (never submits)

**Goal:** for a tailored job, know what the application form asks, prepare the answers from
verified facts and the packet, group jobs into batches of 10, and fill a form up to its last page
on local fixture forms, stopping before Submit.

- **Facts file:** extend `data/autofill_facts.yml` (shape in `config/autofill_facts.example.yml`,
  update the example) with `preferred_name`, `country`, `city`, `degree`, `institution`,
  `graduation_year`, `current_status`. `hiredra/autofill/facts.py` loads it and **refuses** any key
  or value about work authorization, visa, sponsorship, salary, CTC, compensation, gender, race,
  ethnicity, caste, religion, disability, veteran status, date of birth, age or government ID (match
  key names case-insensitively). Those questions are always left to the human.
- **Form scan** `hiredra/autofill/scan.py` (GET only, through `hiredra.sourcing.http.HttpClient`'s
  `get_json` / `get_text`):
  - Greenhouse: `GET https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{id}?questions=true`
    (no key). Board slug and job id come from the job's source fields.
  - Lever: `GET` the job's apply page and parse the form fields from the HTML.
  - Anything else: `{"via": "unsupported"}`; the job stays usable through the copy panel.
  - 404 or the job missing -> `set_status(..., "dropped", detail="closed")`.
  - Output per field: `label`, `type`, `required`, `class`. Classes are decided by keyword rules in
    code: `FACT` (maps to a facts key), `FILE` (résumé or cover letter upload), `DRAFT` (a free-text
    question the profile can answer), `PAUSE` (everything sensitive: the forbidden topics above,
    legal consent, CAPTCHA, login, notice period, plus **anything unknown**). Unknown = PAUSE.
- **Answers** `hiredra/autofill/answers.py`: `FACT` from the facts file, `FILE` -> the packet's
  `cv.pdf` / `cover.pdf`, `DRAFT` -> one NVIDIA call per form for all its draft questions, checked by
  the claim gate (piece 2), a failing answer becomes PAUSE. Write `form_scan.json` and
  `answers.json` into the packet folder and record them as artifacts.
- **Batches:** `migrations/0009_batches.sql` with `batches` (batch_id text like `B0001`, created_at,
  status `open|in_review|closed`), `batch_items` (batch_id, job_id, position, decision
  `applied|skipped|deferred|closed|NULL`, decided_at), `attention` (job_id, batch_id, category,
  field_label, created_at, resolved_at; labels only, never values). `hiredra/autofill/batch.py`:
  when there are at least 10 `tailored` jobs with a scan, take the next 10 by score into a new batch
  and set them to `queued`. Never put a job in two open batches. CLI: `python -m hiredra.autofill
  scan --job KEY | --tailored`, `python -m hiredra.autofill batch`.
- **Decisions:** `queued -> applied` only through the human's decision with a confirmation text;
  `not_yet` keeps `queued`; `deferred -> tailored`; `skipped` / `closed` -> `dropped` with detail
  `gate: skipped` / `closed`.
- **Fill driver** `hiredra/autofill/driver.py` (the only module in `hiredra/autofill/` that imports
  Playwright; `uv add playwright` then `python -m playwright install chromium`):
  - Opens a page in its own Chromium (for this build), fills fields from `answers.json`: text,
    textarea, select, radio, ordinary checkbox, file upload, "Add another"; clicks
    Next/Continue to move between pages.
  - **`safe_click(target)` refuses**, before any click: a `type=submit` control whose label is not
    a recognized Next/Continue label; any control whose visible label or aria-label matches the
    submit vocabulary (submit, apply, send, finish, complete, confirm, done, review and submit;
    case-insensitive); any click on the last page (no Next/Continue present) other than filling
    fields; consent, legal and PAUSE fields. Every refusal is logged with its reason.
  - **Never** `keyboard.press("Enter")`, never `form.submit()`, never `dispatch_event("submit")`.
  - Ends with a report: filled fields (labels only), skipped, pause items, "last page reached", and
    leaves the browser open on the last page for the human.
  - Runs **only against local fixture forms in this build**: `tests/fixtures/forms/` with at least
    a single-page form and a three-page form (HTML files using the field types above, a final
    Submit button, an "Apply now" button and a "Finish" button). Serve them from a temporary local
    HTTP server in tests.
  - CLI: `python -m hiredra.autofill fill --fixture <name>` (fixtures only).
- Tests: the facts loader refuses each forbidden key; scan classification (unknown -> PAUSE) on
  saved real Greenhouse and Lever responses stored as fixtures; answers with a fake chain; batching
  (ordering, at most 10, no double batching); the decision transitions; the driver fills the
  three-page fixture to its last page, refuses the submit-type button, "Apply now" and "Finish", and
  never presses Enter (assert on a page-side counter of submit events: it must stay 0).
  Mark the Playwright tests so they are skipped when Chromium is not installed (CI need not install
  it).
- **Done when:** on the real database, `scan --tailored` classifies the tailored jobs, `batch`
  makes a batch of up to 10, the app's Apply screen shows a queued job's copy panel and pause list,
  and `fill --fixture three_page` reaches the last page with zero submit events.

### Piece 6: one end-to-end run, and the report for the owner

- `python -m hiredra.e2e [--top N] [--skip STAGE ...]` runs, in order: filter, extraction, graph
  sync, graph describe, scoring, graph audit, `resume packets --top N`, `autofill scan --tailored`,
  `autofill batch`. Each stage is a subprocess; stop at the first failure and say which stage and
  why; print a summary table (stage, exit code, seconds, the stage's last line). Default N = 10.
  No stage ever submits anything.
- Add an "End to end" section to `README.md`: the order above, the app (`cd app && npm install &&
  npm run dev`), and the rule that nothing is ever submitted.
- Write `BUILD_REPORT.md` at the repo root of `build/e2e` (public-safe, no personal data): what each
  piece does, the commands, test counts, what was verified on the real database (counts only), what
  is known to be missing or rough, and the owner's review checklist (section 8).
- **Done when:** `python -m hiredra.e2e --top 5 --skip filter --skip extraction` runs through,
  the app shows the result, and the report exists.

---

## 7. Out of scope for this build (do not start these)

- Weight calibration (owner rating sitting 2, the weight search, the apply line).
- The matching cut-off change ("PR 6": partial credit at 0.615, full credit by name only). It waits
  for the owner's yes.
- The weekly crawl and anything in sourcing.
- The embedded browser inside the app, CDP filling from the app, the network guard, experiments X1
  and X2 (they come with real-site filling, later, with the owner present).
- Ashby, Workday, iCIMS and other form readers; per-platform tuning; every edge case.
- Outreach drafts, contact finding, outcome tracking, the learning loop, packaging the app as an
  installer, the dashboard, graph edges.
- Any change to `main`, `CLAUDE.md`, or the owner's private files other than adding keys to
  `data/autofill_facts.yml`.

---

## 8. The owner's review checklist (the report repeats it)

1. Open `data/packets/<job>/cv.docx` and `cv.pdf` for three jobs: one page, the example's layout,
   no claim that is not in the master profile, important terms bold.
2. Open a cover letter: three paragraphs, nothing invented.
3. Start the app (`cd app && npm run dev`): Pipeline, Jobs, Packets, Batches and Apply screens all
   show real data; Release/Reject works.
4. Look at one queued job's Apply screen: copy panel values, pause list (work authorization, salary
   and the like must be there, never filled).
5. Run `python -m hiredra.autofill fill --fixture three_page` and watch it stop on the last page
   without submitting.
6. Read `BUILD_REPORT.md`'s "known missing" list.
7. Only after this, if satisfied, say so: then `build/e2e` is merged into `main` (section 1).

---

## 9. Progress log (append one line per finished piece)

| Piece | Branch / PR | Merged into build/e2e | Notes |
|---|---|---|---|
| 1 Résumé core | build/p1-resume-core / #177 | yes (210bd65) | Word + PDF, owner's example layout, page fit, no model |
| 2 Tailoring + claim gate + cover + packets | | | |
| 3 Sidecar API | | | |
| 4 Desktop app | | | |
| 5 Application filler | | | |
| 6 End-to-end run + report | | | |
