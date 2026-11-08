# Hiredra

[![CI](https://github.com/brindeshwar/hiredra/actions/workflows/ci.yml/badge.svg)](https://github.com/brindeshwar/hiredra/actions/workflows/ci.yml)

**One system, many agents.** Hiredra takes a job seeker from "no jobs" to a ranked list of jobs worth
applying to, and later to tailored applications waiting for their approval. It sources jobs widely,
filters them without spending a single LLM token, has an LLM read each survivor's skills, and scores the
fit with deterministic code against a bounded skill graph.

> **Nothing auto-submits or auto-sends.** Hiredra prepares; a human reviews and clicks Submit.
> No code path clicks Submit, Send or Apply, or posts anything to a third party.

## The pipeline

```
Source       CODE      16,400 job boards on 28 platforms (Workday, Greenhouse, Lever, Ashby, Oracle,
                       Eightfold, Phenom, SmartRecruiters, ...) plus JobSpy; dedup across sources   [built]
Filter       CODE      location -> role type -> seniority (reads the job text, not the title)      [built]
Extract      LLM       skills, required vs nice-to-have, domain, legitimacy: schema-forced JSON    [built]
Graph        CODE      a skill graph bounded to the owner's skills + skills seen in real jobs      [built]
Score        CODE      domain fit + skill coverage - gap distance, ranked                          [built]
Calibrate    CODE      thresholds by an F1 sweep (done); weights from blind ratings          [in progress]
Tailor       AGENT     CV + cover letter that may only claim what the profile states             [planned]
Autofill     AGENT     fills forms from verified facts, pauses, never clicks Submit              [planned]
Human gate             batches of 10, reviewed and submitted by the human                        [planned]
Track        CODE      outcomes, per-channel yield, a learning loop the human approves           [planned]
```

## Numbers (crawl of 7-8 November 2026)

- **1,079,946** postings listed on **13,820** boards in one crawl; **143,459** new.
- Of the **121,909** new postings that aren't copies of another, **23,354** pass the location filter,
  **838** the role-type filter and **137** the seniority gate. Those reach the LLM, so it reads about
  0.1% of what is fetched.
- **2,712** skills in the graph, each embedded locally (Qwen3-Embedding-0.6B on the CPU).
- Skill-match thresholds calibrated on **174** hand-labelled skill pairs (87 tune, 87 holdout).
- **2,290** tests; CI runs them twice (without and with Postgres), plus a full-history secret scan.

## The six hard rules

1. **Nothing auto-submits or auto-sends.** The system prepares; the human commits.
2. **Free runtime only.** Free-tier LLMs (Gemini, an NVIDIA-hosted fallback), public ATS APIs, a local
   embedding model. No paid APIs, data tools or people-finders.
3. **The LLM reads, code decides.** The LLM extracts and normalises text. Every number, score, threshold
   and ranking is computed by deterministic code.
4. **Every claim must be interview-defensible.** Nothing is invented; the tailoring agents may only assert
   facts from the owner's own profile.
5. **No secrets in code or git.** Keys live in `.env`; no plaintext password anywhere.
6. **A human gate before every irreversible or external action.**

## How it reads the web

- Only public job-board APIs and pages, as an identified client
  (`Hiredra/0.1 (+https://github.com/brindeshwar)`). robots.txt is respected; sites whose terms forbid
  scraping are left out; no bot-bypass, proxies or browser fingerprinting.
- The crawl runs on a **conveyor belt**: every board is a task, or many 50-request chunks for a big one,
  laid out before the run. A free worker takes the first task the rules allow. Each platform that limits
  across its tenants (Workday, Eightfold, Phenom) is held to 60% of the rate it was measured to refuse at,
  with at most 2 requests at once on any host. A refusal puts the task back on the belt for later in the
  run, and one writer thread does every database write. See [ARCHITECTURE.md](architecture/ARCHITECTURE.md).

## Quickstart (the built stages)

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                                   # add --group embed for the local embedding model
cp .env.example .env                      # set HIREDRA_PG_PASSWORD, GEMINI_API_KEY (free tier)
docker compose up -d                      # Postgres 17 + pgvector, on localhost only
python -m hiredra.sourcing boards import config/companies.yml
python -m hiredra.sourcing crawl          # the boards that are due
python -m hiredra.filter                  # location, role type, seniority, liveness
python -m hiredra.extraction              # skills per job (paced to the free tier)
python -m hiredra.graph sync              # new skills into the graph
python -m hiredra.scoring run
python -m hiredra.scoring list            # the ranked list
```

Run the tests with `uv run pytest -q`. Set `HIREDRA_TEST_DATABASE_URL` to run the database tests too.

## Credits

Hiredra reuses ideas and code from open-source projects, each credited with its licence in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): career-ops, OpenJobs, freehire, open-jobs,
ats-scrapers, GeoNames and Qwen3-Embedding among them.

## License

MIT, see [LICENSE](LICENSE).
