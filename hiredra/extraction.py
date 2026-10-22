"""Sub-slice 3: skill-extraction agent (build step 3, docs/scoring.md sec 1
Stages 3+4).

One strict Gemini call (``gemini-3.5-flash-lite``, minimal thinking) per
surviving job, via LangChain's ``ChatGoogleGenerativeAI.with_structured_output``
run through a single-node LangGraph graph. This is the only place a model
gets constructed or invoked in this module -- no ``google-genai``/
``google-generativeai`` SDK call and no direct HTTP anywhere else
(CLAUDE.md sec 7). The LLM only reads and normalizes: it extracts+tags
skills, buckets domain, and flags legitimacy/confidence. It never scores --
that's ``hiredra.filter``'s gates (2b) and the not-yet-built Stage 5.

``run_extraction`` reads ``status='filtered'`` and advances survivors to
``status='extracted'`` -- a new intermediate status between the filter
funnel and the not-yet-built scoring stage, so the funnel's status vocabulary
stays honest about what's actually happened to a job. A too-thin JD body
(post-preprocessing) skips the call entirely and still lands at
``'extracted'`` with ``extraction_confidence='low'`` -- "extracted" means
"Step 3 has run for this job," not "Step 3 found something." Resume is
free: a job only leaves ``'filtered'`` once its columns are written and its
status is advanced, so a rerun's status filter alone skips everything
already done (same idempotency as ``run_funnel``/``check_liveness`` --
*not* the orchestrator's separate ``runs``/``--resume`` machinery, which
exists for per-company network-fetch interruption, a different stage).

Hardened in M0.1: a daily request counter (stop before Google's cap), a run lock
shared by every flash-lite spender, labelled stops (the failure table on
``run_extraction``), and an ``extraction_version`` stamp on every job it writes.

    python -m hiredra.extraction [--limit N] [--stale] [--db PATH] [--config PATH]
                                 [--usage-path PATH] [--lock-path PATH]

Exit codes (same as the graph CLI): 0 ok, 2 stopped with a reason, 4 lock held, 1 config error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from hiredra import vocab
from hiredra.lock import LockHeld, RunLock
from hiredra.quota import DailyQuotaExceeded, DailyUsage, _retry_after_seconds, classify_failure, reset_text
from hiredra.storage import (
    DEFAULT_DB_PATH,
    add_note,
    connect,
    init_db,
    iter_jobs,
    set_extraction,
    set_status,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "extraction.yml"
DEFAULT_FOOTERS_PATH = _REPO_ROOT / "config" / "boilerplate_footers.json"

MODEL_NAME = "gemini-3.5-flash-lite"

#: One counter per metered model: every flash-lite agent shares it (Google meters the model).
DEFAULT_USAGE_PATH = _REPO_ROOT / "data" / "usage" / f"{MODEL_NAME}.json"
#: Held by every process that spends flash-lite quota (extraction now, tailoring later).
DEFAULT_LOCK_PATH = _REPO_ROOT / "data" / "llm.lock"

#: This many schema errors in a row means a systematic problem, not one odd JD: stop.
SCHEMA_ERROR_STREAK = 3


@dataclass(frozen=True)
class ExtractionConfig:
    min_description_chars: int
    requests_per_minute: int
    boilerplate_footers: dict = field(default_factory=dict)
    requests_per_day: int = 500
    headroom: float = 0.90

    @property
    def daily_budget(self) -> int:
        """Requests we allow ourselves per day: Google's cap times the headroom (500 x 0.90 = 450)."""
        return int(self.requests_per_day * self.headroom)


def load_extraction_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    footers_path: Path | str = DEFAULT_FOOTERS_PATH,
) -> ExtractionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    footers_file = Path(footers_path)
    footers = json.loads(footers_file.read_text(encoding="utf-8")) if footers_file.exists() else {}
    return ExtractionConfig(
        min_description_chars=int(data.get("min_description_chars", 200)),
        requests_per_minute=int(data.get("requests_per_minute", 15)),
        boilerplate_footers=footers,
        requests_per_day=int(data.get("requests_per_day", 500)),
        headroom=float(data.get("headroom", 0.90)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# preprocessing (characterization pass sec 5, applied in leverage order)
# ─────────────────────────────────────────────────────────────────────────────
_ESCAPED_PUNCT_RE = re.compile(r"\\([-.()#*_\[\]!])")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_ABOUT_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _strip_boilerplate_footer(text: str, company: str, footers: dict) -> tuple[str, bool]:
    """Exact-literal strip of a known company's repeated footer block.

    Only strips on a byte-exact suffix match (e.g. OpenAI's ~2.8k "About
    OpenAI..." tail, identical across every sampled posting) -- a company
    whose boilerplate has drifted even slightly is left untouched rather
    than guessed at.
    """
    footer = footers.get(company)
    if footer and text.endswith(footer):
        return text[: -len(footer)].rstrip(), True
    return text, False


def _strip_leading_about_paragraph(text: str, company: str) -> tuple[str, bool]:
    """Conservative strip of a leading "About <Company>" mission block.

    Only touches the very start of the JD, and only when it opens with
    "About <company>" (case-insensitive) -- never a paragraph found deeper
    in the text. Two shapes, both observed in the characterization sample:
    "About X" as its own bare header paragraph, immediately followed by the
    actual mission paragraph (Stripe/OpenAI's shape -- both paragraphs are
    consumed); or "About X ..." with real content in that same first
    paragraph (only that one paragraph is consumed). If the JD has no blank
    line at all, it's left untouched rather than guessing where it ends
    (mission copy varies across postings, so this is a shape match, not an
    exact-literal one).
    """
    leading = text.lstrip()
    header_re = re.compile(r"^about\s+" + re.escape(company) + r"\s*$", re.IGNORECASE)
    bare_start_re = re.compile(r"^about\s+" + re.escape(company), re.IGNORECASE)

    paragraphs = _ABOUT_PARA_SPLIT_RE.split(leading)
    if len(paragraphs) < 2 or not bare_start_re.match(paragraphs[0]):
        return text, False  # no blank line to stop at -- don't guess

    consume = 2 if header_re.match(paragraphs[0]) else 1
    remainder = "\n\n".join(paragraphs[consume:])
    return remainder.lstrip(), True


def _unescape_markdown(text: str) -> str:
    """Undo markdown-style backslash-escaping of punctuation.

    100% of the characterization sample's jobspy-indeed/linkedin postings
    had this (``\\-``, ``\\.``, ...); 0% of ATS-sourced postings did --
    a safe no-op there, so this applies to every job unconditionally.
    """
    return _ESCAPED_PUNCT_RE.sub(r"\1", text)


def _collapse_blank_lines(text: str) -> str:
    """Runs of 3+ newlines (the Ashby "one bullet per several blank lines"
    pattern) collapse to a single blank line."""
    return _BLANK_RUN_RE.sub("\n\n", text)


def preprocess(text: str, company: str, config: ExtractionConfig) -> tuple[str, list[str]]:
    """Clean ``text`` before the extraction call. Returns ``(clean_text, strip_log)``.

    ``strip_log`` is a list of human-readable strings for the caller to
    record as ``job_events`` notes -- every strip is logged, and since each
    step only acts on an exact or narrowly-shaped match, every strip is
    reversible by re-running against the original stored ``description``.
    """
    log: list[str] = []
    text, did = _strip_boilerplate_footer(text, company, config.boilerplate_footers)
    if did:
        log.append(f"extraction: stripped exact-literal boilerplate footer for {company!r}")
    text, did = _strip_leading_about_paragraph(text, company)
    if did:
        log.append(f"extraction: stripped leading 'About {company}' paragraph")
    text = _unescape_markdown(text)
    text = _collapse_blank_lines(text)
    return text.strip(), log


# ─────────────────────────────────────────────────────────────────────────────
# schema (docs/scoring.md sec 1 Stage 4; the approved schema proposal)
# ─────────────────────────────────────────────────────────────────────────────
class SkillItem(BaseModel):
    name: str
    requirement: Literal["required", "good_to_have"]


class ExtractionResult(BaseModel):
    """``required``/emission order matters (Gemini structured-output subset
    rule): evidence (``skills``) is emitted before the derived judgments that
    need the whole skill picture."""

    skills: list[SkillItem]
    domain_bucket: Literal["agent_llm", "backend", "data", "ml", "generic_swe", "other"]
    legitimacy: Literal["legitimate", "suspect", "likely_ghost"]
    extraction_confidence: Literal["high", "low"]


#: v2 (M0.5 pilot spot-check: only 6 of 20 skill lists were clean). The rules target what the
#: spot-check found: descriptions instead of skill names, combined items ("aws/azure/gcp"),
#: 75-item lists, product names, and "high" confidence on a 0-skill or 2-sentence JD.
_PROMPT_PREFIX = (
    "Read the job description below. Extract its skills, classify the role, and flag the posting. "
    "Only read and normalize what is written -- do not rate, score, or rank anything.\n\n"
    "SKILLS\n"
    "- A skill is a technology, tool, language, framework, platform, or named technical method "
    '(for example "python", "kubernetes", "postgresql", "rag", "rest api design", "data structures").\n'
    '- Use the short canonical name, 1-4 words, lowercase: "postgresql", not "experience with '
    'relational databases such as PostgreSQL".\n'
    '- One skill per item. Split lists: "AWS/Azure/GCP" becomes "aws", "azure", "gcp".\n'
    "- Leave out soft skills and traits (communication, ownership, collaboration), generic activities "
    "(software development, building scalable systems, debugging), degrees, years of experience, and "
    "bare umbrella words on their own (ai, ml, cloud, software, data).\n"
    '- Never list the hiring company itself or its own products as a skill (at Cursor, "cursor" is not '
    "a skill).\n"
    "- At most 25 skills. If the description names more, keep the ones most central to the role.\n"
    '- requirement: "required" when the description says must, required or you have; "good_to_have" '
    'for preferred, nice to have, a plus or bonus. If it doesn\'t say, use "required" for skills '
    'central to the role and "good_to_have" for the rest.\n\n'
    "DOMAIN BUCKET (exactly one)\n"
    "- agent_llm: building with LLMs, agents, RAG or generative AI.\n"
    "- ml: training, evaluating or serving machine-learning models, not centred on LLM applications.\n"
    "- data: data engineering, analytics or data science.\n"
    "- backend: server-side services, APIs, distributed systems, databases or infrastructure.\n"
    "- generic_swe: general, full-stack, frontend, mobile or test-automation software engineering.\n"
    "- other: not a software engineering role.\n\n"
    "LEGITIMACY\n"
    "- legitimate: a real, specific posting. suspect: vague, inconsistent or misleading. "
    "likely_ghost: generic boilerplate with no concrete role behind it.\n\n"
    "EXTRACTION CONFIDENCE\n"
    '- "low" when the description is too short or too vague to list reliable skills (for example '
    'fewer than 3 concrete skills), otherwise "high".\n\n'
    "JOB DESCRIPTION:\n"
)


def _build_prompt(jd_text: str) -> str:
    return _PROMPT_PREFIX + jd_text


def extraction_version() -> str:
    """``<model>:<sha8>``: sha8 is sha256 over the prompt text + the output schema. Any change
    to either changes the stamp, so ``--stale`` can list the jobs an older version wrote."""
    blob = _PROMPT_PREFIX + json.dumps(ExtractionResult.model_json_schema(), sort_keys=True)
    return f"{MODEL_NAME}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# the LangGraph node
# ─────────────────────────────────────────────────────────────────────────────
class _ExtractionState(TypedDict):
    jd_text: str
    result: ExtractionResult


class SchemaError(Exception):
    """Gemini answered, but not with a usable ``ExtractionResult`` (e.g. ``None`` when the
    answer was safety-blocked or empty)."""


#: Exceptions that mean "the model answered, the answer didn't fit the schema".
_SCHEMA_ERRORS = (SchemaError, OutputParserException, ValidationError)


def _call(chain, jd_text: str, usage: DailyUsage | None) -> ExtractionResult:
    """One attempt: check the daily budget, count the attempt, call."""
    if usage is not None:
        usage.reserve(1)  # raises BEFORE the call when today's budget is used
        # Counted BEFORE the call, failed attempts too: Google counts some requests it rejects or
        # times out on, so counting only successes let the local count fall behind (694 vs 644
        # observed on 2026-10-19). Ahead of Google is the safe direction.
        usage.record(1)
    result = chain.invoke(_build_prompt(jd_text))
    if result is None:
        raise SchemaError("structured output came back empty")
    return result


def build_extraction_chain(*, api_key: str | None = None):
    """The LangChain structured-output chain: one Gemini call in, one
    ``ExtractionResult`` out. The only place a model is constructed."""
    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (check .env) -- required to build the extraction chain.")
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=key,
        reasoning_effort="minimal",
    )
    return model.with_structured_output(ExtractionResult)


def build_extraction_graph(chain, usage: DailyUsage | None = None, sleep=time.sleep):
    """A single-node LangGraph graph: START -> extract -> END.

    Deliberately this minimal -- one call, no branching -- since that's all
    Step 3 needs today. This is still the right seam for something like a
    future retry-with-repaired-prompt node without a rewrite.

    Retries (4 attempts) cover only ``transient`` failures (429 per-minute, 5xx, network),
    waiting the server's own retryDelay when it gives one. Every attempt spends quota, so
    each one goes through the daily counter.
    """
    backoff = wait_exponential(multiplier=1, min=2, max=30)

    def _wait(retry_state) -> float:
        hint = _retry_after_seconds(retry_state.outcome.exception())
        return backoff(retry_state) if hint is None else min(hint + 1.0, 120.0)

    retrying = Retrying(
        retry=retry_if_exception(lambda exc: classify_failure(exc) == "transient"),
        stop=stop_after_attempt(4),
        wait=_wait,
        sleep=sleep,
        reraise=True,
    )

    def _extract(state: _ExtractionState) -> dict:
        return {"result": retrying(_call, chain, state["jd_text"], usage)}

    graph = StateGraph(_ExtractionState)
    graph.add_node("extract", _extract)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", END)
    return graph.compile()


class _Pacer:
    """A plain minimum gap between calls, sized to the RPM ceiling -- not a
    token bucket, not a learned rate. Send-gap plus the retry above is the
    whole pacing story."""

    def __init__(self, requests_per_minute: int, sleep=time.sleep):
        self._min_gap = 60.0 / requests_per_minute
        self._last: float | None = None
        self._sleep = sleep

    def wait(self) -> None:
        if self._last is not None:
            elapsed = time.monotonic() - self._last
            if elapsed < self._min_gap:
                self._sleep(self._min_gap - elapsed)
        self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# orchestration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExtractionStats:
    seen: int = 0
    skipped_thin: int = 0
    extracted: int = 0
    schema_failed: int = 0
    stopped: str | None = None  # daily_quota | transient | api_error | bad_response | schema_error
    stop_detail: str = ""


def run_extraction(
    conn,
    config: ExtractionConfig,
    chain=None,
    *,
    usage: DailyUsage | None = None,
    limit: int | None = None,
    sleep=time.sleep,
) -> ExtractionStats:
    """Read status='filtered', run Step 3, advance survivors to 'extracted'.

    ``chain`` is injectable (a fake in tests; ``build_extraction_chain()``'s
    result in real use) so this function never needs a live API key itself.
    ``usage`` is the daily counter (None = no budget check, for tests).

    Failure handling (plan unit M0.1):

    ============  ========================================  ================================
    class         example                                   action
    ============  ========================================  ================================
    daily_quota   per-day 429, or our budget is spent       stop the run
    transient     429 per-minute / 5xx / network, 4 tries   stop the run
    api_error     400 / 401 / 403 / 404                     stop the run
    schema_error  None, unparsable, failed validation       note it, job stays 'filtered',
                                                            go on; 3 in a row -> stop
    unknown       anything else                             re-raise (a bug must be loud)
    ============  ========================================  ================================

    Jobs written before a stop stay written; a re-run continues from the status filter.
    """
    stats = ExtractionStats()
    if usage is not None and usage.remaining() == 0:  # refuse with NO API call
        stats.stopped, stats.stop_detail = "daily_quota", usage.summary()
        return stats
    if chain is None:
        chain = build_extraction_chain()
    graph = build_extraction_graph(chain, usage, sleep=sleep)
    pacer = _Pacer(config.requests_per_minute, sleep=sleep)
    stamp = extraction_version()
    streak = 0

    jobs = list(iter_jobs(conn, status="filtered"))
    for job in jobs if limit is None else jobs[:limit]:
        stats.seen += 1
        job_id = job["job_id"]

        clean_text, strip_log = preprocess(job["description"] or "", job["company"], config)
        for note in strip_log:
            add_note(conn, job_id, note)

        if len(clean_text) < config.min_description_chars:
            add_note(conn, job_id, "extraction: skipped-thin")
            set_extraction(
                conn,
                job_id,
                skills="[]",
                domain_bucket=None,
                legitimacy=None,
                extraction_confidence="low",
                extraction_version=stamp,
            )
            set_status(conn, job_id, "extracted")
            stats.skipped_thin += 1
            continue

        pacer.wait()
        try:
            result: ExtractionResult = graph.invoke({"jd_text": clean_text})["result"]
        except _SCHEMA_ERRORS as err:
            first_line = str(err).splitlines()[0][:120] if str(err) else ""
            short = f"{type(err).__name__}: {first_line}"
            add_note(conn, job_id, f"extraction: schema error ({short})")
            stats.schema_failed += 1
            streak += 1
            if streak >= SCHEMA_ERROR_STREAK:
                stats.stopped = "schema_error"
                stats.stop_detail = f"{streak} schema errors in a row (last: {short})"
                break
            continue
        except Exception as err:
            kind = classify_failure(err)
            if kind is None:
                raise  # unknown: most likely a bug -- never turn it into a status line
            if kind == "daily_quota" and usage is not None and not isinstance(err, DailyQuotaExceeded):
                usage.mark_exhausted("reported by Google (server-side daily quota error)")
            stats.stopped = kind
            stats.stop_detail = str(err).splitlines()[0][:300] if str(err) else type(err).__name__
            break
        streak = 0

        skills = [
            {"name": vocab.resolve_skill(conn, item.name), "requirement": item.requirement} for item in result.skills
        ]
        set_extraction(
            conn,
            job_id,
            skills=json.dumps(skills),
            domain_bucket=result.domain_bucket,
            legitimacy=result.legitimacy,
            extraction_confidence=result.extraction_confidence,
            extraction_version=stamp,
        )
        set_status(conn, job_id, "extracted")
        stats.extracted += 1

    return stats


def stale_jobs(conn) -> dict[str | None, list[dict]]:
    """Extracted/scored jobs whose stamp is not the current one, grouped by their old stamp
    (None = written before stamps existed). Lists only: re-extracting costs 1 request per job,
    so it is always the owner's call."""
    groups: dict[str | None, list[dict]] = {}
    rows = conn.execute(
        "SELECT job_id, company, title, extraction_version FROM jobs "
        "WHERE status IN ('extracted', 'scored') AND (extraction_version IS NULL OR extraction_version != ?) "
        "ORDER BY extraction_version, job_id",
        (extraction_version(),),
    )
    for row in rows:
        groups.setdefault(row["extraction_version"], []).append(dict(row))
    return groups


_STOP_HINTS = {
    "daily_quota": "wait for the daily reset",
    "transient": "re-run in a few minutes",
    "api_error": "check the key, the model name and the request",
    "bad_response": "check the model's answer",
    "schema_error": "read the 'schema error' notes, fix the prompt or schema, re-run",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.extraction")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="process at most N filtered jobs")
    parser.add_argument("--stale", action="store_true", help="list jobs extracted with an older stamp (no API call)")
    parser.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    args = parser.parse_args(argv)

    try:
        config = load_extraction_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as err:
        print(f"config error: {err}", file=sys.stderr)
        return 1

    if args.stale:
        conn = connect(args.db)
        try:
            init_db(conn)
            groups = stale_jobs(conn)
        finally:
            conn.close()
        print(f"current stamp: {extraction_version()}")
        for stamp, jobs in groups.items():
            print(f"  {stamp or '(no stamp)'}: {len(jobs)} job(s)")
            for j in jobs:
                print(f"    {j['job_id']}  {j['company']} -- {j['title']}")
        print(f"stale: {sum(len(j) for j in groups.values())}")
        return 0

    usage = DailyUsage(config.daily_budget, args.usage_path)
    try:
        with RunLock(args.lock_path, what="extraction", kind="flash-lite"):
            if usage.remaining() == 0:
                print(f"NOT RUN: {usage.summary()}")
                return 2
            try:
                chain = build_extraction_chain()
            except RuntimeError as err:
                print(f"config error: {err}", file=sys.stderr)
                return 1
            conn = connect(args.db)
            try:
                init_db(conn)
                stats = run_extraction(conn, config, chain, usage=usage, limit=args.limit)
            finally:
                conn.close()
    except LockHeld as err:
        print(f"NOT RUN: {err}")
        return 4

    if stats.stopped:
        hint = _STOP_HINTS.get(stats.stopped, "")
        reset = f"; resets {reset_text(datetime.now(timezone.utc))}" if stats.stopped == "daily_quota" else ""
        print(f"STOPPED ({stats.stopped}): {stats.stop_detail} -- {hint}{reset}")
    print(
        f"seen={stats.seen} extracted={stats.extracted} skipped_thin={stats.skipped_thin} "
        f"schema_failed={stats.schema_failed} stopped={stats.stopped or 'none'} "
        f"used_today={usage.used()}/{usage.budget}"
    )
    return 2 if stats.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
