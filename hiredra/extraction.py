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

    python -m hiredra.extraction [--db data/hiredra.db] [--config config/extraction.yml]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

import yaml
from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from hiredra import vocab
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


@dataclass(frozen=True)
class ExtractionConfig:
    min_description_chars: int
    requests_per_minute: int
    boilerplate_footers: dict = field(default_factory=dict)


def load_extraction_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    footers_path: Path | str = DEFAULT_FOOTERS_PATH,
) -> ExtractionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    footers_file = Path(footers_path)
    footers = (
        json.loads(footers_file.read_text(encoding="utf-8")) if footers_file.exists() else {}
    )
    return ExtractionConfig(
        min_description_chars=int(data.get("min_description_chars", 200)),
        requests_per_minute=int(data.get("requests_per_minute", 15)),
        boilerplate_footers=footers,
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


_PROMPT_PREFIX = (
    "Read the job description below. Extract every required and "
    "good-to-have skill mentioned, tag each one accordingly, classify the "
    "role into exactly one domain bucket, and flag the posting's legitimacy "
    "and your own confidence in this extraction. Only read and normalize "
    "what is written -- do not rate, score, or rank anything.\n\n"
    "JOB DESCRIPTION:\n"
)


def _build_prompt(jd_text: str) -> str:
    return _PROMPT_PREFIX + jd_text


# ─────────────────────────────────────────────────────────────────────────────
# the LangGraph node
# ─────────────────────────────────────────────────────────────────────────────
class _ExtractionState(TypedDict):
    jd_text: str
    result: ExtractionResult


def _is_transient_api_error(exc: BaseException) -> bool:
    """429 (rate limit) or 5xx -- worth a retry. Anything else (a bad
    request, a schema mismatch) is not, and is left to raise.

    Importing ``google.genai.errors.APIError`` here is inspecting the
    exception type the underlying SDK raises through the LangChain wrapper,
    not calling the SDK -- every actual model invocation in this module
    still goes through ``chain.invoke`` / the LangGraph node exclusively.
    """
    return isinstance(exc, APIError) and (exc.code == 429 or exc.code >= 500)


@retry(
    retry=retry_if_exception(_is_transient_api_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _invoke(chain, jd_text: str) -> ExtractionResult:
    return chain.invoke(_build_prompt(jd_text))


def build_extraction_chain(*, api_key: str | None = None):
    """The LangChain structured-output chain: one Gemini call in, one
    ``ExtractionResult`` out. The only place a model is constructed."""
    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set (check .env) -- required to build the "
            "extraction chain."
        )
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=key,
        reasoning_effort="minimal",
    )
    return model.with_structured_output(ExtractionResult)


def build_extraction_graph(chain):
    """A single-node LangGraph graph: START -> extract -> END.

    Deliberately this minimal -- one call, no branching -- since that's all
    Step 3 needs today. This is still the right seam for something like a
    future retry-with-repaired-prompt node without a rewrite.
    """

    def _extract(state: _ExtractionState) -> dict:
        result = _invoke(chain, state["jd_text"])
        return {"result": result}

    graph = StateGraph(_ExtractionState)
    graph.add_node("extract", _extract)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", END)
    return graph.compile()


class _Pacer:
    """A plain minimum gap between calls, sized to the RPM ceiling -- not a
    token bucket, not a learned rate. Send-gap plus the retry above is the
    whole pacing story."""

    def __init__(self, requests_per_minute: int):
        self._min_gap = 60.0 / requests_per_minute
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            elapsed = time.monotonic() - self._last
            if elapsed < self._min_gap:
                time.sleep(self._min_gap - elapsed)
        self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# orchestration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExtractionStats:
    seen: int = 0
    skipped_thin: int = 0
    extracted: int = 0
    failed: int = 0


def run_extraction(conn, config: ExtractionConfig, chain=None) -> ExtractionStats:
    """Read status='filtered', run Step 3, advance survivors to 'extracted'.

    ``chain`` is injectable (a fake in tests; ``build_extraction_chain()``'s
    result in real use) so this function never needs a live API key itself.
    A job whose call fails (after retries) is left at 'filtered' -- a rerun
    retries it; extraction never drops a job.
    """
    stats = ExtractionStats()
    if chain is None:
        chain = build_extraction_chain()
    graph = build_extraction_graph(chain)
    pacer = _Pacer(config.requests_per_minute)

    jobs = list(iter_jobs(conn, status="filtered"))
    for job in jobs:
        stats.seen += 1
        job_id = job["job_id"]

        clean_text, strip_log = preprocess(job["description"] or "", job["company"], config)
        for note in strip_log:
            add_note(conn, job_id, note)

        if len(clean_text) < config.min_description_chars:
            add_note(conn, job_id, "extraction: skipped-thin")
            set_extraction(
                conn, job_id, skills="[]", domain_bucket=None, legitimacy=None,
                extraction_confidence="low",
            )
            set_status(conn, job_id, "extracted")
            stats.skipped_thin += 1
            continue

        pacer.wait()
        try:
            result: ExtractionResult = graph.invoke({"jd_text": clean_text})["result"]
        except Exception as err:  # noqa: BLE001 -- any call/parse failure: leave for a retry, never drop
            add_note(conn, job_id, f"extraction: call failed ({err})")
            stats.failed += 1
            continue

        skills = [
            {"name": vocab.resolve_skill(conn, item.name), "requirement": item.requirement}
            for item in result.skills
        ]
        set_extraction(
            conn, job_id,
            skills=json.dumps(skills),
            domain_bucket=result.domain_bucket,
            legitimacy=result.legitimacy,
            extraction_confidence=result.extraction_confidence,
        )
        set_status(conn, job_id, "extracted")
        stats.extracted += 1

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.extraction")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    config = load_extraction_config(args.config)
    conn = connect(args.db)
    try:
        init_db(conn)
        stats = run_extraction(conn, config)
    finally:
        conn.close()

    print(
        f"seen={stats.seen} extracted={stats.extracted} "
        f"skipped_thin={stats.skipped_thin} failed={stats.failed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
