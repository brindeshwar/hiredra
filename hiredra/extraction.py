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

    python -m hiredra.extraction [--limit N] [--stale] [--dictionary] [--db PATH] [--config PATH]
                                 [--usage-path PATH] [--lock-path PATH]

Exit codes (same as the graph CLI): 0 ok, 2 stopped with a reason, 4 lock held, 1 config error.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from tenacity import Retrying, retry_if_exception, stop_after_attempt

from hiredra import skilltag, vocab
from hiredra.lock import LockHeld, RunLock
from hiredra.nvidia import NvidiaChain
from hiredra.quota import DailyQuotaExceeded, DailyUsage, classify_failure, jittered_wait, reset_text
from hiredra.storage import (
    add_note,
    connect,
    init_db,
    iter_jobs,
    select_jobs,
    set_dictionary_skills,
    set_extraction,
    set_skills,
    set_status,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "extraction.yml"
DEFAULT_FOOTERS_PATH = _REPO_ROOT / "config" / "boilerplate_footers.json"

MODEL_NAME = "gemini-3.5-flash-lite"

#: Held by every process that spends flash-lite quota (extraction now, tailoring later). The daily counter is one
#: ``api_usage`` row per metered model, shared by every flash-lite agent (Google meters the model).
LOCK_NAME = "llm"

#: This many schema errors in a row means a systematic problem, not one odd JD: stop.
SCHEMA_ERROR_STREAK = 3


@dataclass(frozen=True)
class ExtractionConfig:
    min_description_chars: int
    requests_per_minute: int
    boilerplate_footers: dict = field(default_factory=dict)
    requests_per_day: int = 500
    headroom: float = 0.90
    # Guards (code decides what the model is not trusted to enforce -- see apply_guards).
    max_skills: int = 25
    max_skill_words: int = 4
    generic_skills: frozenset = frozenset()
    skill_synonyms: dict = field(default_factory=dict)
    # The dictionary producer's own stoplist (on top of generic_skills): umbrella words and office tools.
    dictionary_skip: frozenset = frozenset()
    # Fallback provider (docs/model-selection-extraction.md): used when Gemini's daily budget is spent.
    fallback_provider: str | None = None  # "nvidia" or None
    fallback_model: str = ""
    fallback_base_url: str = "https://integrate.api.nvidia.com/v1"
    fallback_requests_per_minute: int = 40
    fallback_headroom: float = 0.90
    fallback_requests_per_day: int = 2000
    fallback_timeout_seconds: float = 120.0
    fallback_max_tokens: int = 4096

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
        max_skills=int(data.get("max_skills", 25)),
        max_skill_words=int(data.get("max_skill_words", 4)),
        generic_skills=frozenset(_norm(x) for x in data.get("generic_skills") or []),
        skill_synonyms={_norm(k): tuple(_norm(v) for v in vs) for k, vs in (data.get("skill_synonyms") or {}).items()},
        dictionary_skip=frozenset(_norm(x) for x in data.get("dictionary_skip") or []),
        **_fallback_fields(data.get("fallback") or {}),
    )


def _fallback_fields(fb: dict) -> dict:
    provider = fb.get("provider")
    if provider not in (None, "nvidia"):
        raise ValueError(f"extraction config: fallback.provider must be nvidia or null, got {provider!r}")
    if provider and not fb.get("model"):
        raise ValueError("extraction config: fallback.model is required when a fallback provider is set")
    return {
        "fallback_provider": provider,
        "fallback_model": str(fb.get("model") or ""),
        "fallback_base_url": str(fb.get("base_url") or "https://integrate.api.nvidia.com/v1"),
        "fallback_requests_per_minute": int(fb.get("requests_per_minute", 40)),
        "fallback_headroom": float(fb.get("headroom", 0.90)),
        "fallback_requests_per_day": int(fb.get("requests_per_day", 2000)),
        "fallback_timeout_seconds": float(fb.get("timeout_seconds", 120)),
        "fallback_max_tokens": int(fb.get("max_tokens", 4096)),
    }


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


def _guard_settings(config: ExtractionConfig | None) -> dict:
    if config is None:
        return {}
    return {
        "max_skills": config.max_skills,
        "max_skill_words": config.max_skill_words,
        "generic_skills": sorted(config.generic_skills),
        "skill_synonyms": {k: list(v) for k, v in sorted(config.skill_synonyms.items())},
    }


def extraction_version(config: ExtractionConfig | None = None, model: str = MODEL_NAME) -> str:
    """``<model>:<sha8>``: sha8 is sha256 over the prompt text + the output schema + the guard
    settings. Any change to them changes the stamp, so ``--stale`` can list the jobs an older
    version wrote. ``config=None`` gives the unguarded stamp (prompt + schema only). ``model`` is
    the model that actually did the extraction (Gemini, or the fallback)."""
    blob = _PROMPT_PREFIX + json.dumps(ExtractionResult.model_json_schema(), sort_keys=True)
    guards = _guard_settings(config)
    if guards:
        blob += json.dumps(guards, sort_keys=True)
    return f"{model}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:8]}"


def dictionary_version(config: ExtractionConfig) -> str:
    """``skilltag:<sha8>`` over the dictionary file and the two stoplists ``dictionary_extras`` applies.
    Separate from the extraction stamp: re-deriving dictionary skills is free, so it never makes a job stale
    for the LLM."""
    text = skilltag.SKILL_DICT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")  # same stamp after CRLF checkout
    blob = text.encode("utf-8") + json.dumps([sorted(config.generic_skills), sorted(config.dictionary_skip)]).encode(
        "utf-8"
    )
    return f"skilltag:{hashlib.sha256(blob).hexdigest()[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# guards: the model is trusted for reading, never for constraints (rule 3)
# ─────────────────────────────────────────────────────────────────────────────
_NORM_RE = re.compile(r"[^a-z0-9+#.]+")


def _norm(text: str) -> str:
    """Lowercase, punctuation to spaces (keeping + # . for c++, c#, node.js), single spaces."""
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


def is_grounded(name: str, jd_norm: str, synonyms: dict) -> bool:
    """Does the JD actually name this skill (or a configured synonym of it)?

    ``jd_norm`` is ``" " + _norm(jd) + " "``. A form of 1-2 characters ("c", "go") must match a
    whole word; a longer one may be a word prefix ("api" matches "apis"). A multi-word form also
    counts when every word of 3+ letters appears somewhere ("data pipelines" in "pipelines for
    data"), and a form also matches with spaces and dots removed ("nodejs" for "node.js").
    """
    n = _norm(name)
    compact_jd = jd_norm.replace(" ", "").replace(".", "")
    for form in (n, *synonyms.get(n, ())):
        if not form:
            continue
        if (f" {form} " if len(form) <= 2 else f" {form}") in jd_norm:
            return True
        if len(form) > 3 and form.replace(" ", "").replace(".", "") in compact_jd:
            return True
        words = [w for w in form.split() if len(w) > 2]
        if len(words) > 1 and all(f" {w}" in jd_norm for w in words):
            return True
    return False


def apply_guards(skills: list, jd_text: str, config: ExtractionConfig) -> tuple[list, dict[str, list[str]]]:
    """Keep only grounded, specific skills, at most ``max_skills`` (required first).

    Returns ``(kept, dropped)``, where ``dropped`` maps a reason to the names it removed:
    ``ungrounded`` (the JD never names it), ``generic`` (on the owner's stoplist), ``too long``
    (more than ``max_skill_words`` words: a description, not a skill name) and ``over cap``.
    """
    jd_norm = f" {_norm(jd_text)} "
    dropped: dict[str, list[str]] = {}
    kept = []
    for item in skills:
        n = _norm(item.name)
        reason = None
        if n in config.generic_skills:
            reason = "generic"
        elif len(n.split()) > config.max_skill_words:
            reason = "too long"
        elif not is_grounded(item.name, jd_norm, config.skill_synonyms):
            reason = "ungrounded"
        if reason:
            dropped.setdefault(reason, []).append(item.name)
        else:
            kept.append(item)
    kept = sorted(kept, key=lambda i: i.requirement != "required")  # stable: required first, order kept
    if len(kept) > config.max_skills:
        dropped["over cap"] = [i.name for i in kept[config.max_skills :]]
        kept = kept[: config.max_skills]
    return kept, dropped


def dictionary_extras(
    llm_names: list[str],
    clean_text: str,
    company: str,
    domain_bucket: str | None,
    slug_names: dict[str, str],
    config: ExtractionConfig,
) -> tuple[list[str], dict[str, list[str]]]:
    """The second skill producer (D10.3): dictionary skills the LLM did not list, as our vocabulary names.

    The step-9 comparison found the LLM leaves out skills named inside long lists, and a prompt change did
    not bring them back; freehire's dictionary finds them deterministically. They are added as
    good_to_have by the caller. Returns ``(extras, dropped)``; a dictionary skill is dropped as ``generic``
    (the owner's generic or dictionary_skip list), ``company`` (the hiring company's own name: freehire
    tags "stripe" on every Stripe posting) or ``listed`` (the LLM listed it, or a skill containing it or
    contained in it as whole words: "spark" beside "apache spark").
    """
    company_norm = f" {_norm(company or '')} "
    listed = [f" {_norm(n)} " for n in llm_names]
    extras: list[str] = []
    dropped: dict[str, list[str]] = {}
    for slug in skilltag.parse(clean_text, skilltag.DOMAIN_CATEGORY.get(domain_bucket or "")):
        name = slug_names[slug]
        n = _norm(name)
        reason = None
        if not n or n in config.generic_skills or n in config.dictionary_skip:
            reason = "generic"
        elif company_norm.startswith(f" {n} "):
            reason = "company"
        elif any(f" {n} " in m or m in f" {n} " for m in listed):
            reason = "listed"
        if reason:
            dropped.setdefault(reason, []).append(name)
        elif name not in extras:
            extras.append(name)
    return extras, dropped


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
    retrying = Retrying(
        retry=retry_if_exception(lambda exc: classify_failure(exc) == "transient"),
        stop=stop_after_attempt(4),
        wait=jittered_wait,  # the owner's rule: 1, 2, 4, 8 s + jitter, or the server's own delay
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
    model: str = MODEL_NAME,
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
    stamp = extraction_version(config, model)
    dict_stamp = dictionary_version(config)
    slug_names = skilltag.vocab_names(conn)
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
                skills=[],
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

        kept, dropped = apply_guards(result.skills, clean_text, config)
        if dropped:
            add_note(
                conn,
                job_id,
                "extraction: guards dropped " + "; ".join(f"{k}: {', '.join(v)}" for k, v in dropped.items()),
            )
        confidence = result.extraction_confidence
        if not kept and confidence == "high":
            confidence = "low"  # no specific, grounded skill survived: not a reliable extraction
        skills = [{"name": vocab.resolve_skill(conn, item.name), "requirement": item.requirement} for item in kept]
        set_extraction(
            conn,
            job_id,
            skills=skills,
            domain_bucket=result.domain_bucket,
            legitimacy=result.legitimacy,
            extraction_confidence=confidence,
            extraction_version=stamp,
        )
        extras, _ = dictionary_extras(
            [s["name"] for s in skills], clean_text, job["company"], result.domain_bucket, slug_names, config
        )
        set_dictionary_skills(conn, job_id, [vocab.resolve_skill(conn, n) for n in extras], dict_stamp)
        set_status(conn, job_id, "extracted")
        stats.extracted += 1

    return stats


def current_stamps(config: ExtractionConfig) -> list[str]:
    """Every stamp that counts as current: Gemini's, and the fallback model's when one is set."""
    stamps = [extraction_version(config)]
    if config.fallback_provider:
        stamps.append(extraction_version(config, config.fallback_model))
    return stamps


def stale_jobs(conn, config: ExtractionConfig | None = None) -> dict[str | None, list[dict]]:
    """Extracted/scored jobs whose stamp is not the current one, grouped by their old stamp
    (None = written before stamps existed). Lists only: re-extracting costs 1 request per job,
    so it is always the owner's call. ``config`` gives the guarded stamp (default: the shipped config)."""
    if config is None:
        config = load_extraction_config()
    groups: dict[str | None, list[dict]] = {}
    stamps = current_stamps(config)
    rows = conn.execute(
        "SELECT job_key AS job_id, company, title, extraction_version FROM jobs "
        "WHERE status IN ('extracted', 'scored') "
        "AND (extraction_version IS NULL OR NOT extraction_version = ANY(%s)) "
        "ORDER BY extraction_version, job_key",
        (stamps,),
    )
    for row in rows:
        groups.setdefault(row["extraction_version"], []).append(dict(row))
    return groups


def reguard_jobs(conn, config: ExtractionConfig) -> dict[str, int]:
    """Apply the current guards to ``extracted`` jobs written by the SAME prompt and schema without
    guards (the unguarded stamp). No API call: the guards only need the stored skills and the JD.
    Each job's skills are filtered, its confidence drops to low if none survive, and it gets the
    guarded stamp. Scored jobs are left alone (their scores would go stale silently)."""
    unguarded, guarded = extraction_version(), extraction_version(config)
    counts = {"jobs": 0, "skills_dropped": 0, "now_low_confidence": 0}
    rows = select_jobs(conn, "j.status = 'extracted' AND j.extraction_version = %s", (unguarded,), "j.job_key")
    for row in rows:
        clean_text, _ = preprocess(row["description"] or "", row["company"], config)
        items = [
            SkillItem(name=s["name"], requirement=s["requirement"])
            for s in row["skills"] or []
            if s.get("source", "llm") == "llm"
        ]
        found = [s for s in row["skills"] or [] if s.get("source") == "dictionary"]  # not the LLM's: kept as is
        kept, dropped = apply_guards(items, clean_text, config)
        confidence = row["extraction_confidence"]
        if not kept and items and confidence == "high":
            confidence = "low"
            counts["now_low_confidence"] += 1
        if dropped:
            add_note(
                conn,
                row["job_id"],
                "extraction: guards dropped " + "; ".join(f"{k}: {', '.join(v)}" for k, v in dropped.items()),
            )
            counts["skills_dropped"] += sum(len(v) for v in dropped.values())
        set_skills(
            conn,
            row["job_id"],
            [{"name": i.name, "requirement": i.requirement} for i in kept] + found,
            extraction_confidence=confidence,
            extraction_version=guarded,
        )
        counts["jobs"] += 1
    return counts


def add_dictionary_skills(conn, config: ExtractionConfig) -> dict[str, int]:
    """Run the dictionary producer (D10.3) on every extracted or scored job whose dictionary stamp is not the
    current one. No API call: it needs only the stored LLM skills and the JD. Scored jobs keep their score
    until the next ``python -m hiredra.scoring run --force``."""
    stamp = dictionary_version(config)
    slug_names = skilltag.vocab_names(conn)
    counts = {"jobs": 0, "scored": 0, "skills_added": 0}
    rows = select_jobs(
        conn,
        "j.status IN ('extracted', 'scored') AND j.extraction_confidence IS NOT NULL "
        "AND j.dictionary_version IS DISTINCT FROM %s",
        (stamp,),
        "j.job_key",
    )
    for row in rows:
        clean_text, _ = preprocess(row["description"] or "", row["company"], config)
        llm = [s["name"] for s in row["skills"] or [] if s.get("source", "llm") == "llm"]
        before = {s["name"] for s in row["skills"] or [] if s.get("source") == "dictionary"}
        extras, _ = dictionary_extras(llm, clean_text, row["company"], row["domain_bucket"], slug_names, config)
        names = [n if n in before else vocab.resolve_skill(conn, n) for n in extras]  # a sighting counts once
        set_dictionary_skills(conn, row["job_id"], names, stamp)
        counts["jobs"] += 1
        counts["scored"] += row["status"] == "scored"
        counts["skills_added"] += len(names)
    return counts


_STOP_HINTS = {
    "daily_quota": "wait for the daily reset",
    "transient": "re-run in a few minutes",
    "api_error": "check the key, the model name and the request",
    "bad_response": "check the model's answer",
    "schema_error": "read the 'schema error' notes, fix the prompt or schema, re-run",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.extraction")
    parser.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="process at most N filtered jobs")
    parser.add_argument("--stale", action="store_true", help="list jobs extracted with an older stamp (no API call)")
    parser.add_argument(
        "--reguard", action="store_true", help="apply the current guards to jobs extracted without them (no API call)"
    )
    parser.add_argument(
        "--dictionary",
        action="store_true",
        help="add the skill dictionary's finds to extracted/scored jobs not yet stamped with it (no API call)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "nvidia"],
        default="auto",
        help="auto: Gemini, then the configured fallback when Gemini's daily budget is spent",
    )
    args = parser.parse_args(argv)

    try:
        config = load_extraction_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as err:
        print(f"config error: {err}", file=sys.stderr)
        return 1

    if args.reguard:
        conn = connect(args.db)
        try:
            init_db(conn)
            counts = reguard_jobs(conn, config)
        finally:
            conn.close()
        print(
            f"reguard: {counts['jobs']} job(s) re-checked, {counts['skills_dropped']} skill(s) dropped, "
            f"{counts['now_low_confidence']} now low confidence; stamp {extraction_version(config)}"
        )
        return 0

    if args.dictionary:
        conn = connect(args.db)
        try:
            init_db(conn)
            counts = add_dictionary_skills(conn, config)
        finally:
            conn.close()
        print(
            f"dictionary: {counts['jobs']} job(s), {counts['skills_added']} skill(s) added as good_to_have; "
            f"stamp {dictionary_version(config)}"
        )
        if counts["scored"]:
            print(f"  {counts['scored']} scored job(s) changed: re-score with `python -m hiredra.scoring run --force`")
        return 0

    if args.stale:
        conn = connect(args.db)
        try:
            init_db(conn)
            groups = stale_jobs(conn, config)
        finally:
            conn.close()
        print(f"current stamps: {', '.join(current_stamps(config))}")
        for stamp, jobs in groups.items():
            print(f"  {stamp or '(no stamp)'}: {len(jobs)} job(s)")
            for j in jobs:
                print(f"    {j['job_id']}  {j['company']} -- {j['title']}")
        print(f"stale: {sum(len(j) for j in groups.values())}")
        return 0

    stages = []
    if args.provider in ("auto", "gemini"):
        stages.append("gemini")
    if args.provider == "nvidia" or (args.provider == "auto" and config.fallback_provider == "nvidia"):
        stages.append("nvidia")
    if "nvidia" in stages and not config.fallback_model:
        print("config error: no fallback.model set in config/extraction.yml", file=sys.stderr)
        return 1

    results: list[tuple[str, ExtractionStats, str]] = []  # (stage, stats, today's usage, read while connected)
    try:
        with RunLock(LOCK_NAME, what="extraction", kind="flash-lite", url=args.db):
            conn = connect(args.db)
            try:
                init_db(conn)
                remaining = args.limit
                for stage in stages:
                    if stage == "gemini":
                        usage = DailyUsage(config.daily_budget, conn, "gemini", MODEL_NAME)
                        if usage.remaining() == 0:
                            print(f"gemini NOT RUN: {usage.summary()}")
                            if stages[-1] == "gemini":
                                return 2
                            continue
                        try:
                            chain = build_extraction_chain()
                        except RuntimeError as err:
                            print(f"config error: {err}", file=sys.stderr)
                            return 1
                        stats = run_extraction(conn, config, chain, usage=usage, limit=remaining)
                    else:
                        usage = DailyUsage(  # e.g. nvidia / nemotron-3.5-super-vl-120b-a12b
                            config.fallback_requests_per_day,
                            conn,
                            config.fallback_provider,
                            config.fallback_model.split("/", 1)[-1],
                        )
                        try:
                            chain = NvidiaChain(
                                config.fallback_model,
                                ExtractionResult,
                                base_url=config.fallback_base_url,
                                timeout=config.fallback_timeout_seconds,
                                max_tokens=config.fallback_max_tokens,
                            )
                        except RuntimeError as err:
                            print(f"nvidia NOT RUN: {err}")
                            break
                        paced = dataclasses.replace(
                            config,
                            requests_per_minute=max(
                                1, int(config.fallback_requests_per_minute * config.fallback_headroom)
                            ),
                        )
                        stats = run_extraction(
                            conn, paced, chain, usage=usage, limit=remaining, model=config.fallback_model
                        )
                    results.append((stage, stats, f"{usage.used()}/{usage.budget}"))
                    if remaining is not None:
                        remaining -= stats.seen
                    # Only a spent daily budget hands over to the fallback; any other stop ends the run.
                    if stats.stopped != "daily_quota" or (remaining is not None and remaining <= 0):
                        break
            finally:
                conn.close()
    except LockHeld as err:
        print(f"NOT RUN: {err}")
        return 4

    for stage, stats, used in results:
        if stats.stopped:
            hint = _STOP_HINTS.get(stats.stopped, "")
            reset = f"; resets {reset_text(datetime.now(timezone.utc))}" if stats.stopped == "daily_quota" else ""
            print(f"{stage} STOPPED ({stats.stopped}): {stats.stop_detail} -- {hint}{reset}")
        print(
            f"{stage}: seen={stats.seen} extracted={stats.extracted} skipped_thin={stats.skipped_thin} "
            f"schema_failed={stats.schema_failed} stopped={stats.stopped or 'none'} "
            f"used_today={used}"
        )
    return 2 if results and results[-1][1].stopped else 0


if __name__ == "__main__":
    sys.exit(main())
