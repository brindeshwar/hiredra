"""The describe step (owner, 2026-11-12): every new skill gets a description, and a check for "is this a skill
we already have under another name?" -- so the graph never again fills up with undescribed names and unmerged
variants.

    python -m hiredra.graph.describe            # after `python -m hiredra.graph sync`, before scoring
    python -m hiredra.graph.describe --dry-run  # the queue and the first prompt; no API call

How (the LLM reads, code decides -- CLAUDE.md rule 3):

- the queue is every node with no description yet (``description_source`` empty): first the ones that jobs waiting
  to be scored use (scoring holds a job until its skills are described, decision D3), then the owner's, then the
  most seen;
- each skill goes to Gemini flash-lite with its 5 nearest known skills (by the stored vectors), 40 skills a
  request. The model answers per skill: a description, or "same as <one of the 5>";
- a "same as" is applied only if code agrees it is a variant of that name (``same_skill``): an acronym, a spelling
  within two letters, or the same words plus filler words only ("mcp server" -> mcp). Never a pair on the
  never-merge list, never the owner's own skill merged away. It is merged the way ``graph clean`` merges (decision
  D2), and every merge is written to ``data/describe/merges.csv``;
- a description is kept only if it meets freehire's standard (``problems``); otherwise, or when the model gives
  none, the skill is marked ``declined`` and stays embedded by its bare name;
- budget (decision D6): at most ``describe.max_requests_per_day`` requests a day, and never into the last
  ``describe.reserve`` of the shared flash-lite meter, which extraction needs. Paced under 15 requests a minute;
  passing failures are retried until they clear.

Generated descriptions are stored, not yet embedded: ``embedding.description_sources`` decides that.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_never

from hiredra.graph.descriptions import load_descriptions
from hiredra.graph.merge import normalized_key
from hiredra.lock import LockHeld, RunLock
from hiredra.quota import DailyBudgetReached, DailyQuotaExceeded, DailyUsage, classify_failure, jittered_wait
from hiredra.vocab import is_acronym

MODEL_NAME = "gemini-3.5-flash-lite"  # the flash-lite meter extraction counts on, too
PROMPT_VERSION = "describe-3"  # describe-3: the same-skill question, and specific (not generic) wording
BATCH = 40
CANDIDATES = 5
REQUESTS_PER_MINUTE = 13  # the free tier's 15, with headroom
MERGES_PATH = Path(__file__).resolve().parents[2] / "data" / "describe" / "merges.csv"

#: freehire descriptions shown as the standard: languages, platforms, tools, practices, soft skills, compliance,
#: an acronym and an engineering method.
EXAMPLES = ("python", "kafka", "kubernetes", "react", "sql", "terraform", "figma", "agile", "stakeholder management",
            "machine learning", "llm", "soc 2", "data quality", "gcp", "content marketing", "fea")  # fmt: skip

MIN_CHARS, MAX_CHARS = 50, 300
_BANNED = re.compile(
    r"(?i:\b(?:we|our|you|your|me|my)\b)|\bI\b(?!/)|\bus\b"  # first/second person ("US" and "I/O" are fine)
    r"|(?i:https?://|www\.)|\*\*|`|^#"  # links, markdown ("C#" is a name, not a heading)
    r"|(?i:\b(?:best|leading|powerful|cutting-edge|revolutionary|world-class|state-of-the-art|seamless(?:ly)?)\b)"
    r"|(?i:\b(?:may refer|might refer|not sure|unclear|ambiguous|cannot determine|could refer|possibly)\b)"
)


class Described(BaseModel):
    name: str
    same_as: str | None = None  # one of the known skills shown for this name, when it is the same skill
    description: str | None = None  # None: ambiguous, too vague, not a skill -- or same_as is given


class DescribedBatch(BaseModel):
    items: list[Described]


def _spells_out(name: str, opening: str) -> bool:
    """An acronym opened by its expansion, as freehire writes "FEA" as "Finite Element Analysis is ...": the
    initials of the opening's first words must be the acronym (a plural "s" on it allowed: GANs)."""
    acronym = re.sub(r"s$", "", name.lower()) if len(name) > 2 else name.lower()
    if not re.fullmatch(r"[a-z]{2,6}", acronym):
        return False
    initials = "".join(w[0] for w in re.split(r"[\s-]+", opening)[: len(acronym)] if w)
    return initials == acronym


def problems(name: str, text: str) -> list[str]:
    """Why ``text`` misses freehire's standard ([] = it meets it): 1-2 sentences, 50-300 characters, a
    definition that opens with the skill's name, neutral wording, plain text."""
    out = []
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        out.append(f"length {len(text)} (want {MIN_CHARS}-{MAX_CHARS})")
    if "\n" in text or "\t" in text:
        out.append("not one line")
    sentences = len(re.findall(r"[.!?](?=\s|$)", text))
    if not 1 <= sentences <= 2 or not text.rstrip().endswith("."):
        out.append(f"{sentences} sentence(s), or no closing full stop (want 1-2)")
    opening = re.sub(r"^(?:an?|the)\s+", "", text.lower())  # "An LLM is ...", as freehire writes
    head = re.sub(r"[^a-z0-9+#]", "", opening[: len(name) + 40])
    words = [w for w in re.split(r"[^a-z0-9+#]+", name.lower()) if w]
    if not words or not (head.startswith(re.sub(r"[^a-z0-9+#]", "", words[0])) or _spells_out(name, opening)):
        out.append("does not open with the skill's name")
    if not re.search(r"\b(is|are|refers to|covers|means|describes)\b", text[: len(name) + 60], re.IGNORECASE):
        out.append("first sentence is not a definition (no 'is' / 'are')")
    banned = _BANNED.search(text)
    if banned:
        out.append(f"wording: {banned.group(0)!r}")
    return out


def build_prompt(batch: Sequence[tuple[str, Sequence[str]]], examples: dict[str, str]) -> str:
    """One request: each skill with the known skills nearest to it."""
    shown = "\n".join(f"- {n}: {examples[n]}" for n in EXAMPLES if n in examples)
    asked = "\n".join(f"{i}. {name} | known: {'; '.join(cands)}" for i, (name, cands) in enumerate(batch, 1))
    return (
        "You maintain a glossary of job skills. For every skill below you get the known skills nearest to it.\n\n"
        "First decide if the skill IS one of its known skills under another name: another spelling, a version, an "
        "abbreviation or its expansion, a translation, or the same skill with a filler word such as 'server', "
        "'tools', 'systems', 'framework' or 'programming'. Then set same_as to that known skill, copied exactly. A "
        "related tool, a competing product, or a broader or narrower skill is NOT the same: react native is not "
        "react, aws bedrock is not aws, gcp is not aws, mysql is not sql. When unsure, leave same_as null.\n\n"
        "Otherwise write its description, to the standard of these entries:\n"
        f"{shown}\n\n"
        "Rules for a description:\n"
        "- 1 or 2 sentences, about 100-250 characters, ending with a full stop.\n"
        "- Start with the skill's name (proper capitalisation is fine), then 'is' or 'are': define it first.\n"
        "- Say what kind of thing it is (a language, a database, a cloud service, a library for a named task, a "
        "practice) and the specific job it does. Avoid wording that fits any tool, such as 'a platform used to "
        "build applications'.\n"
        "- Neutral and factual. No marketing words, no 'you' or 'we', no lists, no markdown, no links.\n"
        "- Every name comes from a technology job posting (software, data, AI, cloud, security), so give an "
        "acronym or a product name its technology meaning. If a name is ambiguous, too vague to describe, or not "
        "a skill at all, give null as its description instead of guessing.\n"
        "- Return one item for every skill below, with the name copied exactly as given (without the known list).\n\n"
        f"Skills:\n{asked}\n"
    )


def build_chain(*, api_key: str | None = None):
    """The structured-output chain (the same model and settings as extraction)."""
    import os

    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (check .env)")
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=key, reasoning_effort="minimal")
    return model.with_structured_output(DescribedBatch)


# ── code decides ──────────────────────────────────────────────────────────────
def _edits(a: str, b: str) -> int:
    """Levenshtein distance (short strings only)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^\w+#]+", name.lower()) if w]


def same_skill(name: str, known: str, fillers: Collection[str]) -> str | None:
    """Why ``name`` can be the skill ``known`` under another name, or None. The model said they are the same; this
    is the check that the names allow it: one is the other's acronym; the same number of words, spelled within two
    letters (three for a name with accented letters: a translation); or ``name`` is ``known``'s words plus filler
    words only."""
    if name == known:
        return None
    if is_acronym(name, known) or is_acronym(known, name):
        return "acronym"
    a, b = normalized_key(name), normalized_key(known)
    words, base = _words(name), _words(known)
    accented = any(ord(ch) > 127 for ch in name)
    # A spelling slip changes letters inside words; it never adds a word ("ai governance" is not "governance").
    if len(words) == len(base) and min(len(a), len(b)) >= 5 and _edits(a, b) <= (3 if accented else 2):
        return "spelling"
    extra = [w for w in words if w not in base]
    if base and all(w in words for w in base) and extra and all(w in fillers for w in extra):
        return "filler words"
    return None


# ── the run ───────────────────────────────────────────────────────────────────
@dataclass
class DescribeStats:
    asked: int = 0
    described: int = 0
    declined: int = 0  # no description, or one that missed the standard
    merged: list[tuple[str, str, str]] = field(default_factory=list)  # (name, into, why)
    refused: list[tuple[str, str]] = field(default_factory=list)  # a same_as the code did not accept
    missing: int = 0  # not returned by the model: asked again next run
    requests: int = 0
    stopped: str | None = None


def queue(conn) -> list[str]:
    """Nodes with no description yet: first those used by jobs waiting to be scored, then the owner's, then the
    most seen."""
    rows = conn.execute(
        "SELECT v.name, v.mine, v.seen_count, EXISTS ("
        "  SELECT 1 FROM job_skills s JOIN jobs j ON j.id = s.job_id"
        "  LEFT JOIN skill_alias a ON a.alias = s.skill"
        "  WHERE j.status = 'extracted' AND COALESCE(a.canonical, s.skill) = v.name) AS waiting "
        "FROM skill_vocab v WHERE v.description_source IS NULL"
    ).fetchall()
    rows.sort(key=lambda r: (not r["waiting"], not r["mine"], -r["seen_count"], r["name"]))
    return [r["name"] for r in rows]


def nearest(names: Sequence[str], vectors: dict[str, np.ndarray], k: int = CANDIDATES) -> dict[str, list[str]]:
    """Each name's ``k`` nearest other nodes by cosine (the stored, current vectors)."""
    pool = list(vectors)
    if not pool:
        return {n: [] for n in names}
    matrix = np.stack([vectors[n] for n in pool])
    out = {}
    for name in names:
        if name not in vectors:
            out[name] = []
            continue
        sims = matrix @ vectors[name]
        out[name] = [pool[i] for i in np.argsort(-sims)[: k + 1] if pool[i] != name][:k]
    return out


class Budget:
    """Decision D6: the describe step's own daily cap, and never into the shared meter's reserve."""

    def __init__(self, shared: DailyUsage, own: DailyUsage, keep: int):
        self.shared, self.own, self.keep = shared, own, keep

    def reserve(self, n: int) -> None:
        self.own.reserve(n)
        self.shared.reserve(n)
        if self.shared.remaining() - n < self.keep:
            raise DailyBudgetReached(
                f"the last {self.keep} flash-lite requests today are kept for extraction ({self.shared.summary()})"
            )

    def record(self, n: int) -> None:
        self.shared.record(n)
        self.own.record(n)

    def mark_exhausted(self, note: str | None = None) -> None:
        self.shared.mark_exhausted(note)


def set_description(conn, name: str, text: str | None, source: str) -> None:
    conn.execute(
        "UPDATE skill_vocab SET description = %s, description_source = %s WHERE name = %s", (text, source, name)
    )


def _refusal(name: str, target: str, shown: Sequence[str], mine: set[str], never, fillers) -> str | None:
    if target not in shown:
        return "not one of the known skills shown"
    if frozenset((name, target)) in never:
        return "on the never-merge list"
    if name in mine:
        return "the owner's own skill is never merged away"
    if same_skill(name, target, fillers) is None:
        return "the names are not variants of each other"
    return None


def describe(
    conn,
    names: Sequence[str],
    candidates: dict[str, list[str]],
    chain,
    *,
    fillers: Collection[str],
    apply_merge: Callable[[str, str, str], bool],
    never: Collection[frozenset[str]] = frozenset(),
    budget: Budget | None = None,
    examples: dict[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda _msg: None,
) -> DescribeStats:
    """Ask about ``names`` in batches; each batch's answers are written before the next request (resumable: a
    re-run asks only what is still undescribed)."""
    examples = examples if examples is not None else load_descriptions()
    mine = {r["name"] for r in conn.execute("SELECT name FROM skill_vocab WHERE mine")}
    stats = DescribeStats(asked=len(names))
    retrying = Retrying(
        retry=retry_if_exception(lambda exc: classify_failure(exc) == "transient"),
        stop=stop_never,  # the owner's rule: keep trying; the daily meter still stops the run
        wait=jittered_wait,
        sleep=sleep,
        reraise=True,
    )
    gap, next_at = 60.0 / REQUESTS_PER_MINUTE, None

    def call(prompt: str) -> DescribedBatch:
        nonlocal next_at
        if budget is not None:
            budget.reserve(1)
            budget.record(1)  # counted before it is sent, failed attempts too (as extraction does)
        if next_at is not None and clock() < next_at:
            sleep(next_at - clock())
        next_at = clock() + gap
        stats.requests += 1
        return chain.invoke(prompt)

    for start in range(0, len(names), BATCH):
        batch = list(names[start : start + BATCH])
        try:
            result = retrying(call, build_prompt([(n, candidates.get(n, [])) for n in batch], examples))
        except Exception as exc:
            kind = classify_failure(exc)
            if kind is None:
                raise
            if isinstance(exc, DailyBudgetReached):
                kind = "daily_budget"  # our own cap or the kept reserve: Google's quota is untouched
            elif kind == "daily_quota" and budget is not None and not isinstance(exc, DailyQuotaExceeded):
                budget.mark_exhausted(str(exc).splitlines()[0][:200])  # Google itself said the day is spent
            stats.stopped = kind
            log(f"STOPPED ({kind}): {str(exc).splitlines()[0][:200]} -- finished batches are saved")
            break
        got = {" ".join(item.name.split()).lower(): item for item in result.items}
        for name in batch:
            item = got.get(name)
            if item is None:
                stats.missing += 1
                continue
            target = " ".join((item.same_as or "").split()).lower()
            if target:
                why = _refusal(name, target, candidates.get(name, []), mine, never, fillers)
                if why is None:
                    reason = same_skill(name, target, fillers)
                    if apply_merge(name, target, reason):
                        stats.merged.append((name, target, reason))
                    continue  # merged now, or already one skill earlier in this run
                stats.refused.append((name, f"same as {target}: {why}"))
            text = " ".join((item.description or "").split())
            if text and not problems(name, text):
                set_description(conn, name, text, "generated")
                stats.described += 1
            else:
                set_description(conn, name, None, "declined")
                stats.declined += 1
        log(
            f"  batch {start // BATCH + 1}: described {stats.described}, merged {len(stats.merged)}, "
            f"declined {stats.declined}"
        )
    return stats


def log_merges(merged: Sequence[tuple[str, str, str]], path: Path = MERGES_PATH) -> None:
    """Every automatic merge, appended for the owner to look over."""
    if not merged:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["at", "name", "merged_into", "why", "prompt"])
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        w.writerows([at, n, into, why, PROMPT_VERSION] for n, into, why in merged)


def main(argv: list[str] | None = None) -> int:
    from hiredra.extraction import load_extraction_config, name_resolver
    from hiredra.graph import clean
    from hiredra.graph.config import load_graph_config
    from hiredra.graph.curation import load_never_merge
    from hiredra.graph.descriptions import embedding_descriptions
    from hiredra.graph.index import load_skills
    from hiredra.graph.sync import input_texts
    from hiredra.storage import connect, get_skill_vocab, resolve_alias, skill_index_status

    ap = argparse.ArgumentParser(prog="python -m hiredra.graph.describe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    ap.add_argument("--limit", type=int, default=None, help="ask about at most this many skills")
    ap.add_argument("--dry-run", action="store_true", help="show the queue and the first prompt; no API call")
    args = ap.parse_args(argv)

    ext, cfg = load_extraction_config(), load_graph_config()
    conn = connect(args.db)
    try:
        status = skill_index_status(conn, cfg.embed_tag, input_texts(embedding_descriptions(conn, cfg)))
        if status["unembedded"] or status["stale"]:
            print(f"NOT RUN: the index is not current ({status}) -- run `python -m hiredra.graph sync` first")
            return 3
        names = queue(conn)
        names = names[: args.limit] if args.limit else names
        vectors = {s.name: s.vector for s in load_skills(conn, cfg) if s.vector is not None}
        candidates = nearest(names, vectors)
        shared = DailyUsage(ext.daily_budget, conn, "gemini", MODEL_NAME)
        own = DailyUsage(ext.describe_requests_per_day, conn, "gemini-describe", MODEL_NAME)
        print(
            f"{len(names)} skill(s) to describe; flash-lite today: {shared.summary()}; describe today: "
            f"{own.summary()}; kept for extraction: {ext.describe_reserve}",
            flush=True,
        )
        if args.dry_run:
            print(build_prompt([(n, candidates[n]) for n in names[:BATCH]], load_descriptions()))
            return 0
        resolver = name_resolver(conn, ext)

        def apply_merge(name: str, into: str, why: str) -> bool:
            into = resolve_alias(conn, into) or into  # merged away earlier in this run: follow it
            if into == name or get_skill_vocab(conn, name) is None:
                return False  # already one skill
            clean.apply(conn, resolver, [clean.Change(name, "merge", (into,), f"same skill ({why})")])
            log_merges([(name, into, why)])  # at once: a stop later in the run must not lose it
            return True

        with RunLock("llm", what="skill descriptions", kind="flash-lite", url=args.db):
            stats = describe(
                conn, names, candidates, build_chain(), fillers=ext.same_skill_fillers, apply_merge=apply_merge,
                never=load_never_merge(), budget=Budget(shared, own, ext.describe_reserve),
                log=lambda m: print(m, flush=True),
            )  # fmt: skip
    except LockHeld as exc:
        print(f"NOT RUN: {exc}")
        return 4
    finally:
        conn.close()
    print(
        f"asked {stats.asked}: described {stats.described}, merged {len(stats.merged)}, declined {stats.declined}, "
        f"missing {stats.missing}; {stats.requests} request(s); stopped={stats.stopped or 'none'}"
    )
    for name, into, why in stats.merged:
        print(f"  merged {name} -> {into} ({why})")
    for name, why in stats.refused[:30]:
        print(f"  not merged {name}: {why}")
    if stats.merged:
        print("  merges changed some jobs' skills: scoring picks them up (`python -m hiredra.scoring run --force`)")
    return 0 if stats.stopped in (None, "daily_budget", "daily_quota") else 2


if __name__ == "__main__":
    sys.exit(main())
