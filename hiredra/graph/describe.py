"""Write a plain-language description for every skill freehire does not describe (owner, 2026-10-29).

    python -m hiredra.graph.describe                 # describe every undescribed graph skill (resumable)
    python -m hiredra.graph.describe --limit 80      # a first slice
    python -m hiredra.graph.describe --holdout 60    # quality test: describe 60 skills freehire DOES describe,
                                                     #   without showing its text, into the holdout file

Why: the embedding input is "name: description" (D1.15), and freehire describes only 603 of our 2,566 skills.

How (the LLM writes, code decides -- CLAUDE.md rule 3):
- one Gemini flash-lite call per BATCH skills (structured output), with freehire descriptions as the examples
  of the standard, so ~1,960 skills take ~50 requests; extraction's daily meter and the ``llm`` run lock are
  shared, and calls are paced under the free tier's 15 requests/minute;
- every description is CHECKED against freehire's style before it is kept (``problems``); one that fails, or a
  name the model calls ambiguous or not a skill, stays undescribed -- the skill is then embedded by its bare
  name, never by a guess;
- results go to ``skill_descriptions_generated.tsv`` beside freehire's file (reviewable, versioned, one row per
  skill). They feed the embedding input only once the measured comparison shows they help.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_after_attempt

from hiredra.graph.descriptions import DESCRIPTIONS_PATH, load_descriptions
from hiredra.lock import LockHeld, RunLock
from hiredra.quota import DailyQuotaExceeded, DailyUsage, classify_failure, jittered_wait

MODEL_NAME = "gemini-3.5-flash-lite"  # the flash-lite meter extraction counts on, too
PROMPT_VERSION = "describe-2"  # describe-1 lacked the "technology job posting" context
GENERATED_PATH = DESCRIPTIONS_PATH.with_name("skill_descriptions_generated.tsv")
HOLDOUT_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration" / "describe_holdout.tsv"
BATCH = 40
REQUESTS_PER_MINUTE = 13  # the free tier's 15, with headroom

#: freehire descriptions shown as the standard: languages, platforms, tools, practices, soft skills, compliance,
#: an acronym and an engineering method. Never described themselves in a holdout.
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
    description: str | None  # None: ambiguous, too vague, or not a skill


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


def build_prompt(names: Sequence[str], examples: dict[str, str]) -> str:
    shown = "\n".join(f"- {n}: {examples[n]}" for n in EXAMPLES if n in examples)
    asked = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    return (
        "You write entries for a glossary of job skills. Each entry is a short, plain-language description of what "
        "the skill is and what it is used for at work.\n\n"
        "Follow the standard of these entries exactly:\n"
        f"{shown}\n\n"
        "Rules:\n"
        "- 1 or 2 sentences, about 100-250 characters, ending with a full stop.\n"
        "- Start with the skill's name (proper capitalisation is fine), then 'is' or 'are': define it first.\n"
        "- Neutral and factual. No marketing words, no 'you' or 'we', no lists, no markdown, no links.\n"
        "- Every name comes from a technology job posting (software, data, AI, cloud, security), so give an "
        "acronym or a product name its technology meaning. Describe the common meaning of the name in those "
        "postings. If a name is still ambiguous, too vague to describe, or not a skill at all, give null as its "
        "description instead of guessing.\n"
        "- Return one item for every name below, with the name copied exactly as given.\n\n"
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


@dataclass
class DescribeStats:
    asked: int = 0
    kept: int = 0
    declined: int = 0  # the model said: ambiguous / not a skill
    rejected: dict[str, list[str]] = field(default_factory=dict)  # name -> what missed the standard
    missing: int = 0  # not returned by the model at all
    requests: int = 0
    stopped: str | None = None


def read_tsv(path: Path) -> dict[str, str]:
    """``name -> description`` from a generated file (``#`` lines are its header)."""
    if not path.exists():
        return {}
    rows = [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return {name: text for name, text in rows}


def _header(path: Path) -> None:
    """The file's header on first use; later, a marker line whenever a newer prompt version starts writing, so
    every row can be traced to the prompt that wrote it."""
    marker = f"# rows below: {MODEL_NAME}, {PROMPT_VERSION}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Skill descriptions written by {MODEL_NAME} (hiredra/graph/describe.py) for skills freehire does not\n"
            "# describe, each checked against freehire's standard before it was kept.\n"
            "# Columns: name<TAB>description. Review freely; delete a row to have it rewritten.\n"
            f"{marker}\n",
            encoding="utf-8",
            newline="\n",
        )
        return
    markers = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("# rows below:")]
    if not markers or markers[-1] != marker:
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(marker + "\n")


def describe(
    names: Sequence[str],
    chain,
    path: Path,
    *,
    usage: DailyUsage | None = None,
    examples: dict[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda _msg: None,
) -> DescribeStats:
    """Describe ``names`` in batches, appending every kept description to ``path`` as it lands (resumable: names
    already in the file are skipped)."""
    examples = examples if examples is not None else load_descriptions()
    done = read_tsv(path)
    todo = [n for n in dict.fromkeys(names) if n not in done]
    stats = DescribeStats(asked=len(todo))
    _header(path)
    retrying = Retrying(
        retry=retry_if_exception(lambda exc: classify_failure(exc) == "transient"),
        stop=stop_after_attempt(4),
        wait=jittered_wait,
        sleep=sleep,
        reraise=True,
    )
    gap, next_at = 60.0 / REQUESTS_PER_MINUTE, None

    def call(prompt: str) -> DescribedBatch:
        nonlocal next_at
        if usage is not None:
            usage.reserve(1)
            usage.record(1)  # counted before it is sent, failed attempts too (as extraction does)
        if next_at is not None and clock() < next_at:
            sleep(next_at - clock())
        next_at = clock() + gap
        stats.requests += 1
        return chain.invoke(prompt)

    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            result = retrying(call, build_prompt(batch, examples))
        except Exception as exc:
            kind = "daily_quota" if isinstance(exc, DailyQuotaExceeded) else classify_failure(exc)
            if kind is None:
                raise
            if kind == "daily_quota" and usage is not None:
                usage.mark_exhausted(str(exc).splitlines()[0][:200])
            stats.stopped = kind
            log(f"STOPPED ({kind}): {str(exc).splitlines()[0][:200]} -- finished batches are saved")
            break
        got = {item.name: item.description for item in result.items}
        rows = []
        for name in batch:
            if name not in got:
                stats.missing += 1
            elif not got[name]:
                stats.declined += 1
            elif bad := problems(name, got[name].strip()):
                stats.rejected[name] = bad
            else:
                rows.append(f"{name}\t{got[name].strip()}\n")
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.writelines(rows)
        stats.kept += len(rows)
        log(f"  batch {start // BATCH + 1}: kept {len(rows)}/{len(batch)}")
    return stats


def undescribed_skills(conn) -> list[str]:
    """Graph skills with no freehire description, most-seen first (the ones scoring meets most)."""
    have = load_descriptions()
    rows = conn.execute("SELECT name FROM skill_vocab ORDER BY mine DESC, seen_count DESC, name").fetchall()
    return [n for (n,) in rows if n.lower() not in have]


def holdout_skills(n: int, seed: int = 20261029) -> list[str]:
    """``n`` freehire-described names (never an example), for the blind quality test."""
    pool = sorted({name for name in load_descriptions() if name not in EXAMPLES and " " not in name.strip("-")})
    return random.Random(seed).sample(pool, n)


def main(argv: list[str] | None = None) -> int:
    from hiredra.storage import connect

    ap = argparse.ArgumentParser(prog="python -m hiredra.graph.describe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    ap.add_argument("--limit", type=int, default=None, help="describe at most this many skills")
    ap.add_argument("--holdout", type=int, default=None, help="quality test: describe N freehire-described skills")
    ap.add_argument("--budget", type=int, default=450, help="flash-lite requests/day this meter allows")
    args = ap.parse_args(argv)

    conn = connect(args.db)
    try:
        if args.holdout:
            names, path = holdout_skills(args.holdout), HOLDOUT_PATH
            examples = {k: v for k, v in load_descriptions().items() if k not in names}
        else:
            names, path, examples = undescribed_skills(conn), GENERATED_PATH, None
        names = names[: args.limit] if args.limit else names
        usage = DailyUsage(args.budget, conn, "gemini", MODEL_NAME)
        print(f"{len(names)} skill(s) to describe into {path}; today: {usage.summary()}", flush=True)
        with RunLock("llm", what="skill descriptions", kind="flash-lite", url=args.db):
            stats = describe(names, build_chain(), path, usage=usage, examples=examples,
                             log=lambda m: print(m, flush=True))  # fmt: skip
    except LockHeld as exc:
        print(f"NOT RUN: {exc}")
        return 4
    finally:
        conn.close()
    print(
        f"asked {stats.asked}, kept {stats.kept}, declined {stats.declined}, rejected {len(stats.rejected)}, "
        f"missing {stats.missing}; {stats.requests} request(s); stopped={stats.stopped or 'none'}"
    )
    for name, why in sorted(stats.rejected.items())[:30]:
        print(f"  rejected {name}: {'; '.join(why)}")
    return 2 if stats.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
