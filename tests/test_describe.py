"""The describe step (hiredra/graph/describe.py). A fake chain stands in for Gemini: no API call."""

from __future__ import annotations

import numpy as np
import pytest

from hiredra.graph import describe as d
from hiredra.graph.descriptions import DESCRIPTIONS_PATH, load_descriptions
from hiredra.quota import DailyBudgetReached, DailyUsage
from hiredra.storage import connect, get_skill_vocab, init_db, touch_skill_vocab

GOOD = "Dagster is an open-source orchestrator for data pipelines, used to schedule and monitor data assets."


class FakeChain:
    """Answers each prompt from a script: name -> description (str/None) or ("same", known); records prompts."""

    def __init__(self, answers, fail=None):
        self.answers, self.fail, self.prompts = answers, fail, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise self.fail
        lines = prompt.split("Skills:\n")[1].splitlines()
        names = [ln.split(". ", 1)[1].split(" | known:")[0] for ln in lines if ". " in ln]
        items = []
        for n in (n for n in names if n in self.answers):
            a = self.answers[n]
            if isinstance(a, tuple):
                items.append(d.Described(name=n, same_as=a[1]))
            else:
                items.append(d.Described(name=n, description=a))
        return d.DescribedBatch(items=items)


@pytest.mark.parametrize(
    "name, text, problem",
    [
        ("dagster", GOOD, None),
        (
            "llm",
            "An LLM (large language model) is a model trained on large amounts of text to generate language.",
            None,
        ),
        ("c#", "C# is a general-purpose programming language built for Microsoft's .NET platform.", None),
        ("dagster", "Short.", "length"),
        ("dagster", "Dagster is " + "very " * 70 + "long.", "length"),
        ("dagster", "Dagster is a tool. It runs jobs. It also schedules them. It has a UI.", "sentence"),
        (
            "dagster",
            "Dagster is an orchestrator for data pipelines used to schedule and monitor data assets",
            "full stop",
        ),
        (
            "dagster",
            "An orchestrator for data pipelines, used to schedule and monitor data assets in production.",
            "name",
        ),
        (
            "dagster",
            "Dagster, used by data teams to schedule pipelines and watch the assets they produce daily.",
            "definition",
        ),
        (
            "dagster",
            "Dagster is the leading orchestrator for data pipelines, used to schedule and monitor assets.",
            "wording",
        ),
        (
            "dagster",
            "Dagster is a tool you use to schedule and monitor data pipelines and the assets they produce.",
            "wording",
        ),
        (
            "dagster",
            "Dagster may refer to a data orchestrator used to schedule and monitor pipelines and assets.",
            "wording",
        ),
        (
            "dagster",
            "**Dagster** is an orchestrator for data pipelines, used to schedule and monitor data assets.",
            "wording",
        ),
    ],
)
def test_the_standard(name, text, problem):
    found = d.problems(name, text)
    assert (found == []) if problem is None else any(problem in p for p in found), found


def test_the_checker_accepts_almost_all_of_freehire_itself():
    rows = [ln.split("\t") for ln in DESCRIPTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]  # fmt: skip
    passing = sum(1 for slug, label, text in rows if not d.problems(label or slug.replace("-", " "), text))
    assert passing / len(rows) >= 0.97  # the reference meets its own standard (measured 1,039 of 1,064)


def test_the_prompt_shows_the_standard_and_each_skill_with_its_known_skills():
    prompt = d.build_prompt([("dagster", ["airflow", "prefect"]), ("mcp server", ["mcp"])], load_descriptions())
    assert "- kafka: Kafka is" in prompt and "- stakeholder management:" in prompt
    assert "1. dagster | known: airflow; prefect\n2. mcp server | known: mcp" in prompt
    assert "react native is not react" in prompt and "null" in prompt


FILLERS = frozenset({"server", "tools", "systems", "framework", "programming"})


@pytest.mark.parametrize(
    ("name", "known", "why"),
    [
        ("mcp server", "mcp", "filler words"),
        ("rag systems", "rag", "filler words"),
        ("langraph", "langgraph", "spelling"),
        ("microsserviços", "microservices", "spelling"),
        ("model context protocol", "mcp", "acronym"),
        ("react native", "react", None),  # "native" is not a filler word
        ("aws bedrock", "aws", None),
        ("gcp", "aws", None),
        ("mysql", "sql", None),
        ("sas", "sap", None),  # too short for a spelling match
        ("mcp", "mcp server", None),  # a short name is never folded into a longer one
        ("ai governance", "governance", None),  # found on the first real run: a spelling slip never adds a word
        ("ai image generation", "image generation", None),
        ("unity3d", "unity", "spelling"),
        ("ibatis", "mybatis", "spelling"),
    ],
)
def test_same_skill_only_where_the_names_allow_it(name, known, why):
    assert d.same_skill(name, known, FILLERS) == why


def _db(pg_url, *names, mine=()):
    conn = connect(pg_url)
    init_db(conn)
    for n in names:
        touch_skill_vocab(conn, n)
    for n in mine:
        conn.execute("UPDATE skill_vocab SET mine = true, source = 'profile' WHERE name = %s", (n,))
    return conn


def _run(conn, names, chain, candidates=None, **kw):
    merges = []
    stats = d.describe(
        conn, names, candidates or {}, chain, fillers=FILLERS, examples={}, sleep=lambda s: None,
        apply_merge=lambda n, into, why: merges.append((n, into, why)) or True, **kw,
    )  # fmt: skip
    return stats, merges


def test_descriptions_declines_merges_and_refusals_are_each_recorded(pg_url):
    conn = _db(pg_url, "dagster", "xyz", "prefect", "zzz", "mcp server", "mcp", "react native", "react")
    chain = FakeChain({
        "dagster": GOOD, "xyz": None, "prefect": "Prefect is great.",  # "zzz" is never answered
        "mcp server": ("same", "mcp"), "react native": ("same", "react"),
    })  # fmt: skip
    cands = {"mcp server": ["mcp", "langchain"], "react native": ["react", "flutter"]}
    stats, merges = _run(conn, ["dagster", "xyz", "prefect", "zzz", "mcp server", "react native"], chain, cands)
    assert merges == [("mcp server", "mcp", "filler words")]
    assert stats.refused == [("react native", "same as react: the names are not variants of each other")]
    assert (stats.described, stats.declined, stats.missing, stats.requests) == (1, 3, 1, 1)
    assert get_skill_vocab(conn, "dagster")["description_source"] == "generated"
    assert get_skill_vocab(conn, "prefect")["description_source"] == "declined"  # missed the standard
    assert get_skill_vocab(conn, "react native")["description_source"] == "declined"  # refused, no description
    assert get_skill_vocab(conn, "zzz")["description_source"] is None  # asked again next run
    assert sorted(d.queue(conn)) == ["mcp", "mcp server", "react", "zzz"]  # the fake merge changed nothing
    conn.close()


def test_a_same_as_outside_the_candidates_or_merging_the_owners_skill_is_refused(pg_url):
    conn = _db(pg_url, "langraph", "langgraph", "panda", "pandas", mine=["panda"])
    chain = FakeChain({"langraph": ("same", "langgraph"), "panda": ("same", "pandas")})
    stats, merges = _run(conn, ["langraph", "panda"], chain, {"langraph": ["langchain"], "panda": ["pandas"]})
    assert merges == []
    assert [why.split(": ", 1)[1] for _, why in stats.refused] == [
        "not one of the known skills shown",
        "the owner's own skill is never merged away",
    ]
    conn.close()


def test_the_queue_puts_skills_of_waiting_jobs_first(pg_url):
    from tests.test_graph_clean import _job

    conn = _db(pg_url, "aaa", "bbb", "ccc")
    conn.execute("UPDATE skill_vocab SET seen_count = 9 WHERE name = 'aaa'")
    conn.execute("UPDATE skill_vocab SET description = 'x', description_source = 'generated' WHERE name = 'ccc'")
    _job(conn, 1, [("bbb", "required")])
    assert d.queue(conn) == ["bbb", "aaa"]
    conn.close()


def test_nearest_lists_the_closest_other_skills():
    v = {n: np.array(x, dtype=np.float32) for n, x in {"a": [1, 0], "b": [0.9, 0.436], "c": [0, 1]}.items()}
    assert d.nearest(["a"], v, k=2) == {"a": ["b", "c"]}


def test_a_batch_is_one_request_and_requests_are_paced(pg_url):
    names = [f"skill{i}" for i in range(2 * d.BATCH + 1)]
    conn = _db(pg_url, *names)
    answers = {n: f"{n.capitalize()} is a made-up tool used here to test batching and pacing." for n in names}
    now, waits = [0.0], []
    stats = d.describe(
        conn, names, {}, FakeChain(answers), fillers=FILLERS, apply_merge=lambda *a: True, examples={},
        clock=lambda: now[0], sleep=lambda s: waits.append(s) or now.__setitem__(0, now[0] + s),
    )  # fmt: skip
    assert (stats.requests, stats.described) == (3, len(names))
    assert len(waits) == 2 and all(w == pytest.approx(60 / d.REQUESTS_PER_MINUTE) for w in waits)
    conn.close()


def test_the_budget_keeps_extractions_reserve_and_never_marks_googles_quota(pg_url):
    names = [f"skill{i}" for i in range(3 * d.BATCH)]
    conn = _db(pg_url, *names)
    shared = DailyUsage(450, conn, "gemini", d.MODEL_NAME)
    own = DailyUsage(60, conn, "gemini-describe", d.MODEL_NAME)
    shared.record(398)  # 52 left: two requests may go, a third would cut into the 50 kept for extraction
    answers = {n: f"{n.capitalize()} is a made-up tool used here to test the daily request budget." for n in names}
    stats, _ = _run(conn, names, FakeChain(answers), budget=d.Budget(shared, own, 50))
    assert (stats.requests, stats.described, stats.stopped) == (2, 2 * d.BATCH, "daily_budget")
    assert (shared.used(), own.used()) == (400, 2) and not shared.exhausted()
    tight = d.Budget(
        DailyUsage(450, conn, "gemini", d.MODEL_NAME), DailyUsage(2, conn, "gemini-describe", d.MODEL_NAME), 0
    )
    with pytest.raises(DailyBudgetReached):
        tight.reserve(1)  # describe's own cap (2 used of 2)
    conn.close()


def test_an_unknown_error_is_raised_not_swallowed(pg_url):
    conn = _db(pg_url, "dagster")
    with pytest.raises(KeyError):
        _run(conn, ["dagster"], FakeChain({}, fail=KeyError("bug")))
    conn.close()


def test_a_passing_failure_is_retried_until_it_clears(pg_url):
    # the owner's rule: keep trying, never stop on a passing failure
    import httpx

    class Busy(FakeChain):
        def __init__(self, answers, busy):
            super().__init__(answers)
            self.busy = busy

        def invoke(self, prompt):
            if self.busy:
                self.busy -= 1
                self.prompts.append(prompt)
                raise httpx.ConnectError("connection reset")
            return super().invoke(prompt)

    conn = _db(pg_url, "dagster")
    chain = Busy({"dagster": GOOD}, busy=7)
    stats, _ = _run(conn, ["dagster"], chain)
    assert (stats.described, stats.stopped, len(chain.prompts)) == (1, None, 8)
    conn.close()


@pytest.mark.parametrize(
    "name, text, ok",
    [
        (
            "gnn",
            "Graph Neural Networks are deep learning models that learn representations from graph-structured data.",
            True,
        ),
        (
            "gans",
            "Generative Adversarial Networks are models in which two networks compete to generate realistic data.",
            True,
        ),
        (
            "gnn",
            "Neural networks for graphs are deep learning models that learn from the nodes and edges of a graph.",
            False,
        ),
        (
            "kubernetes",
            "Container orchestration is what Kubernetes provides, running containers across many machines.",
            False,
        ),
    ],
)
def test_an_acronym_may_open_with_its_expansion_as_freehire_writes_fea(name, text, ok):
    assert (not any("open with" in p for p in d.problems(name, text))) is ok
