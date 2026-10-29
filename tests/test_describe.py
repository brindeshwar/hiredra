"""The skill-description writer (hiredra/graph/describe.py). A fake chain stands in for Gemini: no API call."""

from __future__ import annotations

import pytest

from hiredra.graph import describe as d
from hiredra.graph.descriptions import load_descriptions
from hiredra.quota import DailyQuotaExceeded

GOOD = "Dagster is an open-source orchestrator for data pipelines, used to schedule and monitor data assets."


class FakeChain:
    """Answers each prompt from a script: name -> description (or None), and records the prompts."""

    def __init__(self, answers, fail=None):
        self.answers, self.fail, self.prompts = answers, fail, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise self.fail
        names = [ln.split(". ", 1)[1] for ln in prompt.split("Skills:\n")[1].splitlines() if ". " in ln]
        return d.DescribedBatch(
            items=[d.Described(name=n, description=self.answers.get(n)) for n in names if n in self.answers]
        )


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
    rows = [ln.split("\t") for ln in d.DESCRIPTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]  # fmt: skip
    passing = sum(1 for slug, label, text in rows if not d.problems(label or slug.replace("-", " "), text))
    assert passing / len(rows) >= 0.97  # the reference meets its own standard (measured 1,039 of 1,064)


def test_the_prompt_shows_freehire_examples_and_lists_every_name():
    prompt = d.build_prompt(["dagster", "prefect"], load_descriptions())
    assert "- kafka: Kafka is" in prompt and "- stakeholder management:" in prompt
    assert "1. dagster\n2. prefect" in prompt and "null" in prompt


def test_kept_declined_rejected_and_missing_are_sorted_out_and_saved(tmp_path):
    path = tmp_path / "gen.tsv"
    chain = FakeChain({"dagster": GOOD, "xyz": None, "prefect": "Prefect is great."})  # "zzz" is never answered
    stats = d.describe(["dagster", "xyz", "prefect", "zzz"], chain, path, examples={}, sleep=lambda s: None)
    assert (stats.kept, stats.declined, list(stats.rejected), stats.missing, stats.requests) == (
        1,
        1,
        ["prefect"],
        1,
        1,
    )
    assert d.read_tsv(path) == {"dagster": GOOD}
    assert path.read_text(encoding="utf-8").startswith("# Skill descriptions written by")


def test_a_rerun_resumes_and_batches_cut_the_requests(tmp_path):
    path = tmp_path / "gen.tsv"
    names = [f"skill{i}" for i in range(d.BATCH + 5)]
    answers = {
        n: f"{n.capitalize()} is a made-up tool used here to test batching of skill descriptions." for n in names
    }
    chain = FakeChain(answers)
    first = d.describe(names[: d.BATCH], chain, path, examples={}, sleep=lambda s: None)
    second = d.describe(names, chain, path, examples={}, sleep=lambda s: None)
    assert (first.requests, first.kept) == (1, d.BATCH)  # one request for a whole batch
    assert (second.asked, second.requests, second.kept) == (5, 1, 5)  # only the rest
    assert len(d.read_tsv(path)) == d.BATCH + 5


def test_requests_are_paced_under_the_per_minute_limit(tmp_path):
    names = [f"skill{i}" for i in range(3 * d.BATCH)]
    answers = {n: f"{n.capitalize()} is a made-up tool used here to test the pacing of requests." for n in names}
    now, waits = [0.0], []
    d.describe(names, FakeChain(answers), tmp_path / "g.tsv", examples={}, clock=lambda: now[0],
               sleep=lambda s: waits.append(s) or now.__setitem__(0, now[0] + s))  # fmt: skip
    assert len(waits) == 2 and all(w == pytest.approx(60 / d.REQUESTS_PER_MINUTE) for w in waits)


class Meter:
    def __init__(self, room):
        self.room, self.recorded, self.exhausted = room, 0, None

    def reserve(self, n):
        if self.recorded + n > self.room:
            raise DailyQuotaExceeded("budget used")

    def record(self, n):
        self.recorded += n

    def mark_exhausted(self, note=None):
        self.exhausted = note


def test_the_daily_meter_counts_every_request_and_stops_the_run_cleanly(tmp_path):
    names = [f"skill{i}" for i in range(2 * d.BATCH)]
    answers = {n: f"{n.capitalize()} is a made-up tool used here to test the daily request meter." for n in names}
    meter = Meter(room=1)
    stats = d.describe(names, FakeChain(answers), tmp_path / "g.tsv", usage=meter, examples={}, sleep=lambda s: None)
    assert (meter.recorded, stats.kept, stats.stopped) == (1, d.BATCH, "daily_quota") and meter.exhausted


def test_an_unknown_error_is_raised_not_swallowed(tmp_path):
    with pytest.raises(KeyError):
        d.describe(
            ["dagster"], FakeChain({}, fail=KeyError("bug")), tmp_path / "g.tsv", examples={}, sleep=lambda s: None
        )


def test_the_holdout_never_uses_an_example_and_is_reproducible():
    picks = d.holdout_skills(30)
    assert picks == d.holdout_skills(30) and not set(picks) & set(d.EXAMPLES)


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


def test_each_prompt_version_marks_the_rows_it_writes(tmp_path, monkeypatch):
    path = tmp_path / "gen.tsv"
    answers = {"dagster": GOOD, "prefect": GOOD.replace("Dagster", "Prefect")}
    d.describe(["dagster"], FakeChain(answers), path, examples={}, sleep=lambda s: None)
    d.describe(["dagster"], FakeChain(answers), path, examples={}, sleep=lambda s: None)  # same version: no new marker
    monkeypatch.setattr(d, "PROMPT_VERSION", "describe-9")
    d.describe(["prefect"], FakeChain(answers), path, examples={}, sleep=lambda s: None)
    markers = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("# rows below:")]
    assert [m.split(", ")[-1] for m in markers] == [d.PROMPT_VERSION.replace("9", "2"), "describe-9"]
    assert d.read_tsv(path) == answers
