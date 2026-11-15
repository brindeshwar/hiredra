"""Tests for the end-to-end runner: order, skipping, stopping at the first failure, and the summary."""

from __future__ import annotations

from hiredra import e2e


def _stages(*codes: int):
    return [(f"s{i}", ["-c", f"import sys; print('stage {i} ran'); sys.exit({code})"]) for i, code in enumerate(codes)]


def test_runs_every_stage_in_order_and_skips_on_request():
    lines = []
    results = e2e.run(_stages(0, 0, 0), skip={"s1"}, out=lines.append)
    assert [r.name for r in results] == ["s0", "s2"] and all(r.exit_code == 0 for r in results)
    assert [r.last_line for r in results] == ["stage 0 ran", "stage 2 ran"]
    assert "Done." in e2e.summary(results)


def test_stops_at_the_first_failure_and_names_a_held_lock():
    results = e2e.run(_stages(0, 4, 0), out=lambda _l: None)
    assert [(r.name, r.exit_code) for r in results] == [("s0", 0), ("s1", 4)]  # s2 never ran
    text = e2e.summary(results)
    assert "STOPPED at s1 (exit 4)" in text and "lock" in text


def test_top_and_no_drafts_reach_the_commands():
    seen = []
    stages = [("packets", ["-c", "import sys; print(sys.argv[1:])", "{top}"]),
              ("forms", ["-c", "import sys; print(sys.argv[1:])"])]  # fmt: skip
    results = e2e.run(stages, top=7, no_drafts=True, out=seen.append)
    assert results[0].last_line == "['7']" and results[1].last_line == "['--no-drafts']"
    assert e2e.NAMES == ["filter", "extraction", "sync", "describe", "scoring", "audit", "packets", "forms", "batch"]
