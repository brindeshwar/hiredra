"""Tests for the Step 7 rating sheet (hiredra/scoring/rate.py, plan unit S7.1)."""

from __future__ import annotations

import csv
import json

import pytest

from hiredra.scoring import rate
from hiredra.storage import connect, init_db, upsert_job

FEATURE_KEYS = (
    "score_features",
    "coverage",
    "gap_distance",
    "score_terms",
    "nearest",
    "fit ",
    "band",
    "score_at_sampling",
)


@pytest.fixture()
def conn():
    c = connect(":memory:")
    init_db(c)
    for i in range(60):  # 50 clean + 10 on the review list
        jid = f"greenhouse:{i}"
        upsert_job(c, {"job_id": jid, "url": f"https://x/{i}", "title": f"Engineer {i}", "company": "Acme",
                       "source": "greenhouse", "description": f"Build things.\nJob {i}."})  # fmt: skip
        c.execute(
            "UPDATE jobs SET status = 'scored', score = ?, score_features = ?, skills = ?, review_reason = ? WHERE job_id = ?",
            (
                round(i * 0.0371, 4),
                json.dumps({"coverage": 0.5, "gap_distance": 0.1}),
                json.dumps(
                    [{"name": "python", "requirement": "required"}, {"name": "aws", "requirement": "good_to_have"}]
                ),
                json.dumps(["suspect"]) if i >= 50 else None,
                jid,
            ),
        )
    c.commit()
    yield c
    c.close()


def test_sample_is_stratified_split_and_excludes_the_review_list(conn):
    picks = rate.sample(conn)
    assert len(picks) == 30
    assert sum(p["split"] == "tune" for p in picks) == 20 and sum(p["split"] == "holdout" for p in picks) == 10
    assert all(int(p["job_id"].split(":")[1]) < 50 for p in picks)  # never a review-list job
    pool = sorted(range(50), key=lambda i: round(i * 0.0371, 4))
    for d in range(10):  # 3 per decile: 2 tune + 1 holdout
        stratum = {f"greenhouse:{i}" for i in pool[d * 5 : (d + 1) * 5]}
        mine = [p for p in picks if p["job_id"] in stratum]
        assert len(mine) == 3 and sorted(p["split"] for p in mine) == ["holdout", "tune", "tune"]
    assert [p["sitting"] for p in picks] == [1] * 15 + [2] * 15


def test_sample_is_deterministic_and_each_sitting_spans_the_score_range(conn):
    a, b = rate.sample(conn), rate.sample(conn)
    assert a == b
    pool = sorted(range(50), key=lambda i: round(i * 0.0371, 4))
    decile = {f"greenhouse:{i}": n // 5 for n, i in enumerate(pool)}
    order = [decile[p["job_id"]] for p in a]
    assert order != sorted(order)  # not drawn decile by decile
    for sitting in (1, 2):  # each sitting sees low and high scores alike
        mine = [decile[p["job_id"]] for p in a if p["sitting"] == sitting]
        assert min(mine) <= 2 and max(mine) >= 7


def test_sample_refuses_a_pool_that_is_too_small(conn):
    conn.execute("UPDATE jobs SET review_reason = '[\"suspect\"]' WHERE CAST(substr(job_id, 12) AS INT) >= 20")
    with pytest.raises(rate.SheetError):
        rate.sample(conn)


def test_the_sheet_is_blind(conn):
    picks = rate.sample(conn)
    sheet = rate.render_sheet(conn, picks)
    for key in FEATURE_KEYS:
        assert key not in sheet
    for p in picks:  # no score value appears either
        assert (
            f"{p['score_at_sampling']:.4f}" not in sheet
            and str(p["score_at_sampling"]) not in sheet
            or p["score_at_sampling"] == 0
        )
    assert sheet.count("Rating: [ ] apply   [ ] maybe   [ ] skip") == 30
    assert "aws, python" in sheet  # skills alphabetical


def _tick(sheet: str, n: int, rating: str, reason: str) -> str:
    """Rate block J<n> the way the owner would."""
    head = f"### J{n:02d}."
    a = sheet.index(head)
    b = sheet.index("job_id:", a)
    block = sheet[a:b].replace(f"[ ] {rating}", f"[x] {rating}").replace("Reason: ", f"Reason: {reason}")
    return sheet[:a] + block + sheet[b:]


def test_ingest_appends_ticked_jobs_and_skips_the_rest(conn, tmp_path):
    sheet, key, out = tmp_path / "s.md", tmp_path / "k.json", tmp_path / "r.csv"
    rate.write_sample(conn, sheet=sheet, key=key)
    text = _tick(sheet.read_text("utf-8"), 1, "apply", "agent work, fresher friendly")
    text = _tick(text, 2, "skip", "needs 5 years")
    sheet.write_text(text, "utf-8")

    assert rate.ingest(sheet=sheet, key=key, ratings=out) == (2, 0)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["rating"] for r in rows] == ["apply", "skip"]
    assert rows[0]["reason"] == "agent work, fresher friendly"
    assert set(rows[0]) == set(rate.CSV_FIELDS) and rows[0]["split"] in ("tune", "holdout")

    assert rate.ingest(sheet=sheet, key=key, ratings=out) == (0, 2)  # re-ingest adds nothing


def test_a_changed_rating_is_a_new_row_and_the_latest_wins(conn, tmp_path):
    sheet, key, out = tmp_path / "s.md", tmp_path / "k.json", tmp_path / "r.csv"
    rate.write_sample(conn, sheet=sheet, key=key)
    base = sheet.read_text("utf-8")
    sheet.write_text(_tick(base, 1, "maybe", "unsure"), "utf-8")
    rate.ingest(sheet=sheet, key=key, ratings=out)
    sheet.write_text(_tick(base, 1, "apply", "on reflection, yes"), "utf-8")
    assert rate.ingest(sheet=sheet, key=key, ratings=out) == (1, 0)
    assert [r["rating"] for r in csv.DictReader(out.open(encoding="utf-8"))] == ["maybe", "apply"]


@pytest.mark.parametrize(
    ("rating_line", "reason", "problem"),
    [
        ("Rating: [x] apply   [x] maybe   [ ] skip", "two boxes", "exactly one"),
        ("Rating: [x] apply   [ ] maybe   [ ] skip", "", "reason is required"),
        ("Rating: [ ] apply   [ ] maybe   [ ] skip", "a reason but no tick", "exactly one"),
    ],
)
def test_a_badly_filled_job_is_refused_with_its_label(conn, rating_line, reason, problem):
    sheet = rate.render_sheet(conn, rate.sample(conn))
    a = sheet.index("### J03.")
    b = sheet.index("job_id:", a)
    block = (
        sheet[a:b]
        .replace("Rating: [ ] apply   [ ] maybe   [ ] skip", rating_line)
        .replace("Reason: ", f"Reason: {reason}")
    )
    with pytest.raises(rate.SheetError, match=f"J03: .*{problem}"):
        rate.parse_sheet(sheet[:a] + block + sheet[b:])


def test_sample_refuses_to_overwrite_an_existing_sheet(conn, tmp_path):
    sheet, key = tmp_path / "s.md", tmp_path / "k.json"
    rate.write_sample(conn, sheet=sheet, key=key)
    with pytest.raises(rate.SheetError, match="already exists"):
        rate.write_sample(conn, sheet=sheet, key=key)
    assert rate.write_sample(conn, sheet=sheet, key=key, force=True) == 30
