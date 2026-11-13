"""Tests for the sidecar API (piece 3): auth, the job views and detail, triage, runs and packet files."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hiredra.api.__main__ import main as api_main
from hiredra.api.app import create_app
from hiredra.api.runs import Runs
from hiredra.storage import connect, get_events, get_job, init_db

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _jobs(conn):
    from tests.test_graph_clean import _job

    keys = [_job(conn, n, [("python", "required")]) for n in (1, 2, 3)]
    conn.execute("UPDATE jobs SET status = 'scored', score = 2.0 WHERE job_key = %s", (keys[0],))
    conn.execute(
        "UPDATE jobs SET status = 'scored', score = 1.0, review_reason = ARRAY['suspect'] WHERE job_key = %s",
        (keys[1],),
    )
    conn.execute("UPDATE jobs SET status = 'tailored', score = 1.5 WHERE job_key = %s", (keys[2],))
    return keys


@pytest.fixture()
def api(pg_url, tmp_path):
    conn = connect(pg_url)
    init_db(conn)
    keys = _jobs(conn)
    stages = {"echo": ["-c", "print('stage ran')"], "slow": ["-c", "import time; time.sleep(5)"],
              "fails": ["-c", "import sys; print('NOT RUN: lock held'); sys.exit(4)"]}  # fmt: skip
    client = TestClient(create_app(token=TOKEN, db_url=pg_url, runs=Runs(stages), packets_dir=tmp_path))
    yield client, conn, keys, tmp_path
    client.app.state.runs.stop_all()
    conn.close()


def test_health_is_open_everything_else_needs_the_token(api):
    client, *_ = api
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/status", headers=AUTH).status_code == 200


def test_status_and_the_job_views(api):
    client, _, keys, _ = api
    st = client.get("/status", headers=AUTH).json()
    assert st["jobs"]["scored"] == 2 and st["jobs"]["tailored"] == 1 and "usage" in st
    ids = lambda view: [i["job_id"] for i in client.get("/jobs", params={"view": view}, headers=AUTH).json()["items"]]  # noqa: E731
    assert ids("ranked") == [keys[0]] and ids("review") == [keys[1]] and ids("tailored") == [keys[2]]
    assert ids("all") == [keys[0], keys[2], keys[1]]  # best score first
    assert client.get("/jobs", params={"view": "nope"}, headers=AUTH).status_code == 400
    detail = client.get(f"/jobs/{keys[0]}", headers=AUTH).json()
    assert detail["job_id"] == keys[0] and detail["skills"][0]["name"] == "python"
    assert client.get("/jobs/greenhouse:999", headers=AUTH).status_code == 404


def test_release_and_reject_round_trip_with_events(api):
    client, conn, keys, _ = api
    assert client.post(f"/jobs/{keys[1]}/release", headers=AUTH).json() == {"ok": True}
    assert get_job(conn, keys[1])["released_at"] is not None
    assert client.post(f"/jobs/{keys[0]}/reject", json={"reason": "not for me"}, headers=AUTH).json() == {"ok": True}
    job = get_job(conn, keys[0])
    assert (job["status"], job["status_detail"]) == ("dropped", "review: not for me")
    assert any(e["kind"] == "status" for e in get_events(conn, keys[0]))
    assert client.post(f"/jobs/{keys[2]}/reject", headers=AUTH).status_code == 409  # tailored: not from the list
    events = client.get(f"/jobs/{keys[0]}/events", headers=AUTH).json()
    assert events and events[0]["detail"]


def _wait(client, run_id):
    for _ in range(100):
        run = client.get(f"/runs/{run_id}", headers=AUTH).json()
        if run["state"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_runs_start_follow_refuse_doubles_and_report_failures(api):
    client, *_ = api
    run = client.post("/runs/echo", headers=AUTH).json()
    done = _wait(client, run["id"])
    assert (done["state"], done["exit_code"], done["log"]) == ("ok", 0, ["stage ran"])
    slow = client.post("/runs/slow", headers=AUTH).json()
    assert client.post("/runs/slow", headers=AUTH).status_code == 409  # one run per stage
    failed = _wait(client, client.post("/runs/fails", headers=AUTH).json()["id"])
    assert failed["state"] == "failed" and failed["exit_code"] == 4 and "NOT RUN" in failed["log"][0]
    assert client.post("/runs/nope", headers=AUTH).status_code == 404
    assert {r["stage"] for r in client.get("/runs", headers=AUTH).json()} == {"echo", "slow", "fails"}
    assert client.get("/runs/missing", headers=AUTH).status_code == 404
    assert slow["state"] == "running"


def test_packet_listing_and_files_stay_inside_the_packet(api):
    client, _, keys, packets = api
    folder = packets / keys[2].replace(":", "_")
    folder.mkdir()
    (folder / "cv.pdf").write_bytes(b"%PDF-1.4 test")
    (folder / "secret.txt").write_text("no")
    got = client.get(f"/packets/{keys[2]}", headers=AUTH).json()
    assert got["files"] == ["cv.pdf"] and got["artifacts"] == []
    r = client.get(f"/packets/{keys[2]}/files/cv.pdf", headers=AUTH)
    assert r.status_code == 200 and r.content == b"%PDF-1.4 test"
    assert client.get(f"/packets/{keys[2]}/files/secret.txt", headers=AUTH).status_code == 400
    assert client.get(f"/packets/{keys[2]}/files/cover.pdf", headers=AUTH).status_code == 404


def test_the_server_refuses_to_start_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv("HIREDRA_API_TOKEN", raising=False)
    assert api_main([]) == 2 and "HIREDRA_API_TOKEN" in capsys.readouterr().err
    with pytest.raises(ValueError):
        create_app(token="")
