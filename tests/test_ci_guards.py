"""Tests for scripts/ci_guards.py: each guard catches its planted violation and ignores the allowed cases."""

from pathlib import Path

import pytest

from scripts import ci_guards


def _write(root: Path, rel: str, code: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")


def _guards(root: Path) -> list[str]:
    return [v.guard for v in ci_guards.run(root)]


def test_real_repo_is_clean():
    assert ci_guards.run() == []


def test_missing_directories_pass(tmp_path):
    assert ci_guards.run(tmp_path) == []


@pytest.mark.parametrize(
    "code",
    [
        "import playwright\n",
        "from playwright.sync_api import sync_playwright\n",
        "import playwright.async_api as pw\n",
        "import importlib\nimportlib.import_module('playwright.sync_api')\n",
        "__import__('playwright')\n",
    ],
)
def test_guard_a_flags_playwright_outside_the_driver(tmp_path, code):
    _write(tmp_path, "hiredra/gate/review.py", code)
    assert _guards(tmp_path) == ["G-a"]


def test_guard_a_allows_the_driver_and_other_packages(tmp_path):
    _write(tmp_path, "hiredra/autofill/driver.py", "from playwright.sync_api import sync_playwright\n")
    _write(tmp_path, "hiredra/resume/render.py", "from playwright.sync_api import sync_playwright\n")
    _write(tmp_path, "hiredra/autofill/notes.py", "# playwright is only used by driver.py\nX = 'playwright'\n")
    assert ci_guards.run(tmp_path) == []


@pytest.mark.parametrize(
    "code",
    [
        "import httpx\nhttpx.post('https://x', json={})\n",
        "def f(client):\n    client.put('https://x')\n",
        "def f(client):\n    client.request('POST', 'https://x')\n",
        "def f(client):\n    client.request(method='put', url='https://x')\n",
        "import urllib.request\nurllib.request.Request('https://x', data=b'1')\n",
        "from urllib.request import urlopen\nurlopen('https://x', data=b'1')\n",
    ],
)
def test_guard_b_flags_write_requests(tmp_path, code):
    _write(tmp_path, "hiredra/outreach/drafts.py", code)
    assert set(_guards(tmp_path)) == {"G-b"}


def test_guard_b_allows_reads_and_other_packages(tmp_path):
    _write(tmp_path, "hiredra/autofill/scan.py", "import httpx\nhttpx.get('https://x')\n")
    _write(tmp_path, "hiredra/gate/batch.py", "def f(client):\n    client.request('GET', 'https://x')\n")
    _write(tmp_path, "hiredra/sourcing/http.py", "import httpx\nhttpx.post('https://x')\n")
    assert ci_guards.run(tmp_path) == []


@pytest.mark.parametrize(
    "code",
    [
        "import smtplib\n",
        "from email.mime.text import MIMEText\n",
        "import slack_sdk\n",
        "from twilio.rest import Client\n",
    ],
)
def test_guard_c_flags_mail_and_messaging_clients_anywhere(tmp_path, code):
    _write(tmp_path, "hiredra/tracking/notify.py", code)
    assert _guards(tmp_path) == ["G-c"]


def test_guard_c_allows_unrelated_email_modules(tmp_path):
    _write(tmp_path, "hiredra/tracking/parse.py", "import email.utils\nimport smtpd_like_name_but_not_smtplib\n")
    assert ci_guards.run(tmp_path) == []


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ci_guards, "REPO", tmp_path)
    monkeypatch.setattr(ci_guards, "run", lambda root=tmp_path: [])
    assert ci_guards.main() == 0
    monkeypatch.setattr(ci_guards, "run", lambda root=tmp_path: [ci_guards.Violation("G-a", "x.py", 1, "d")])
    assert ci_guards.main() == 1
    assert "G-a: x.py:1: d" in capsys.readouterr().out
