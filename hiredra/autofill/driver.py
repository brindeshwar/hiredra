"""The fill driver: types the prepared answers into an application form and moves through its pages, and never
submits it (hard rule 1; plan decision D29). The only module in ``hiredra/autofill/`` allowed to import Playwright (CI
guard G-a).

What it does: fills text fields and text areas, picks select options, ticks ordinary radio buttons and checkboxes,
attaches the résumé and cover letter, and clicks a Next / Continue button to reach the next page. On the last page it
fills fields and stops, and the human reviews and clicks Submit.

What it never does, checked in code before every click (``refusal``):

- click a control of ``type=submit`` unless its label is a recognised Next / Continue;
- click anything whose visible label, value or aria-label says submit, apply, send, finish, complete, confirm, done
  or review (in any case);
- click anything on the last page (a page with no Next / Continue);
- touch a field the scan marked PAUSE (consent, legal, demographic, salary, work authorization...);
- press Enter, call ``form.submit()`` or dispatch a submit event (none of these exist anywhere in this module).

During this build it runs only against local fixture forms (``tests/fixtures/forms/``); the first fill on a real site
happens later, with the owner watching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Locator, Page

#: Labels that move to the next page. Matched on the whole label, case-insensitive.
NEXT_LABELS = re.compile(r"^\s*(next|continue|next page|next step|save (and|&) continue|proceed)\s*(›|>|→)?\s*$", re.I)
#: Words that mean "send the application": a control carrying any of them is never clicked.
SUBMIT_WORDS = re.compile(r"submit|apply|send|finish|complete|confirm|done|review", re.I)


@dataclass
class Report:
    filled: list[str] = field(default_factory=list)  # labels only, never values
    skipped: list[str] = field(default_factory=list)
    paused: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)  # "label: reason"
    pages: int = 0
    last_page_reached: bool = False


def _label(el: Locator) -> str:
    parts = [el.inner_text(timeout=1000) if el.evaluate("e => e.tagName") != "INPUT" else "",
             el.get_attribute("value") or "", el.get_attribute("aria-label") or "", el.get_attribute("title") or ""]  # fmt: skip
    return " ".join(" ".join(p.split()) for p in parts if p).strip()


def refusal(el: Locator) -> str | None:
    """Why ``el`` must not be clicked, or None. Every click the driver makes goes through this."""
    label = _label(el)
    kind = (el.get_attribute("type") or "").lower()
    if NEXT_LABELS.match(label) and not SUBMIT_WORDS.search(label):
        return None  # a plain Next / Continue, whatever its type
    if kind == "submit":
        return "a submit-type control that is not a Next / Continue"
    if SUBMIT_WORDS.search(label):
        return f"its label says {SUBMIT_WORDS.search(label).group(0)!r}"
    return "not a recognised Next / Continue control"


def safe_click(el: Locator, report: Report) -> bool:
    reason = refusal(el)
    if reason:
        report.refused.append(f"{_label(el) or '(no label)'}: {reason}")
        return False
    el.click()
    return True


def _next_button(page: Page) -> Locator | None:
    """The visible Next / Continue control on the current page, if there is one."""
    for el in page.locator(
        "button:visible, input[type=button]:visible, input[type=submit]:visible, a[role=button]:visible"
    ).all():
        if NEXT_LABELS.match(_label(el)) and not SUBMIT_WORDS.search(_label(el)):
            return el
    return None


def _fill_one(page: Page, f: dict, report: Report) -> bool:
    """Fill one prepared field if its control is visible on this page. True when it was handled here."""
    target = page.locator(f'[name="{f["key"]}"]')
    if target.count() == 0 or not target.first.is_visible():
        return False
    label = f["label"]
    if f["class"] == "PAUSE":
        report.paused.append(label)
        return True
    if f["class"] == "FILE":
        path = Path(f.get("value") or "")
        if path.exists():
            target.first.set_input_files(str(path))
            report.filled.append(label)
        else:
            report.skipped.append(f"{label}: file not found")
        return True
    value = f.get("value")
    if not value:
        report.skipped.append(f"{label}: no value")
        return True
    first = target.first
    kind = (first.get_attribute("type") or first.evaluate("e => e.tagName")).lower()
    if kind == "select":
        first.select_option(label=value)
    elif kind in ("radio", "checkbox"):
        option = page.locator(f'[name="{f["key"]}"][value="{value}"]')
        if option.count():
            option.first.check()
        else:
            report.skipped.append(f"{label}: no option {value!r}")
            return True
    else:
        first.fill(value)  # typing only: no key presses, so no Enter
    report.filled.append(label)
    return True


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "forms"


def run_fixture(name: str, *, headless: bool = True, keep_open=None) -> tuple[Report, dict]:
    """Fill the local fixture form ``name`` (``tests/fixtures/forms/<name>.html``) with its answers file, served from
    127.0.0.1 only. Returns the report and the page's own counters: ``submits`` (submit events) and ``finals`` (clicks
    on its final buttons), which must both stay 0. ``keep_open`` (a callable) runs before the browser closes, so a
    human can look at the last page."""
    import functools
    import http.server
    import json
    import threading

    from playwright.sync_api import sync_playwright

    answers = json.loads((FIXTURES / f"{name}.answers.json").read_text(encoding="utf-8"))
    repo = FIXTURES.parents[2]
    for f in answers["fields"]:
        if f["class"] == "FILE" and f.get("value"):
            f["value"] = str(repo / f["value"])

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # no request log on the console
            pass

    handler = functools.partial(Quiet, directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/{name}.html")
            report = fill_form(page, answers)
            counters = page.evaluate("() => ({submits: window.__submits, finals: window.__finals})")
            if keep_open is not None:
                keep_open()
            browser.close()
    finally:
        server.shutdown()
    return report, counters


def fill_form(page: Page, answers: dict, *, max_pages: int = 10) -> Report:
    """Fill every page of the form open in ``page``; stop on the last page without clicking anything there."""
    report = Report()
    pending = list(answers.get("fields") or [])
    for _ in range(max_pages):
        report.pages += 1
        pending = [f for f in pending if not _fill_one(page, f, report)]
        nxt = _next_button(page)
        if nxt is None:
            report.last_page_reached = True
            # Record (never click) what a human would press here, so the hand-over is visible in the report.
            for el in page.locator("button:visible, input[type=submit]:visible").all():
                if refusal(el):
                    report.refused.append(f"{_label(el) or '(no label)'}: last page, left for you")
            break
        if not safe_click(nxt, report):
            break
        page.wait_for_load_state("domcontentloaded")
    report.skipped += [f"{f['label']}: not on any page" for f in pending if f["class"] != "PAUSE"]
    return report
