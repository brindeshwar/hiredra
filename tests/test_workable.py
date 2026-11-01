"""Workable provider (build step 3): cases ported from freehire's ``workable_test.go`` (MIT). Faked transport, no
network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import workable


class _Http:
    def __init__(self, body):
        self.body = body
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def _fetch(http, board="huggingface"):
    entry = CompanyEntry(name="Hugging Face", provider="workable", extra={"slug": board})
    return workable.WorkableProvider().fetch(entry, FetchContext(http=http))


def test_one_request_lists_the_board_with_its_text():
    http = _Http({"jobs": [{
        "title": "Backend Engineer", "shortcode": "ABC123", "url": "https://apply.workable.com/j/ABC123",
        "published_on": "2024-01-15", "city": "Berlin", "state": "", "country": "Germany", "telecommuting": True,
        "type": "Full-time", "description": "<p>Build <strong>things</strong>.</p><script>x()</script>",
    }]})  # fmt: skip
    [job] = _fetch(http)
    assert http.asked == ["https://apply.workable.com/api/v1/widget/accounts/huggingface?details=true"]
    assert (job.job_id, job.url, job.title, job.company) == (
        "workable:ABC123", "https://apply.workable.com/j/ABC123", "Backend Engineer", "Hugging Face"
    )  # fmt: skip
    assert (job.location, job.is_remote, job.job_type) == ("Berlin, Germany", True, "fulltime")
    assert job.description == "Build things."
    assert job.date_posted == "2024-01-15"


@pytest.mark.parametrize(
    ("kind", "want"),
    [("Part-time", "parttime"), ("Contract", "contract"), ("Temporary", "contract"), ("Internship", "intern"),
     ("Other", None), (None, None)],
)  # fmt: skip
def test_job_type(kind, want):
    [job] = _fetch(_Http({"jobs": [{"title": "X", "shortcode": "C1", "type": kind}]}))
    assert job.job_type == want
    assert job.url == "https://apply.workable.com/j/C1" and job.is_remote is None


def test_an_emptied_board_is_empty_and_a_listing_without_jobs_an_error():
    assert _fetch(_Http({"name": "Acme", "jobs": []})) == []
    with pytest.raises(FetchError):
        _fetch(_Http({"name": "Acme"}))
    with pytest.raises(FetchError):
        _fetch(_Http(FetchError("boom")))


def test_a_job_listed_once_per_location_is_one_job_with_every_location():
    entry = {"title": "QA Engineer", "shortcode": "D04A", "type": "Full-time"}
    http = _Http({"jobs": [
        {**entry, "city": "Berlin", "country": "Germany"},
        {**entry, "city": "Bengaluru", "country": "India"},
        {**entry, "city": "Berlin", "country": "Germany"},
        {"title": "Designer", "shortcode": "E05B", "country": "Poland", "telecommuting": True},
    ]})  # fmt: skip
    qa, designer = _fetch(http)
    assert (qa.job_id, qa.location, qa.is_remote) == ("workable:D04A", "Berlin, Germany; Bengaluru, India", None)
    assert (designer.location, designer.is_remote) == ("Poland", True)


def test_a_posting_without_title_or_shortcode_is_skipped():
    assert _fetch(_Http({"jobs": [{"title": "X"}, {"shortcode": "Y"}]})) == []


def test_board_url_and_a_bad_board():
    provider = workable.WorkableProvider()
    entry = CompanyEntry(name="Acme", provider="workable", extra={"slug": "acme-co"})
    assert provider.board_url(entry) == "https://apply.workable.com/acme-co/"
    with pytest.raises(FetchError):
        _fetch(_Http({"jobs": []}), board="../admin")
