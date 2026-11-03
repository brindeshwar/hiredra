# Third-party notices

Hiredra reuses work from the open-source projects below. This file lists each one, its
license, what Hiredra uses from it, and where that use lives in this repo.

None of these projects' files are copied into this repo, except where a row says so.
Their repos are cloned locally into `reference/` (git-ignored) only to read them and,
for the data rows, to read their published data on the owner's machine.

| Source | License | What Hiredra uses | Where |
|---|---|---|---|
| [career-ops](https://github.com/santifer/career-ops) | MIT | Design of the ATS providers (Greenhouse, Lever, Ashby), the provider shape `{id, detect(), fetch()}`, and the HTTP retry/backoff pattern, re-written in Python. No JavaScript is copied or run. | `hiredra/sourcing/http.py`, `hiredra/sourcing/provider.py`, `hiredra/sourcing/providers/` |
| [OpenJobs](https://github.com/outscal/OpenJobs) (a fork of career-ops) | MIT | Its company dataset `data/companies_v2.json` (the `ats_links` field), read from the local clone to build the candidate company pool. No code is copied or run. | `discovery/harvest.py` |
| [freehire](https://github.com/strelov1/freehire) | MIT | Location parsing rules from `internal/dict/location`, re-written in Python: the US-state / country-code clash rule, the GeoNames last-resort city rule, UK postcodes, Cyrillic city prefixes, title restrictions, and its curated place names and region-name tables (copied as data). Its location test cases, copied as a test fixture. The Lever description assembly (`description` + `lists` + `additional`) and the years-parser age/cap rules (`internal/job/jobfacts`), re-written. Its GeoNames city export `cities1000.tsv`, copied unchanged. The company-name key (`internal/dict/normalize`: `CompanySlug`, `Slug`, the legal-form list) and the role fingerprint (`internal/job/jobhash/rolefingerprint.go`), re-written, with their test cases. The company-merge design (`cmd/merge-companies`, `docs/agents/company-identity.md`: group by the squashed key, a curated list for the rest, old spellings kept as aliases), re-written. The board-checking idea (`cmd/harvest-boards`: a guessed slug counts only if the live ATS answers) and the homonym guard of `cmd/import-yc` (a board listing more jobs than the startup has people is a namesake), re-written. The board-link detection of `internal/ingest/atsdetect` (Greenhouse, Lever and Ashby URL shapes) and the careers-page walk of `cmd/harvest-ats` (common careers paths and hosts, one link deeper), re-written. The orphan worklist of `cmd/harvest-orphans` (companies seen only through aggregators, board ids guessed from the name with the `group`/`S.p.A.` tail off, a guess two employers propose dropped), re-written. The Workday adapter (`internal/ingest/sources/workday.go`: the CXS listing and detail endpoints, the shared-host board form, paging by the first page's total, the facet split past the 2,000 cap), re-written, with its test cases. The Oracle Recruiting Cloud and SmartRecruiters adapters (`oracle.go`: the requisition list with the offset inside the finder, the detail by id, the corporate-text fallback; `smartrecruiters.go`: paging by `totalFound`, the role sections before the company one, a detail without a `postingUrl` treated as unread), re-written, with their test cases. The Workable and iCIMS adapters (`workable.go`: the widget listing with `details=true`; `icims.go`: the sitemap followed one level, the `?in_iframe=1` page's JobPosting, the `UNAVAILABLE` placeholder), its sitemap and ld+json readers (`sitemap.go`, `jsonld.go`), re-written, with their test cases. The Taleo Business Edition and SuccessFactors adapters (`taleobe.go`: the RSS feed through the dispatcher, a gone board read from its redirect; `successfactors.go`: the job sitemap, the microdata page, the richest description region) and its microdata readers (`html.go`: `itempropHTML`, `metaProperty`), re-written, with their test cases. The Keka and Zoho Recruit adapters (`keka.go`: the portal's jobs-active API; `zoho.go`: the careers page's `#jobs` list, the detail record's `Job_Description` and `Date_Opened`, the fixpoint JS unescape), re-written, with their test cases. The Teamtailor, Rippling and Eightfold adapters (`teamtailor.go`: the `jobs.rss` feed and its `remoteStatus` reading; `rippling.go`: the board list and the detail's role text; `eightfold.go`: the two list generations, paging by the stated count, the detail by id), re-written, with their test cases. The Amazon and IBM adapters (`amazon.go`: the amazon.jobs search API; `ibm.go`: IBM's site-search API in a total order, the facet suffix cut, the mojibake and bullet repairs, the location rule), re-written, with their test cases. The Apple adapter (`apple.go`: the position rows deduped before the detail API, the summary, description and qualification sections), re-written, with its test cases. The RemoteOK, Remotive, Jobicy, Himalayas, Working Nomads and We Work Remotely adapters (`remoteok.go`, `remotive.go`, `jobicy.go`, `himalayas.go`: the unresolved `name` company and the self-promotion trailer; `workingnomads.go`: the id from the link; `weworkremotely.go`: the `Company: Role` title), re-written, with their test cases. The Hacker News adapter (`hackernews.go`: the `whoishiring` thread search, the `Company | Role | Location` header and its role-first and commitment rules), re-written, with its test cases. The Muse adapter (`themuse.go`: the public jobs API, the page limit past `page_count`, the posting mapping), re-written, with its test cases. The EPAM adapter (`epam.go`: the gzip sitemap's English vacancy links and their uid, the JobPosting's applicant countries and `TELECOMMUTE`, the description rebuilt from the page's `__NEXT_DATA__` sections with leaked headings dropped), re-written, with its test cases. The Getro job page's `__NEXT_DATA__` `currentJob` record (`getro.go`), read in our board discovery (no Getro adapter: its search API is closed to us). Its skill descriptions (`internal/dict/skilltag/descriptions.tsv`) joined with the display labels from `labels.go`, copied as data. Its skill-tagging dictionary (`internal/dict/skilltag`: word and phrase aliases, ambiguous and proper-noun words, acronym tiers, labels, aliases), exported unchanged as data, and its matcher (`skilltag.go`, `internal/dict/wordmatch`) re-written in Python; its own test texts with the Go code's output, copied as a test fixture. | `hiredra/location.py`, `hiredra/geonames_cities.tsv`, `tests/fixtures/freehire_location_cases.json`, `hiredra/sourcing/providers/lever.py`, `hiredra/filter.py`, `hiredra/sourcing/identity.py`, `hiredra/sourcing/companies.py`, `hiredra/sourcing/discover.py`, `hiredra/sourcing/providers/workday.py`, `hiredra/sourcing/providers/oracle.py`, `hiredra/sourcing/providers/smartrecruiters.py`, `hiredra/sourcing/providers/workable.py`, `hiredra/sourcing/providers/icims.py`, `hiredra/sourcing/providers/taleobe.py`, `hiredra/sourcing/providers/successfactors.py`, `hiredra/sourcing/providers/keka.py`, `hiredra/sourcing/providers/zohorecruit.py`, `hiredra/sourcing/providers/teamtailor.py`, `hiredra/sourcing/providers/rippling.py`, `hiredra/sourcing/providers/eightfold.py`, `hiredra/sourcing/providers/amazon.py`, `hiredra/sourcing/providers/ibm.py`, `hiredra/sourcing/providers/apple.py`, `hiredra/sourcing/providers/epam.py`, `hiredra/sourcing/providers/remoteboards.py`, `hiredra/sourcing/providers/hackernews.py`, `hiredra/sourcing/providers/themuse.py`, `hiredra/sourcing/providers/_common.py`, `hiredra/sourcing/_html.py`, `tests/test_workday.py`, `tests/test_oracle.py`, `tests/test_smartrecruiters.py`, `tests/test_workable.py`, `tests/test_icims.py`, `tests/test_taleobe.py`, `tests/test_successfactors.py`, `tests/test_keka.py`, `tests/test_zohorecruit.py`, `tests/test_teamtailor.py`, `tests/test_rippling.py`, `tests/test_eightfold.py`, `tests/test_amazon.py`, `tests/test_ibm.py`, `tests/test_apple.py`, `tests/test_epam.py`, `tests/test_remoteboards.py`, `tests/test_hackernews.py`, `tests/test_themuse.py`, `tests/test_identity.py`, `hiredra/graph/skill_descriptions.tsv`, `hiredra/skill_dict.json`, `hiredra/skilltag.py`, `tests/fixtures/freehire_skilltag_cases.json` |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Alibaba Qwen team), run with [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (Apache 2.0) on [PyTorch](https://pytorch.org) (BSD-3) | Apache 2.0 | The local skill-embedding model (decision D1.15). Its weights are downloaded at first use into `data/models/` (git-ignored); nothing of it is copied into this repository. The libraries are the optional `embed` dependency group. | `hiredra/graph/embedder.py`, `config/graph.yml` |
| [ats-scrapers](https://github.com/kalil0321/ats-scrapers) | MIT | Its public company lists per ATS (`ats-companies/greenhouse.csv`, `lever.csv`, `ashby.csv`: name, slug, careers URL), read by `boards import` into the local database to find boards (decision D2.2). The files are not in this repository. | `hiredra/sourcing/config.py` (`load_inventory`) |
| [yc-oss/api](https://github.com/yc-oss/api) | none stated | Its public Y Combinator directory (`companies/all.json`): company names, websites, hiring regions and team sizes, read on the owner's machine to guess job-board slugs (owner decision, 2026-10-31). Facts only; never copied into this repository or republished. | `hiredra/sourcing/discover.py` (`yc_candidates`) |
| [GeoNames](https://www.geonames.org/) (via freehire) | CC BY 4.0 | The `cities1000` place names and countries in `hiredra/geonames_cities.tsv` (freehire's export of it, unchanged). | `hiredra/geonames_cities.tsv` |
| [open-jobs](https://github.com/elliottdehn/open-jobs) | CC0 1.0 | `tools/locparse.py`, ported to Python module `hiredra/location.py`, with its bare-city table copied as `hiredra/city_countries.json`. Its published job export: the `enrichment_json` skill fields (market skill vocabulary) and the per-ATS crawled postings. Its board roster `slugs.json`. | `hiredra/location.py`, `hiredra/city_countries.json`, `hiredra/graph/harvest.py`, `discovery/harvest.py`, `scripts/generate_company_list.py` |

Versions read: career-ops at commit `fe06051`, OpenJobs at `cdcc533`, open-jobs at `a2d88f4`, freehire at `8b55a65` (the skill dictionary at `0e715dfc`), ats-scrapers at `6b44a1b`.

Later units will add one more use, recorded here when they land: the HTML/CSS structure of
career-ops' CV and cover-letter templates, adapted into Jinja2 templates.

## Notes on the data

- **open-jobs** dedicates its repository to the public domain under CC0 1.0. The job export
  is published by that project; the repo names no separate license for it, and Hiredra uses
  it on that basis. Hiredra does not redistribute the export. It reads it at build time, and
  only derived values (skill names, board slugs, counts) land in the local database.
  The job postings themselves come from public employer job boards.
- **OpenJobs**' dataset is read from the local clone and is not redistributed. Its README
  notes the data is a point-in-time snapshot.

## License texts

### career-ops (MIT)

```
MIT License

Copyright (c) 2026 Santiago Fernández de Valderrama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### OpenJobs (MIT)

OpenJobs uses the same MIT text as above, with these copyright lines:

```
Copyright (c) 2026 Santiago Fernández de Valderrama (original career-ops work)
Copyright (c) 2026 Outscal (modifications and additions in this fork — Outscal-Jobs)
```

### freehire (MIT)

freehire uses the same MIT text as career-ops above, with this copyright line:

```
Copyright (c) 2026 freehire contributors
```

### ats-scrapers (MIT)

ats-scrapers uses the same MIT text as career-ops above, with this copyright line:

```
Copyright (c) 2026 Kalil Bouzigues
```

### GeoNames (CC BY 4.0)

`hiredra/geonames_cities.tsv` contains data from GeoNames (https://www.geonames.org/), licensed under
Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/). Hiredra uses freehire's
export of the `cities1000` dump unchanged, apart from a two-line provenance header.

### open-jobs (CC0 1.0 Universal)

open-jobs is dedicated to the public domain under the Creative Commons CC0 1.0 Universal
dedication. No notice is legally required; it is credited here as a courtesy. Full text:
https://creativecommons.org/publicdomain/zero/1.0/legalcode
