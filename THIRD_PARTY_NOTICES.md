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
| [open-jobs](https://github.com/elliottdehn/open-jobs) | CC0 1.0 | `tools/locparse.py`, ported to Python module `hiredra/location.py`, with its bare-city table copied as `hiredra/city_countries.json`. Its published job export: the `enrichment_json` skill fields (market skill vocabulary) and the per-ATS crawled postings. Its board roster `slugs.json`. | `hiredra/location.py`, `hiredra/city_countries.json`, `hiredra/graph/harvest.py`, `discovery/harvest.py`, `scripts/generate_company_list.py` |

Versions read: career-ops at commit `fe06051`, OpenJobs at `cdcc533`, open-jobs at `a2d88f4`.

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

### open-jobs (CC0 1.0 Universal)

open-jobs is dedicated to the public domain under the Creative Commons CC0 1.0 Universal
dedication. No notice is legally required; it is credited here as a courtesy. Full text:
https://creativecommons.org/publicdomain/zero/1.0/legalcode
