# Hiredra end-to-end build: report for the owner's review

The brief is `BUILD_E2E.md`. This build adds the parts after scoring, so the whole system now runs end to end:
tailored packets (a résumé and cover letter per job), a local API, a desktop app, and an application filler that
prepares every form and never submits. "Functional, not perfect": it works on real jobs; refinement and edge cases
come later.

Everything is on the branch `build/e2e`. **`main` is untouched**, and `build/e2e` is merged into `main` only when the
owner says they are satisfied.

## What was built

| Piece | What it does | PR |
|---|---|---|
| 1 Résumé core | An editable Word file and a one-page PDF from one data shape, in the owner's example layout (US Letter, Arial 9.5 pt, the example's section order). Page fit tightens spacing and fonts, then leaves out whole bullets; it never rewrites. | #177 |
| 2 Tailoring | A model (NVIDIA while building, Gemini untouched) writes a summary and bullets from the master profile; code takes company, role and dates from the profile and builds the skills section. A **claim gate** in code refuses any number, skill or banned phrase the profile does not support; what still fails after one retry is replaced by the profile's own words. Cover letter with the same gate. Packets under `data/packets/<job>/`. | #178 |
| 3 Sidecar API | `python -m hiredra.api`: local only (127.0.0.1), a fresh token per launch, jobs, triage, pipeline runs, packets, batches, apply views and decisions. Never returns a secret. | #179, #181 |
| 4 Desktop app | Electron + React in `app/`: Pipeline, Jobs, Packets, Batches and Apply screens. It starts and stops the Python service itself. Every Electron security flag is on and tested. Adding a screen is one file and one line. | #180 |
| 5 Application filler | Reads Greenhouse (public API) and Lever (apply page) forms with GET only; sorts each field into filled-from-facts, attached file, drafted answer, or left for you (salary, notice period, work authorization, relocation, demographics, consent, CAPTCHA, conditional and unknown questions are always yours). Batches of up to 10. The fill driver clicks only Next / Continue and refuses every submit-like control. | #181 |
| 6 End to end | `python -m hiredra.e2e`: every stage in order, stopping at the first failure. This report. | this PR |

## How to run it

```bash
python -m hiredra.e2e --top 10                       # the whole pipeline; leaves a batch for you
python -m hiredra.resume packets --top 5             # just packets
python -m hiredra.autofill scan --tailored           # read forms, prepare answers
python -m hiredra.autofill batch                     # queue the next batch of up to 10
python -m hiredra.autofill fill --fixture three_page # watch the filler stop before Submit (local test form)
```

The desktop app (PowerShell: one command per line):

```bash
cd app
npm install
npm run dev
```

## Verified on the real database (counts only)

- **The end-to-end run** (`python -m hiredra.e2e --top 5 --skip filter --skip extraction`, 2026-11-15): every
  stage exited 0 in about 7 minutes: graph sync, describe (0 model requests: nothing new to describe), scoring, graph
  audit (all checks pass), 5 packets with NVIDIA, form reading, and batch **B0002** (5 jobs) queued.
- **Packets**: 15 jobs have one so far, every résumé one page. About half needed no repair; a flagged packet had a
  piece replaced by the profile's own words, never an unchecked claim.
- **Forms**: 5 Greenhouse and Lever forms read for real. Per form, 5 to 10 fields are filled from the facts file and
  the CV is attached; salary and CTC, notice period, relocation, consent, demographics, CAPTCHA, conditional and
  unrecognised questions are left for you (2 to 16 per form).
- **Batches**: B0001 (10 jobs) and B0002 (5 jobs) are queued, waiting for you in the app.
- **The filler on the local three-page test form**: all 3 pages, 9 fields filled, 2 left for you, "Submit
  application", "Apply now" and "Finish" refused: **0 submit events, 0 final-button clicks**.
- **The desktop app**: built and launched (dev and built modes), every screen showing real data; closing it stops its
  Python service.

## Tests

- Python: the full suite passes on Postgres (2,446 before this piece, plus 3 for the runner). New in this build: résumé
  core 11, tailoring and claim gate 14, API 6, application filler 12 (several parametrized), runner 3.
- Desktop app: type check clean, 10 tests (security flags and rules, API client, screen registry).
- CI on every PR: lint, the safety guards, tests, tests on Postgres, the app's type check and tests, secret scan.

## Known missing or rough (refinement later)

- **Résumé wording**: the model mostly keeps the profile's own sentences and rarely adds the bold terms the example
  uses; letters sometimes write "100 plus" or "Q and A". Quality work on tailoring is the next big step.
- **Scores and order**: weights are placeholders (most jobs score 2.00), so which jobs get packets first is provisional.
  The matching fix ("PR 6": partial credit from 0.615, full credit by name only) still waits for the owner's yes.
- **Form coverage**: Greenhouse and Lever only. Ashby, Workday, Amazon and the rest are batched with their packets but
  no prepared answers ("not supported yet"); the copy panel and the packet files still work for them.
- **Draft answers**: none of the real forms read so far asked a free-text question the rules mark as draftable, so
  drafting has only been exercised in tests.
- **Filling real sites**: deliberately not done. The filler runs only on local test forms; the in-app browser, the
  network guard and the first real fill come later, with the owner watching.
- **Facts**: the private facts file has no city yet (not in the profile), so location fields are left for you.
- **Packaging**: the app runs from the repo (`npm run dev` / `npm start`); there is no installer yet.

## The owner's review checklist

1. Open `data/packets/<job>/cv.docx` and `cv.pdf` for three jobs: one page, the example's layout, no claim that is
   not in the master profile, important terms bold.
2. Open a cover letter: three paragraphs, nothing invented.
3. Start the app (`cd app`, `npm run dev`): Pipeline, Jobs, Packets, Batches and Apply all show real data;
   Release / Reject works.
4. Open a queued Greenhouse or Lever job on the Apply screen: copy-panel values, and the "only you can answer"
   list (salary, notice period, work authorization and the like must be there, never filled).
5. Run `python -m hiredra.autofill fill --fixture three_page` and watch it stop on the last page without submitting.
6. Read the "Known missing or rough" list above.
7. Only then, if satisfied, say so: `build/e2e` is merged into `main`.
