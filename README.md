# AI peer review: feedback, revision and learning

Exploratory analysis of reflections from 39 students across four writing cycles. This repository contains the public report, aggregate results and analysis code.

[Read the report](https://wwang93.github.io/ai-peer-review-analysis/)

The report is in Chinese; the analysis code and figure labels are in English. The analysis date is 27 September 2026.

## What the analysis supports

The three research questions concern student-reported AI feedback, reported revisions and reported learning. RQ1 and RQ2 use comparable questions across four rounds. The learning question changed in Round 4, so RQ3 uses Rounds 1 to 3 for longitudinal comparison and treats Round 4 as supplementary material.

The models identify candidate patterns for qualitative review. Normalized NMF loadings are not percentages of students, verified code frequencies or measures of writing ability. Independent human coding has not been completed. AI assisted the preparation of code, narrative interpretation and provisional codes.

## Files

| File | Contents |
| --- | --- |
| `index.html`, `report.md`, `site.css` | Published report and source |
| `prepare_data.py` | Auditable cleaning of the original Excel worksheet |
| `model.py` | Pooled NMF models, candidate topic counts, participant bootstrap and sensitivity analyses |
| `plot_results.py` | Five figures generated only from aggregate tables |
| `build_site.py` | Static HTML generation using Python's standard library |
| `validate_public.py` | Public-file checks and numerical reconciliation |
| `run_config.json`, `topic_labels.json` | Recorded analysis settings and provisional labels |
| `topics.csv`, `topic_by_round.csv` | Aggregate topics and round means |
| `paired_changes.csv`, `sensitivity.csv`, `model_candidates.csv` | Paired estimates and model diagnostics |
| `analysis_denominators.csv` | Counts before and after model filtering |
| `provisional_codebook.csv` | Candidate definitions without quotations or participant references |
| `01_*.png` through `05_*.png`, corresponding SVGs | Figures for viewing and reuse |

## Rebuild the public report

Use Python 3.12. Package versions match the recorded analysis environment.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python plot_results.py
python build_site.py
python validate_public.py
```

GitHub Pages serves the root of the `main` branch. The committed `index.html` needs no server or external JavaScript dependencies. Rebuilding figures and the report does not require student text.

## Refit the models with authorized data

Participant responses are not included. Place the authorized workbook outside the repository, then run:

```sh
python prepare_data.py /path/to/authorized-workbook.xlsx
python model.py candidates
python model.py final
```

The workbook must have a `raw` worksheet, a participant ID in the first column and response columns named `reflection1.1` through `reflection4.5`, with four questions in Rounds 1 to 3 and five in Round 4. `prepare_data.py` retains participants with at least one substantive response. The original analysis has 39 such participants.

All response-level outputs, representative texts and model factors are written to the ignored `private_run/` directory. Set `AI_PEER_REVIEW_WORKDIR` to use a different private working directory. Do not publish that directory. The scripts copy the provisional topic labels there on the first run.

Candidate topic labels are specific to this corpus. If the data, topic count or preprocessing changes, read the representative responses and check topic ordering before reusing labels. The final fit runs 40 participant bootstrap refits and 2,000 resamples for fixed-model intervals. Refit bootstraps retain the original feature space; interval estimates do not include all model-selection uncertainty.

The public preparation script follows the original Excel cleaning rules. Source-file inventory, document OCR and cross-source matching were separate audit steps and are not required for the 39-student model. Their aggregate findings are described in the report.

## Data availability

Only aggregate outputs are public. The repository excludes source workbooks, student journals, quotations, participant identifiers, response-level weights, model factors and human-review worksheets. No synthetic data are presented as study observations. The report retains methodological limitations and the provisional status of the qualitative interpretation.

No reuse license has been assigned. Contact the repository owner about permissions or access to restricted research materials.
