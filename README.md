# Central Bank Communication and Market Reactions

This repository is the public artifact archive for the thesis:

**Central Bank Communication and Market Reactions: Text-Based Hawk/Dove Classification and Event-Study Evidence from the FOMC and ECB**

It collects the thesis raw data, model objects, and code used to study whether the tone of central bank communication helps explain financial market reactions and predict near-term policy outcomes.

Authors: Adrian Piotr Pulchny and Kajetan Moravec Cvelbar  
Institution: Copenhagen Business School, MSc in Advanced Economics and Finance  
Artifact set assembled on 2026-05-14

## At A Glance

The project compares three ways of measuring monetary policy tone across FOMC and ECB statements and press conferences:

| Method | Role in the thesis | Main outputs |
| --- | --- | --- |
| Dictionary score | Transparent hawkishness measure built from policy, inflation, employment, and residual-risk language | `net_score`, component scores, dictionary labels |
| Random Forest score | Supervised bigram model trained to recover rate-decision language | `rf_score`, feature importance, confusion matrices |
| RoBERTa/BERT score | Sentence-level neural benchmark aggregated to document level | `bert_score`, BERT labels |

These measures are merged with intraday market surprises from USMPD and EA-MPD, then evaluated in event-study regressions and expanding-window forecast exercises.

## Workflow

```mermaid
flowchart LR
  A[Raw FOMC and ECB text] --> B[Text cleaning and document panels]
  C[USMPD and EA-MPD surprises] --> B
  D[BERT sentence predictions] --> B
  B --> E[Dictionary, Random Forest, and BERT tone scores]
  E --> F[RQ1 market-reaction regressions]
  E --> G[RQ2 expanding-window forecasts]
  F --> H[Model outputs and reproducible artifacts]
  G --> H
```

## Repository Map

```text
code/                 R scripts for the thesis model pipeline, figure generation, and data checks
data/raw/             Raw FOMC, ECB, USMPD, EA-MPD, recovery, and BERT sentence-prediction files
models/               Saved Random Forest model and expanding-window result objects
```

The manuscript text, generated tables, and generated figures are maintained outside this repository. This archive is kept focused on the public raw data, model objects, and code artifacts.

## Key Files

| Artifact | Path |
| --- | --- |
| RQ1 Random Forest object | `models/rf_rq1_models_v31.rds` |
| RQ2 expanding-window object | `models/rf_rq2_xw_results_v31.rds` |
| Main pipeline script | `code/pipeline_v31_final.R` |
| Figure-generation script | `code/generate_v31_figures_full_FIXED_PATHS.R` |

## Raw Data Included

The raw folder from the thesis workspace is included under `data/raw/`, including:

- FOMC statement and press-conference econometric text files
- ECB decision and statement or press-conference files
- USMPD and EA-MPD Excel workbooks
- BERT sentence-level prediction files
- Manual text-recovery file

Only the two zero-byte Excel lock files, `~$EA-MPD.xlsx` and `~$USMPD.xlsx`, were excluded.

The `data/` directory is intentionally raw-only. Processed panels, generated tables, and generated figures are not stored in this repository.

## Reproduction Notes

The final outputs are already stored in this repository. The scripts in `code/` are included for transparency and reruns.

- `code/pipeline_v31_final.R` reads from `data/raw/` and writes rerun outputs to `reproduction/output_v31/`.
- `code/generate_v31_figures_full_FIXED_PATHS.R` regenerates figures from processed outputs produced by the pipeline, then writes regenerated figures to `reproduction/figures_v31/`.
- `code/data_chapter_checks_FIXED_V5.R` reads from `data/raw/` and writes checks to `reproduction/data_check/`.

The full RF/BERT pipeline can take several hours and requires the R packages loaded at the top of the scripts.

## Important Caveat

The uploaded model objects are the thesis artifacts identified in the local working folder. Generated result tables, figures, and manuscript files are intentionally not stored in this repository.

## Citation

If you use this archive, please cite the repository metadata in `CITATION.cff`.

## Licenses

Code is released under the MIT License. See `LICENSE-CODE`.

Data and repository documentation are released under Creative Commons Attribution 4.0 International. See `LICENSE-DATA`.
