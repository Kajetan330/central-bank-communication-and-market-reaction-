# Central Bank Communication Thesis Results

This repository stores the data, model objects, code, tables, and figures for the thesis:

**Central Bank Communication and Market Reactions: Text-Based Hawk/Dove Classification and Event-Study Evidence from the FOMC and ECB**

Authors: Adrian Piotr Pulchny and Kajetan Moravec Cvelbar  
Institution: Copenhagen Business School, MSc in Advanced Economics and Finance  
Final artifact set: v31, assembled on 2026-05-14

## Contents

```text
code/                 R scripts used for the final v31 pipeline and figure/data checks
data/raw/             Raw input data, including USMPD, EA-MPD, source text CSVs, and BERT sentence predictions
data/processed/       Final v31 processed panels and score datasets
data/cache/           Cached ECB HTML recovery object used by the pipeline
models/               Final saved Random Forest model/result objects
results/tables/       Main result tables and diagnostics from the v31 run
results/figures/      Final generated thesis figures
results/figure_data/  CSV data behind generated figures
results/review_checks/Additional review/fix diagnostics generated after the v31 run
docs/                 Selected thesis chapters/appendix material related to data, methods, and results
```

## Main Artifacts

- Main processed dataset: `data/processed/combined_final_v2.csv`
- Final RQ1 Random Forest model object: `models/rf_rq1_models_v31.rds`
- Final RQ2 expanding-window result object: `models/rf_rq2_xw_results_v31.rds`
- Main RF performance table: `results/tables/rf_rq1_performance_v31.csv`
- Main RF feature importance table: `results/tables/rf_feature_importance_v31.csv`
- RQ1 method comparison: `results/tables/rq1_method_comparison_v31.csv`
- RQ2 method comparison: `results/tables/rq2_method_comparison_v31.csv`
- Figure manifest: `results/figure_data/00_figure_manifest.csv`

## Raw Data

The raw data from the original thesis folder is included under `data/raw/`.

Excluded raw-folder files:

- `~$EA-MPD.xlsx`
- `~$USMPD.xlsx`

These were zero-byte Excel lock files, not data.

## Reproduction Notes

The final outputs are already stored in this repository. The copied R scripts in `code/` are included for transparency and reproduction.

The repository copies of the main scripts have been adjusted to look for data inside this repo:

- `code/pipeline_v31_final.R` reads from `data/raw/` and writes rerun outputs to `reproduction/output_v31/`.
- `code/generate_v31_figures_full_FIXED_PATHS.R` reads from `data/processed/` and `results/tables/`, then writes regenerated figures to `reproduction/figures_v31/`.
- `code/data_chapter_checks_FIXED_V5.R` reads from `data/raw/` and writes checks to `reproduction/data_check/`.

The full RF/BERT pipeline can take several hours and requires the R packages loaded at the top of the scripts.

## Important Caveat

The v31 outputs are the final thesis artifacts identified in the local working folder. A later review check identifies several RF leakage-adjacent terms in `results/review_checks/10_rf_leakage_adjacent_terms.csv` and suggests blocklist additions in `results/review_checks/11_suggested_blocklist_additions_from_v31.csv`. Those files are diagnostic. The full RF pipeline was not re-run after that suggested expanded blocklist.

## Licenses

Code is released under the MIT License. See `LICENSE-CODE`.

Data, figures, tables, and documentation are released under Creative Commons Attribution 4.0 International. See `LICENSE-DATA`.
