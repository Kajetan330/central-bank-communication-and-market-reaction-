v31 missing-fix outputs
=======================
Input folder:  C:/Users/Kajetan/Desktop/final thesis/final model/final outputs
Output folder: C:/Users/Kajetan/Desktop/final thesis/final model/final missing fix outputs

Use 00_checklist_manifest.csv to see which review items are data-covered and which are prose-only.
The most important generated files are:
  03_press_conference_coefficients_for_results.csv
  04_press_conference_coefficients_wide.csv
  01_broad_outcome_coefficients_all_scores.csv
  05_fe_forward_current_all.csv
  09_rf_feature_count_diagnostics.csv
  10_rf_leakage_adjacent_terms.csv
  12_rf_feature_importance_filtered_for_main_text.csv

Important limitation:
  The script can identify residual leakage-adjacent terms in v31, but it cannot prove a before/after blocklist improvement unless the full RF pipeline is re-run with the expanded blocklist.
