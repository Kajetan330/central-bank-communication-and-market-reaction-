# =============================================================================
# generate_v31_figures_full.R
# =============================================================================
# Purpose:
#   Generate the updated figure set for the current v31 thesis Results chapter
#   and appendix diagnostics.
#
# Output folder requested by Kajetan:
#   C:/Users/Kajetan/Desktop/final thesis/final model/final model figures
#
# What this script creates:
#   1. PNG figures ready to upload to Overleaf.
#   2. CSV figure-data files for every figure, so all plotted numbers are auditable.
#   3. A manifest describing which figures should go in the main chapter vs appendix.
#
# How to use:
#   1. Put this script in: C:/Users/Kajetan/Desktop/final thesis/final model
#   2. Make sure the v31 CSV files are in the same folder.
#   3. Run in RStudio: source("generate_v31_figures_full.R")
#   4. Upload the PNG files from "final model figures" to Overleaf.
#
# Notes:
#   - The script does NOT rerun the full RF/BERT model. It uses the thesis model outputs.
#   - The old tone-rate overlay requires official policy-rate LEVELS. The v31 files
#     contain bps_change, so this script creates a cumulative policy-rate-index figure
#     for diagnostics. Do not describe it as an official policy-rate level.
# =============================================================================

# ----------------------------- 0. PACKAGES ----------------------------------
required_pkgs <- c(
  "readr", "dplyr", "tidyr", "ggplot2", "stringr", "forcats",
  "purrr", "scales", "tibble", "lmtest", "sandwich"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
  library(purrr)
  library(scales)
  library(tibble)
  library(lmtest)
  library(sandwich)
})

# ----------------------------- 1. PATHS -------------------------------------
# Use forward slashes in R even on Windows.
input_dir <- "C:/Users/Kajetan/Desktop/final thesis/final model"
output_dir <- "C:/Users/Kajetan/Desktop/final thesis/final model/final model figures"
figure_data_dir <- file.path(output_dir, "figure_data")

# Optional fallback: if you run the script from another folder and the input path
# does not exist, use the current working directory as input_dir. Output still goes
# to the requested folder if possible.
if (!dir.exists(input_dir)) {
  warning("input_dir does not exist. Using getwd() instead: ", getwd())
  input_dir <- getwd()
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figure_data_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------- 2. INPUT FILES -------------------------------
files <- list(
  fed_stmt = "fed_stmt_with_bert_v31.csv",
  fed_pc   = "fed_pc_with_bert_v31.csv",
  ecb_stmt = "ecb_stmt_with_bert_v31.csv",
  ecb_pc   = "ecb_pc_with_bert_v31.csv",
  combined = "combined_final_v2.csv",
  rf_imp   = "rf_feature_importance_v31.csv",
  cm_long  = "confusion_matrices_long_v31.csv",
  cm_agree = "confusion_matrix_agreement_v31.csv",
  rf_perf  = "rf_rq1_performance_v31.csv",
  rq1_mc   = "rq1_method_comparison_v31.csv",
  dual_r2  = "r2_dual_mandate_comparison_v31.csv",
  fe       = "fe_ladder_results_v31.csv",
  rf_xw    = "rf_rq2_xw_performance_v31.csv",
  rq2_rf   = "rq2_method_comparison_v31.csv",
  rq2_bert = "rq2_bert_method_comparison_v31.csv"
)

required_paths <- file.path(input_dir, unlist(files))
missing_files <- required_paths[!file.exists(required_paths)]
if (length(missing_files) > 0) {
  stop("Missing required v31 input files:\n", paste(missing_files, collapse = "\n"))
}

read_v31 <- function(key) {
  readr::read_csv(file.path(input_dir, files[[key]]), show_col_types = FALSE)
}

# ----------------------------- 3. HELPERS -----------------------------------
write_fig_data <- function(df, filename) {
  path <- file.path(figure_data_dir, filename)
  readr::write_csv(df, path)
  message("Wrote data: ", path)
  invisible(path)
}

save_fig <- function(p, filename, width = 10, height = 6, dpi = 300) {
  path <- file.path(output_dir, filename)
  ggplot2::ggsave(
    filename = path, plot = p, width = width, height = height,
    dpi = dpi, bg = "white"
  )
  message("Wrote figure: ", path)
  invisible(path)
}

normalise_label <- function(x) {
  x <- tolower(as.character(x))
  dplyr::case_when(
    x %in% c("hawkish", "hike", "h") ~ "hawkish",
    x %in% c("dovish", "cut", "d") ~ "dovish",
    x %in% c("neutral", "hold", "n") ~ "neutral",
    TRUE ~ NA_character_
  )
}

zscore <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(NA_real_, length(x)))
  as.numeric((x - mean(x, na.rm = TRUE)) / s)
}

rolling_mean <- function(x, k = 12) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    lo <- max(1, i - k + 1)
    vals <- x[lo:i]
    if (all(is.na(vals))) {
      out[i] <- NA_real_
    } else {
      out[i] <- mean(vals, na.rm = TRUE)
    }
  }
  out
}

clean_corpus_name <- function(x) {
  dplyr::recode(
    x,
    "FOMC stmt" = "FOMC statements",
    "FOMC STMT" = "FOMC statements",
    "fomc_stmt" = "FOMC statements",
    "FOMC PC" = "FOMC press conferences",
    "fomc_pc" = "FOMC press conferences",
    "ECB stmt" = "ECB statements",
    "ECB STMT" = "ECB statements",
    "ecb_stmt" = "ECB statements",
    "ECB PC" = "ECB press conferences",
    "ecb_pc" = "ECB press conferences",
    .default = as.character(x)
  )
}

method_label <- function(x) {
  dplyr::recode(
    x,
    "net_score" = "Dictionary",
    "dict" = "Dictionary",
    "rf_score" = "Random Forest",
    "RF" = "Random Forest",
    "bert_score" = "RoBERTa/BERT",
    "BERT" = "RoBERTa/BERT",
    .default = as.character(x)
  )
}

target_label <- function(x) {
  dplyr::recode(
    x,
    "bps_next" = "Rate t+1",
    "bps_next2" = "Rate t+2",
    "y2_next" = "2Y yield t+1",
    "y2_next2" = "2Y yield t+2",
    .default = as.character(x)
  )
}

fit_robust_term <- function(data, outcome, regressors, term, extra_cols = tibble()) {
  vars <- c(outcome, regressors)
  vars <- vars[vars %in% names(data)]
  if (!(outcome %in% vars) || !(term %in% vars)) return(NULL)

  d <- data[, vars, drop = FALSE]
  d <- d[stats::complete.cases(d), , drop = FALSE]
  if (nrow(d) < 20) return(NULL)

  f <- stats::reformulate(regressors, response = outcome)
  m <- tryCatch(stats::lm(f, data = d), error = function(e) NULL)
  if (is.null(m)) return(NULL)

  ct <- tryCatch(lmtest::coeftest(m, vcov. = sandwich::vcovHC(m, type = "HC1")), error = function(e) NULL)
  if (is.null(ct) || !(term %in% rownames(ct))) return(NULL)

  tibble(
    term = term,
    estimate = unname(ct[term, 1]),
    se = unname(ct[term, 2]),
    pvalue = unname(ct[term, 4]),
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se,
    n = stats::nobs(m),
    r2 = summary(m)$r.squared
  ) %>% bind_cols(extra_cols)
}

compute_direct_oos <- function(data, actual, pred, method, corpus_long) {
  if (!(actual %in% names(data)) || !(pred %in% names(data))) return(NULL)
  d <- data %>% select(all_of(c(actual, pred))) %>% filter(stats::complete.cases(.))
  if (nrow(d) < 20) return(NULL)
  y <- d[[actual]]
  yhat <- d[[pred]]
  denom <- sum((y - mean(y, na.rm = TRUE))^2, na.rm = TRUE)
  if (denom == 0 || is.na(denom)) return(NULL)
  tibble(
    method = method,
    corpus_long = corpus_long,
    target = actual,
    target_clean = target_label(actual),
    n = nrow(d),
    oos_r2 = 1 - sum((y - yhat)^2, na.rm = TRUE) / denom,
    rmse = sqrt(mean((y - yhat)^2, na.rm = TRUE)),
    naive_rmse = sqrt(mean((y - mean(y, na.rm = TRUE))^2, na.rm = TRUE))
  )
}

compute_incremental_r2 <- function(data, actual, pred, control, method, corpus_long) {
  needed <- c(actual, pred, control)
  if (!all(needed %in% names(data))) return(NULL)
  d <- data %>% select(all_of(needed)) %>% filter(stats::complete.cases(.))
  if (nrow(d) < 20) return(NULL)

  f_base <- stats::reformulate(control, response = actual)
  f_full <- stats::reformulate(c(pred, control), response = actual)
  m_base <- tryCatch(stats::lm(f_base, data = d), error = function(e) NULL)
  m_full <- tryCatch(stats::lm(f_full, data = d), error = function(e) NULL)
  if (is.null(m_base) || is.null(m_full)) return(NULL)

  ct <- tryCatch(lmtest::coeftest(m_full, vcov. = sandwich::vcovHC(m_full, type = "HC1")), error = function(e) NULL)
  pred_est <- pred_se <- pred_p <- NA_real_
  if (!is.null(ct) && pred %in% rownames(ct)) {
    pred_est <- unname(ct[pred, 1])
    pred_se  <- unname(ct[pred, 2])
    pred_p   <- unname(ct[pred, 4])
  }

  tibble(
    method = method,
    corpus_long = corpus_long,
    target = actual,
    target_clean = target_label(actual),
    pred_col = pred,
    control_col = control,
    n = nrow(d),
    r2_base = summary(m_base)$r.squared,
    r2_full = summary(m_full)$r.squared,
    incremental_r2 = r2_full - r2_base,
    pred_estimate = pred_est,
    pred_se = pred_se,
    pred_pvalue = pred_p
  )
}

# ----------------------------- 4. STYLE -------------------------------------
corpus_levels <- c(
  "FOMC statements", "FOMC press conferences",
  "ECB statements", "ECB press conferences"
)
method_levels <- c("Dictionary", "Random Forest", "RoBERTa/BERT", "All")
label_levels <- c("dovish", "neutral", "hawkish")
target_levels <- c("Rate t+1", "Rate t+2", "2Y yield t+1", "2Y yield t+2")

pal_method <- c(
  "Dictionary" = "#2C3E50",
  "Random Forest" = "#1F77B4",
  "RoBERTa/BERT" = "#C0392B",
  "All" = "#7F8C8D"
)
pal_label <- c(
  "dovish" = "#3B6FB6",
  "neutral" = "#9E9E9E",
  "hawkish" = "#B13A3A"
)

thesis_theme <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      axis.title = element_text(face = "bold")
    )
}

# ----------------------------- 5. READ DATA ---------------------------------
fed_stmt <- read_v31("fed_stmt") %>%
  mutate(corpus_key = "fomc_stmt", corpus_long = "FOMC statements", bank = "FOMC", document_type = "statement")
fed_pc <- read_v31("fed_pc") %>%
  mutate(corpus_key = "fomc_pc", corpus_long = "FOMC press conferences", bank = "FOMC", document_type = "press_conference")
ecb_stmt <- read_v31("ecb_stmt") %>%
  mutate(corpus_key = "ecb_stmt", corpus_long = "ECB statements", bank = "ECB", document_type = "statement")
ecb_pc <- read_v31("ecb_pc") %>%
  mutate(corpus_key = "ecb_pc", corpus_long = "ECB press conferences", bank = "ECB", document_type = "press_conference")

all_docs <- bind_rows(fed_stmt, fed_pc, ecb_stmt, ecb_pc) %>%
  mutate(
    meeting_date = as.Date(meeting_date),
    corpus_long = factor(corpus_long, levels = corpus_levels),
    rate_label = case_when(
      is.na(bps_change) ~ NA_character_,
      bps_change > 0 ~ "hawkish",
      bps_change < 0 ~ "dovish",
      TRUE ~ "neutral"
    ),
    dict_label = normalise_label(label_text),
    rf_label_norm = normalise_label(rf_label),
    bert_label_norm = normalise_label(bert_label)
  )

statements <- all_docs %>% filter(document_type == "statement")

rf_imp   <- read_v31("rf_imp")
cm_long  <- read_v31("cm_long")
cm_agree <- read_v31("cm_agree")
rf_perf  <- read_v31("rf_perf")
rq1_mc   <- read_v31("rq1_mc")
dual_r2  <- read_v31("dual_r2")
fe_tbl   <- read_v31("fe")
rf_xw    <- read_v31("rf_xw")
rq2_rf   <- read_v31("rq2_rf")
rq2_bert <- read_v31("rq2_bert")

# =============================================================================
# MAIN RESULTS FIGURES
# =============================================================================

# ----------------------------- FIG 6.1 ---------------------------------------
# Score distributions by method and corpus. Replaces old figures 1, 9 and 13.
fig6_1_data <- all_docs %>%
  select(corpus_key, corpus_long, bank, document_type, meeting_date,
         net_score, rf_score, bert_score) %>%
  pivot_longer(
    cols = c(net_score, rf_score, bert_score),
    names_to = "score_name", values_to = "score"
  ) %>%
  filter(!is.na(score)) %>%
  mutate(
    method = factor(method_label(score_name), levels = method_levels),
    corpus_long = factor(corpus_long, levels = corpus_levels)
  ) %>%
  group_by(corpus_long, method) %>%
  mutate(score_z = zscore(score)) %>%
  ungroup()

fig6_1_summary <- fig6_1_data %>%
  group_by(corpus_long, method) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    min = min(score, na.rm = TRUE),
    p10 = quantile(score, 0.10, na.rm = TRUE),
    median = median(score, na.rm = TRUE),
    p90 = quantile(score, 0.90, na.rm = TRUE),
    max = max(score, na.rm = TRUE),
    .groups = "drop"
  )

write_fig_data(fig6_1_data, "fig6_1_score_distribution_points.csv")
write_fig_data(fig6_1_summary, "fig6_1_score_distribution_summary.csv")

p_fig6_1 <- ggplot(fig6_1_data, aes(x = score, fill = method)) +
  geom_histogram(aes(y = after_stat(density)), bins = 28, alpha = 0.45, colour = "white", linewidth = 0.15) +
  geom_density(aes(colour = method), linewidth = 0.7, alpha = 0.8) +
  geom_vline(xintercept = 0, linewidth = 0.25, linetype = "dashed") +
  facet_grid(method ~ corpus_long) +
  scale_fill_manual(values = pal_method, drop = FALSE) +
  scale_colour_manual(values = pal_method, drop = FALSE) +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(
    title = "Distribution of document-level tone scores",
    subtitle = "Dictionary, Random Forest and RoBERTa/BERT scores by corpus",
    x = "Score (positive = more hawkish)", y = "Density", fill = NULL, colour = NULL
  ) +
  thesis_theme(9)

save_fig(p_fig6_1, "fig6_1_score_distributions.png", width = 12, height = 7.5)

# Also produce old-name dictionary-only and RF-only diagnostic figures.
fig_dict_only <- fig6_1_data %>% filter(method == "Dictionary")
p_old_dict <- ggplot(fig_dict_only, aes(x = score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 28, fill = "#2C3E50", alpha = 0.55, colour = "white") +
  geom_density(colour = "#2C3E50", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.25) +
  facet_wrap(~ corpus_long, ncol = 2) +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(title = "Dictionary net score distributions", x = "net_score", y = "Density") +
  thesis_theme(10)
save_fig(p_old_dict, "net_score_distribution_normal.png", width = 9, height = 6)

fig_rf_only <- fig6_1_data %>% filter(method == "Random Forest")
p_old_rf <- ggplot(fig_rf_only, aes(x = score)) +
  geom_histogram(aes(y = after_stat(density)), bins = 28, fill = "#1F77B4", alpha = 0.55, colour = "white") +
  geom_density(colour = "#1F77B4", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.25) +
  facet_wrap(~ corpus_long, ncol = 2) +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(title = "Random Forest score distributions", x = "rf_score", y = "Density") +
  thesis_theme(10)
save_fig(p_old_rf, "rf_score_distribution_normal.png", width = 9, height = 6)

p_overlay <- ggplot(fig6_1_data, aes(x = score_z, colour = method)) +
  geom_density(linewidth = 0.85, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 2) +
  scale_colour_manual(values = pal_method, drop = FALSE) +
  labs(
    title = "Z-standardised score distribution overlay",
    x = "Within-corpus z-score", y = "Density", colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_overlay, "comparison_distribution_overlay.png", width = 9.5, height = 6)

# ----------------------------- FIG 6.2 ---------------------------------------
# Agreement with rate decisions by method.
fig6_2_data <- cm_agree %>%
  filter(label_b_name == "rate", label_a_name %in% c("dict", "RF", "BERT")) %>%
  mutate(
    corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels),
    method = factor(method_label(label_a_name), levels = method_levels),
    agreement_pct = 100 * agreement
  )

write_fig_data(fig6_2_data, "fig6_2_agreement_with_rate.csv")

p_fig6_2 <- ggplot(fig6_2_data, aes(x = corpus_long, y = agreement_pct, fill = method)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_text(
    aes(label = sprintf("%.1f", agreement_pct)),
    position = position_dodge(width = 0.75), vjust = -0.35, size = 3
  ) +
  scale_fill_manual(values = pal_method, drop = FALSE) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 105), expand = expansion(mult = c(0, 0.03))) +
  labs(
    title = "Agreement with realised rate-decision sign",
    subtitle = "Share of documents where each text label matches hike / hold / cut",
    x = NULL, y = "Agreement rate", fill = NULL
  ) +
  thesis_theme(10) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

save_fig(p_fig6_2, "fig6_2_agreement_with_rate.png", width = 10, height = 6)
save_fig(p_fig6_2, "comparison_agreement_barplot.png", width = 10, height = 6)

# ----------------------------- FIG 6.3 ---------------------------------------
# RF confusion matrices against realised rate decision.
fig6_3_data <- cm_long %>%
  filter(label_a_name == "RF", label_b_name == "rate") %>%
  mutate(
    corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels),
    predicted = factor(normalise_label(label_a), levels = label_levels),
    actual = factor(normalise_label(label_b), levels = label_levels)
  ) %>%
  group_by(corpus_long, actual) %>%
  mutate(pct_of_actual = n / sum(n, na.rm = TRUE)) %>%
  ungroup()

write_fig_data(fig6_3_data, "fig6_3_rf_confusion_matrix.csv")

p_fig6_3 <- ggplot(fig6_3_data, aes(x = predicted, y = actual, fill = n)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(n, "\n", sprintf("%.0f%%", 100 * pct_of_actual))), size = 3.2) +
  facet_wrap(~ corpus_long, ncol = 2) +
  scale_fill_gradient(low = "#F5F7FA", high = "#1F4E79") +
  labs(
    title = "Random Forest confusion matrices",
    subtitle = "Rows are realised rate decisions; columns are RF predictions. Cell labels show count and row percentage.",
    x = "RF predicted label", y = "Actual rate-decision label", fill = "Count"
  ) +
  thesis_theme(10)

save_fig(p_fig6_3, "fig6_3_rf_confusion_matrices.png", width = 9, height = 7)
save_fig(p_fig6_3, "rf_confusion_heatmap.png", width = 9, height = 7)

# ----------------------------- FIG 6.4 ---------------------------------------
# RF signed feature importance.
fig6_4_data <- rf_imp %>%
  mutate(
    corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels),
    direction_label = if_else(direction >= 0, "hawkish", "dovish"),
    signed_importance = MeanDecreaseGini * if_else(direction >= 0, 1, -1),
    abs_signed_importance = abs(signed_importance),
    term_original = if_else(is.na(term_original) | term_original == "", term_stem, term_original)
  ) %>%
  group_by(corpus_long) %>%
  slice_max(order_by = abs_signed_importance, n = 12, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    term_plot = paste(term_original, corpus_long, sep = "___"),
    term_plot = forcats::fct_reorder(term_plot, signed_importance)
  )

write_fig_data(fig6_4_data, "fig6_4_rf_signed_feature_importance.csv")

p_fig6_4 <- ggplot(fig6_4_data, aes(x = signed_importance, y = term_plot, fill = direction_label)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.35) +
  facet_wrap(~ corpus_long, scales = "free_y", ncol = 2) +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  scale_fill_manual(values = pal_label, drop = FALSE) +
  labs(
    title = "Random Forest signed feature importance",
    subtitle = "Top bigrams by Gini importance, signed by hike-minus-cut direction",
    x = "Signed importance", y = NULL, fill = NULL
  ) +
  thesis_theme(10)

save_fig(p_fig6_4, "fig6_4_rf_feature_importance.png", width = 11, height = 8)
save_fig(p_fig6_4, "rf_feature_importance_bars.png", width = 11, height = 8)

# ----------------------------- FIG 6.5 ---------------------------------------
# RQ1 method-comparison R2.
fig6_5_data <- rq1_mc %>%
  mutate(corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels)) %>%
  pivot_longer(
    cols = c(r2_dict, r2_rf, r2_bert, r2_all),
    names_to = "method_raw", values_to = "r2"
  ) %>%
  mutate(
    method = recode(
      method_raw,
      "r2_dict" = "Dictionary",
      "r2_rf" = "Random Forest",
      "r2_bert" = "RoBERTa/BERT",
      "r2_all" = "All"
    ),
    method = factor(method, levels = method_levels),
    outcome = str_replace_all(outcome, "_", " ")
  )

write_fig_data(fig6_5_data, "fig6_5_rq1_method_r2.csv")

p_fig6_5 <- ggplot(fig6_5_data, aes(x = method, y = outcome, fill = r2)) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_text(aes(label = sprintf("%.3f", r2)), size = 3) +
  facet_wrap(~ corpus_long, scales = "free_y", ncol = 1) +
  scale_fill_gradient(low = "#F7FBFF", high = "#1F4E79") +
  labs(
    title = "RQ1 explanatory power by text-scoring method",
    subtitle = "Controlled market-reaction regressions: single-score models and three-way horse race",
    x = NULL, y = NULL, fill = expression(R^2)
  ) +
  thesis_theme(10)

save_fig(p_fig6_5, "fig6_5_rq1_method_r2_heatmap.png", width = 8.5, height = 8)

# ----------------------------- FIG 6.6 ---------------------------------------
# Dual-mandate R2 levels and gains.
fig6_6_levels <- dual_r2 %>%
  mutate(corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels)) %>%
  pivot_longer(
    cols = c(r2_A_net, r2_B_pol_ec, r2_C_full),
    names_to = "spec_raw", values_to = "r2"
  ) %>%
  mutate(
    spec = recode(
      spec_raw,
      "r2_A_net" = "A: net score",
      "r2_B_pol_ec" = "B: policy + econ",
      "r2_C_full" = "C: full decomposition"
    ),
    spec = factor(spec, levels = c("A: net score", "B: policy + econ", "C: full decomposition")),
    outcome = str_replace_all(outcome, "_", " ")
  )

fig6_6_gains <- dual_r2 %>%
  mutate(
    corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels),
    outcome = str_replace_all(outcome, "_", " "),
    delta_pct_points = 100 * delta_C_minus_A
  )

write_fig_data(fig6_6_levels, "fig6_6_dual_mandate_r2_levels.csv")
write_fig_data(fig6_6_gains, "fig6_6_dual_mandate_r2_gains.csv")

p_fig6_6 <- ggplot(fig6_6_gains, aes(x = reorder(outcome, delta_C_minus_A), y = delta_C_minus_A, fill = corpus_long)) +
  geom_col(width = 0.72) +
  coord_flip() +
  facet_wrap(~ corpus_long, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("FOMC statements" = "#2C3E50", "ECB statements" = "#1F77B4"), drop = FALSE) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.001)) +
  labs(
    title = "R² gain from decomposing dictionary tone",
    subtitle = "Spec C full decomposition relative to Spec A aggregate net score",
    x = NULL, y = expression(Delta~R^2), fill = NULL
  ) +
  thesis_theme(10) + theme(legend.position = "none")

save_fig(p_fig6_6, "fig6_6_dual_mandate_r2_gains.png", width = 8.5, height = 6)

p_dm_levels <- ggplot(fig6_6_levels, aes(x = spec, y = r2, fill = spec)) +
  geom_col(width = 0.7) +
  facet_grid(corpus_long ~ outcome, scales = "free_y") +
  labs(title = "Dual-mandate decomposition R² levels", x = NULL, y = expression(R^2), fill = NULL) +
  thesis_theme(8) + theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_fig(p_dm_levels, "dm_decomposition_r2_levels.png", width = 12, height = 6.5)

# ----------------------------- FIG 6.7 ---------------------------------------
# RQ2 direct expanding-window OOS R2.
targets <- c("bps_next", "bps_next2", "y2_next", "y2_next2")
corpus_objects <- list(
  "FOMC statements" = fed_stmt,
  "FOMC press conferences" = fed_pc,
  "ECB statements" = ecb_stmt,
  "ECB press conferences" = ecb_pc
)

fig6_7_rf <- rf_xw %>%
  mutate(
    method = "Random Forest",
    corpus_long = factor(clean_corpus_name(corpus), levels = corpus_levels),
    target_clean = factor(target_label(target), levels = target_levels),
    oos_r2 = xw_r2,
    n = n_forecasts,
    rmse = xw_rmse
  ) %>%
  select(method, corpus_long, target, target_clean, n, oos_r2, rmse, naive_rmse)

fig6_7_bert <- purrr::imap_dfr(corpus_objects, function(dat, corpus_nm) {
  purrr::map_dfr(targets, function(tgt) {
    compute_direct_oos(
      data = dat,
      actual = tgt,
      pred = paste0("bert_pred_", tgt, "_full"),
      method = "RoBERTa/BERT",
      corpus_long = corpus_nm
    )
  })
}) %>%
  mutate(
    corpus_long = factor(corpus_long, levels = corpus_levels),
    target_clean = factor(target_clean, levels = target_levels)
  )

fig6_7_data <- bind_rows(fig6_7_rf, fig6_7_bert) %>%
  mutate(
    method = factor(method, levels = method_levels),
    corpus_long = factor(corpus_long, levels = corpus_levels),
    target_clean = factor(target_clean, levels = target_levels)
  )

write_fig_data(fig6_7_data, "fig6_7_rq2_direct_oos_r2.csv")

p_fig6_7 <- ggplot(fig6_7_data, aes(x = target_clean, y = oos_r2, fill = method)) +
  geom_hline(yintercept = 0, linewidth = 0.35, colour = "grey40") +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  facet_wrap(~ corpus_long, ncol = 2) +
  scale_fill_manual(values = pal_method, drop = FALSE) +
  labs(
    title = "RQ2 direct expanding-window forecast performance",
    subtitle = "Positive values outperform the sample-mean benchmark; negative values underperform it",
    x = NULL, y = "Direct out-of-sample R²", fill = NULL
  ) +
  thesis_theme(10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

save_fig(p_fig6_7, "fig6_7_rq2_direct_oos_r2.png", width = 10.5, height = 7)

# ----------------------------- FIG 6.8 ---------------------------------------
# RQ2 incremental R2 beyond inertia.
fig6_8_data <- purrr::imap_dfr(corpus_objects, function(dat, corpus_nm) {
  purrr::map_dfr(targets, function(tgt) {
    control <- if (str_starts(tgt, "bps")) "bps_curr" else "y2_curr"
    bind_rows(
      compute_incremental_r2(dat, tgt, paste0("rf_pred_", tgt, "_full"), control, "Random Forest", corpus_nm),
      compute_incremental_r2(dat, tgt, paste0("bert_pred_", tgt, "_full"), control, "RoBERTa/BERT", corpus_nm)
    )
  })
}) %>%
  mutate(
    method = factor(method, levels = method_levels),
    corpus_long = factor(corpus_long, levels = corpus_levels),
    target_clean = factor(target_clean, levels = target_levels)
  )

write_fig_data(fig6_8_data, "fig6_8_rq2_incremental_r2_beyond_inertia.csv")

p_fig6_8 <- ggplot(fig6_8_data, aes(x = target_clean, y = incremental_r2, fill = method)) +
  geom_hline(yintercept = 0, linewidth = 0.35, colour = "grey40") +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  facet_wrap(~ corpus_long, ncol = 2) +
  scale_fill_manual(values = pal_method, drop = FALSE) +
  labs(
    title = "RQ2 incremental explanatory power beyond inertia",
    subtitle = "Increase in R² when the text forecast is added to the inertia-only baseline",
    x = NULL, y = expression(Delta~R^2), fill = NULL
  ) +
  thesis_theme(10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

save_fig(p_fig6_8, "fig6_8_rq2_incremental_r2.png", width = 10.5, height = 7)

# =============================================================================
# APPENDIX / OLD-FIGURE DIAGNOSTICS
# =============================================================================

# ----------------------------- OLD FIG 2 -------------------------------------
fig02_data <- statements %>%
  arrange(corpus_long, meeting_date) %>%
  select(corpus_key, corpus_long, bank, meeting_date, net_score, bps_change, rate_label) %>%
  mutate(rate_label = factor(rate_label, levels = label_levels))
write_fig_data(fig02_data, "app_fig02_net_score_timeseries_statements.csv")

p_fig02 <- ggplot(fig02_data, aes(x = meeting_date, y = net_score)) +
  geom_hline(yintercept = 0, linewidth = 0.25, linetype = "dashed") +
  geom_point(aes(colour = rate_label), alpha = 0.65, size = 1.4, na.rm = TRUE) +
  geom_smooth(method = "loess", se = FALSE, colour = "black", linewidth = 0.8, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 1, scales = "free_x") +
  scale_colour_manual(values = pal_label, drop = FALSE) +
  labs(
    title = "Dictionary net score over time: statements",
    x = NULL, y = "net_score", colour = "Rate decision"
  ) +
  thesis_theme(10)
save_fig(p_fig02, "net_score_timeseries_statements.png", width = 10, height = 7)

# ----------------------------- OLD FIG 3 -------------------------------------
# Cumulative bps-change index, not official rate level.
fig03_data <- statements %>%
  arrange(corpus_long, meeting_date) %>%
  group_by(corpus_long) %>%
  mutate(
    bps_change_zero = replace_na(bps_change, 0),
    policy_rate_index_bps = cumsum(bps_change_zero),
    z_net_score = zscore(net_score),
    z_policy_rate_index = zscore(policy_rate_index_bps)
  ) %>%
  ungroup() %>%
  select(corpus_long, bank, meeting_date, net_score, bps_change, policy_rate_index_bps,
         z_net_score, z_policy_rate_index) %>%
  pivot_longer(
    cols = c(z_net_score, z_policy_rate_index),
    names_to = "series", values_to = "z_value"
  ) %>%
  mutate(series = recode(series, "z_net_score" = "Dictionary net score", "z_policy_rate_index" = "Cumulative rate-change index"))
write_fig_data(fig03_data, "app_fig03_tone_rate_overlay_index.csv")

p_fig03 <- ggplot(fig03_data, aes(x = meeting_date, y = z_value, colour = series)) +
  geom_line(alpha = 0.35, linewidth = 0.4, na.rm = TRUE) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.8, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 1, scales = "free_x") +
  labs(
    title = "Dictionary tone and cumulative rate-change index",
    subtitle = "Both series are z-standardised within corpus; the rate series is not an official policy-rate level",
    x = NULL, y = "Within-corpus z-score", colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_fig03, "tone_rate_overlay.png", width = 10, height = 7)

# ----------------------------- OLD FIG 4 -------------------------------------
# Maturity response profile: coefficient on net_score with rate-surprise control.
maturity_specs <- tribble(
  ~corpus_nm, ~outcome, ~maturity, ~control,
  "FOMC statements", "UST2Y", 2, "MP1",
  "FOMC statements", "UST5Y", 5, "MP1",
  "FOMC statements", "UST10Y", 10, "MP1",
  "FOMC statements", "UST30Y", 30, "MP1",
  "FOMC press conferences", "UST2Y", 2, "MP1",
  "FOMC press conferences", "UST5Y", 5, "MP1",
  "FOMC press conferences", "UST10Y", 10, "MP1",
  "FOMC press conferences", "UST30Y", 30, "MP1",
  "ECB statements", "OIS_5Y", 5, "OIS_MP1",
  "ECB statements", "OIS_10Y", 10, "OIS_MP1",
  "ECB press conferences", "OIS_5Y", 5, "OIS_MP1",
  "ECB press conferences", "OIS_10Y", 10, "OIS_MP1"
)

fig04_data <- pmap_dfr(maturity_specs, function(corpus_nm, outcome, maturity, control) {
  dat <- corpus_objects[[corpus_nm]]
  fit_robust_term(
    data = dat,
    outcome = outcome,
    regressors = c("net_score", control),
    term = "net_score",
    extra_cols = tibble(corpus_long = corpus_nm, outcome = outcome, maturity = maturity, control = control)
  )
}) %>%
  mutate(corpus_long = factor(corpus_long, levels = corpus_levels))
write_fig_data(fig04_data, "app_fig04_maturity_response_profile.csv")

p_fig04 <- ggplot(fig04_data, aes(x = maturity, y = estimate, colour = corpus_long, group = corpus_long)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.35, colour = "grey40") +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.25, linewidth = 0.6) +
  geom_point(size = 2.2) +
  geom_line(linewidth = 0.6, alpha = 0.75) +
  facet_wrap(~ corpus_long, ncol = 2, scales = "free_y") +
  scale_x_continuous(breaks = c(2, 5, 10, 30)) +
  labs(
    title = "Maturity response profile",
    subtitle = "Coefficient on dictionary net_score from controlled yield regressions",
    x = "Yield maturity", y = expression(beta[net_score]), colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_fig04, "maturity_response_profile.png", width = 9.5, height = 6.5)

# ----------------------------- OLD FIG 5 -------------------------------------
fig05_base <- fe_tbl %>%
  filter(score_term %in% c("net_score", "rf_score", "bert_score")) %>%
  mutate(
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se,
    score_method = factor(method_label(score_term), levels = method_levels),
    plot_label = str_replace_all(label, " -> ", "\n")
  )

fig05_selected <- fig05_base %>%
  filter(fe_spec == "A") %>%
  arrange(pvalue) %>%
  slice_head(n = 8) %>%
  pull(label) %>%
  unique()

fig05_data <- fig05_base %>%
  filter(label %in% fig05_selected) %>%
  mutate(fe_spec = factor(fe_spec, levels = c("A", "B", "C"), labels = c("Base", "+ chair FE", "+ chair + era FE")))
write_fig_data(fig05_data, "app_fig05_coef_stability_fe_ladder.csv")

p_fig05 <- ggplot(fig05_data, aes(x = fe_spec, y = estimate, ymin = ci_low, ymax = ci_high, colour = score_method)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3, colour = "grey40") +
  geom_pointrange(position = position_dodge(width = 0.35), linewidth = 0.55) +
  facet_wrap(~ plot_label, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = pal_method, drop = FALSE) +
  labs(
    title = "Coefficient stability across fixed-effect specifications",
    subtitle = "Eight lowest-p-value base specifications; whiskers are 95% intervals",
    x = NULL, y = "Coefficient estimate", colour = NULL
  ) +
  thesis_theme(9) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_fig(p_fig05, "coef_stability_fe_ladder.png", width = 11, height = 8)

# ----------------------------- OLD FIG 6 -------------------------------------
fig06_data <- all_docs %>%
  select(corpus_long, meeting_date, net_score_fwd, net_score_curr, rate_label) %>%
  filter(!is.na(net_score_fwd), !is.na(net_score_curr)) %>%
  mutate(rate_label = factor(rate_label, levels = label_levels))
write_fig_data(fig06_data, "app_fig06_fwbw_score_scatter.csv")

p_fig06 <- ggplot(fig06_data, aes(x = net_score_fwd, y = net_score_curr, colour = rate_label)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.25) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.25) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.25, colour = "grey50") +
  geom_point(alpha = 0.65, size = 1.5, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 2) +
  scale_colour_manual(values = pal_label, drop = FALSE) +
  coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
  labs(
    title = "Forward-looking versus current-condition dictionary tone",
    x = "Forward-looking net_score", y = "Current-condition net_score", colour = "Rate decision"
  ) +
  thesis_theme(10)
save_fig(p_fig06, "fwbw_score_scatter.png", width = 9.5, height = 7)

# ----------------------------- OLD FIG 11 ------------------------------------
rolling_data <- all_docs %>%
  arrange(corpus_long, meeting_date) %>%
  select(corpus_long, meeting_date, rate_label, dict_label, rf_label_norm, bert_label_norm) %>%
  pivot_longer(cols = c(dict_label, rf_label_norm, bert_label_norm), names_to = "method_raw", values_to = "pred_label") %>%
  mutate(
    method = recode(method_raw, "dict_label" = "Dictionary", "rf_label_norm" = "Random Forest", "bert_label_norm" = "RoBERTa/BERT"),
    method = factor(method, levels = method_levels),
    agreement = as.numeric(!is.na(rate_label) & !is.na(pred_label) & rate_label == pred_label)
  ) %>%
  group_by(corpus_long, method) %>%
  arrange(meeting_date, .by_group = TRUE) %>%
  mutate(rolling_agreement_12 = rolling_mean(agreement, k = 12)) %>%
  ungroup()
write_fig_data(rolling_data, "app_fig11_rolling_agreement_rate.csv")

p_fig11 <- ggplot(rolling_data, aes(x = meeting_date, y = rolling_agreement_12, colour = method)) +
  geom_line(linewidth = 0.7, alpha = 0.85, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 1, scales = "free_x") +
  scale_colour_manual(values = pal_method, drop = FALSE) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Rolling agreement with rate decisions",
    subtitle = "Twelve-meeting rolling average",
    x = NULL, y = "Agreement rate", colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_fig11, "rolling_agreement_rate.png", width = 10, height = 7)

# ----------------------------- OLD FIG 12 ------------------------------------
fig12_data <- all_docs %>%
  select(corpus_long, meeting_date, rate_label, net_score, rf_score, bert_score) %>%
  pivot_longer(cols = c(rf_score, bert_score), names_to = "comparison_score", values_to = "other_score") %>%
  mutate(
    comparison = recode(comparison_score, "rf_score" = "Dictionary vs Random Forest", "bert_score" = "Dictionary vs RoBERTa/BERT"),
    rate_label = factor(rate_label, levels = label_levels)
  ) %>%
  filter(!is.na(net_score), !is.na(other_score))
write_fig_data(fig12_data, "app_fig12_comparison_score_scatter.csv")

p_fig12 <- ggplot(fig12_data, aes(x = net_score, y = other_score, colour = rate_label)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.25) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.25) +
  geom_point(alpha = 0.60, size = 1.45, na.rm = TRUE) +
  facet_grid(comparison ~ corpus_long) +
  scale_colour_manual(values = pal_label, drop = FALSE) +
  coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
  labs(
    title = "Cross-method score scatter",
    x = "Dictionary net_score", y = "Alternative score", colour = "Rate decision"
  ) +
  thesis_theme(8)
save_fig(p_fig12, "comparison_score_scatter.png", width = 12, height = 7)

# ----------------------------- OLD FIG 14 ------------------------------------
fig14_data <- statements %>%
  select(corpus_long, bank, meeting_date, policy_score, econ_score, net_score) %>%
  pivot_longer(cols = c(policy_score, econ_score, net_score), names_to = "score_component", values_to = "score") %>%
  mutate(
    score_component = recode(score_component, "policy_score" = "Policy score", "econ_score" = "Economic score", "net_score" = "Aggregate net score")
  )
write_fig_data(fig14_data, "app_fig14_dm_decomposition_timeseries.csv")

p_fig14 <- ggplot(fig14_data, aes(x = meeting_date, y = score, colour = score_component)) +
  geom_line(alpha = 0.22, linewidth = 0.35, na.rm = TRUE) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.85, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 1, scales = "free_x") +
  coord_cartesian(ylim = c(-1, 1)) +
  labs(
    title = "Dictionary decomposition over time",
    subtitle = "Policy, economic and aggregate net scores for statement corpora",
    x = NULL, y = "Score", colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_fig14, "dm_decomposition_timeseries.png", width = 10, height = 7)

# ----------------------------- OLD FIG 15 ------------------------------------
# Reuse the FE-ladder data as a redesigned robustness grid.
p_fig15 <- p_fig05 + labs(
  title = "Robustness specification grid",
  subtitle = "Redesigned from old dictionary-only grid: current v31 strongest score/outcome pairs"
)
save_fig(p_fig15, "robustness_specification_grid.png", width = 11, height = 8)
write_fig_data(fig05_data, "app_fig15_robustness_specification_grid.csv")

# ----------------------------- OLD FIG 16 ------------------------------------
fig16_data <- statements %>%
  select(corpus_long, bank, meeting_date, net_score, rf_score, bert_score) %>%
  pivot_longer(cols = c(net_score, rf_score, bert_score), names_to = "score_name", values_to = "score") %>%
  filter(!is.na(score)) %>%
  mutate(method = factor(method_label(score_name), levels = method_levels)) %>%
  group_by(corpus_long, method) %>%
  mutate(score_z = zscore(score)) %>%
  ungroup()
write_fig_data(fig16_data, "app_fig16_cross_method_divergence.csv")

p_fig16 <- ggplot(fig16_data, aes(x = meeting_date, y = score_z, colour = method)) +
  geom_line(alpha = 0.22, linewidth = 0.35, na.rm = TRUE) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.85, na.rm = TRUE) +
  facet_wrap(~ corpus_long, ncol = 1, scales = "free_x") +
  scale_colour_manual(values = pal_method, drop = FALSE) +
  labs(
    title = "Cross-method temporal divergence",
    subtitle = "Z-standardised scores for statement corpora",
    x = NULL, y = "Within-corpus z-score", colour = NULL
  ) +
  thesis_theme(10)
save_fig(p_fig16, "cross_method_divergence.png", width = 10, height = 7)

# =============================================================================
# MANIFEST AND LATEX NOTES
# =============================================================================
figure_manifest <- tribble(
  ~filename, ~recommended_location, ~include_in_main_results, ~latex_label, ~source_data,
  "fig6_1_score_distributions.png", "Section 6.1", TRUE, "fig:score_distributions_v31", "fig6_1_score_distribution_points.csv; fig6_1_score_distribution_summary.csv",
  "fig6_2_agreement_with_rate.png", "Section 6.2", TRUE, "fig:agreement_with_rate_v31", "fig6_2_agreement_with_rate.csv",
  "fig6_3_rf_confusion_matrices.png", "Section 6.3", TRUE, "fig:rf_confusion_v31", "fig6_3_rf_confusion_matrix.csv",
  "fig6_4_rf_feature_importance.png", "Section 6.3", TRUE, "fig:rf_feature_importance_v31", "fig6_4_rf_signed_feature_importance.csv",
  "fig6_5_rq1_method_r2_heatmap.png", "Section 6.4", TRUE, "fig:rq1_method_r2_v31", "fig6_5_rq1_method_r2.csv",
  "fig6_6_dual_mandate_r2_gains.png", "Section 6.4.2", TRUE, "fig:dual_mandate_r2_gains_v31", "fig6_6_dual_mandate_r2_gains.csv",
  "fig6_7_rq2_direct_oos_r2.png", "Section 6.5", TRUE, "fig:rq2_direct_oos_r2_v31", "fig6_7_rq2_direct_oos_r2.csv",
  "fig6_8_rq2_incremental_r2.png", "Section 6.5 or Appendix", FALSE, "fig:rq2_incremental_r2_v31", "fig6_8_rq2_incremental_r2_beyond_inertia.csv",
  "net_score_timeseries_statements.png", "Appendix C", FALSE, "fig:dict_timeseries_stmt_v31", "app_fig02_net_score_timeseries_statements.csv",
  "tone_rate_overlay.png", "Appendix C only", FALSE, "fig:tone_rate_overlay_index_v31", "app_fig03_tone_rate_overlay_index.csv",
  "maturity_response_profile.png", "Optional Section 6.4 or Appendix B", FALSE, "fig:maturity_profile_v31", "app_fig04_maturity_response_profile.csv",
  "coef_stability_fe_ladder.png", "Appendix B", FALSE, "fig:coef_stability_v31", "app_fig05_coef_stability_fe_ladder.csv",
  "fwbw_score_scatter.png", "Appendix C", FALSE, "fig:fwbw_scatter_v31", "app_fig06_fwbw_score_scatter.csv",
  "comparison_distribution_overlay.png", "Appendix C", FALSE, "fig:score_distribution_overlay_v31", "fig6_1_score_distribution_points.csv",
  "rolling_agreement_rate.png", "Appendix C", FALSE, "fig:rolling_agreement_v31", "app_fig11_rolling_agreement_rate.csv",
  "comparison_score_scatter.png", "Appendix C", FALSE, "fig:score_scatter_v31", "app_fig12_comparison_score_scatter.csv",
  "dm_decomposition_timeseries.png", "Appendix C or Section 6.4.2", FALSE, "fig:dm_decomposition_ts_v31", "app_fig14_dm_decomposition_timeseries.csv",
  "robustness_specification_grid.png", "Appendix B", FALSE, "fig:robustness_grid_v31", "app_fig15_robustness_specification_grid.csv",
  "cross_method_divergence.png", "Appendix C", FALSE, "fig:cross_method_divergence_v31", "app_fig16_cross_method_divergence.csv"
)
write_fig_data(figure_manifest, "00_figure_manifest.csv")

figure_recommendations <- tribble(
  ~old_png_name, ~current_action, ~reason,
  "net_score_distribution_normal.png", "Replace with fig6_1_score_distributions.png", "Dictionary-only distribution is too narrow for the current three-method v31 results.",
  "net_score_timeseries_statements.png", "Appendix only", "Useful diagnostic but not central to RQ1/RQ2.",
  "tone_rate_overlay.png", "Appendix only / cautious", "Uses cumulative bps-change index, not official rate level.",
  "maturity_response_profile.png", "Optional appendix or main", "Useful if you want to visualise short-end versus long-end yield attenuation.",
  "coef_stability_fe_ladder.png", "Appendix", "Robustness diagnostic; not necessary in the main narrative unless space allows.",
  "fwbw_score_scatter.png", "Appendix", "Diagnostic for decomposition, not a main result.",
  "rf_confusion_heatmap.png", "Include main", "Important for explaining RF classification accuracy and errors.",
  "rf_feature_importance_bars.png", "Include main or appendix", "Important for RF interpretability.",
  "rf_score_distribution_normal.png", "Replace with fig6_1_score_distributions.png", "Merged method comparison is cleaner.",
  "comparison_agreement_barplot.png", "Include main", "Direct visual counterpart to agreement table.",
  "rolling_agreement_rate.png", "Appendix", "Regime diagnostic; not essential for main results.",
  "comparison_score_scatter.png", "Appendix", "Diagnostic for method disagreement.",
  "comparison_distribution_overlay.png", "Appendix or replace with fig6_1", "Distribution comparison already covered by fig6_1.",
  "dm_decomposition_timeseries.png", "Appendix / optional", "Current v31 evidence is better captured by R2-gain figure.",
  "robustness_specification_grid.png", "Appendix only", "Old version was dictionary-centred; current v31 uses redesigned strongest-result grid.",
  "cross_method_divergence.png", "Appendix", "Interesting but not central to final Results chapter."
)
write_fig_data(figure_recommendations, "00_old_figure_recommendations.csv")

readme_lines <- c(
  "Generated v31 thesis figures",
  "================================",
  paste0("Input folder:  ", input_dir),
  paste0("Output folder: ", output_dir),
  "",
  "Recommended MAIN Results figures:",
  "  1. fig6_1_score_distributions.png",
  "  2. fig6_2_agreement_with_rate.png",
  "  3. fig6_3_rf_confusion_matrices.png",
  "  4. fig6_4_rf_feature_importance.png",
  "  5. fig6_5_rq1_method_r2_heatmap.png",
  "  6. fig6_6_dual_mandate_r2_gains.png",
  "  7. fig6_7_rq2_direct_oos_r2.png",
  "",
  "Optional / appendix figures are also generated using the old draft filenames.",
  "All plotted numbers are saved in the figure_data subfolder.",
  "",
  "Important caution:",
  "  tone_rate_overlay.png uses a cumulative bps-change index because v31 files do",
  "  not contain official policy-rate level series. Do not label it as an official",
  "  policy-rate path unless you replace the index with official rate levels."
)
writeLines(readme_lines, con = file.path(output_dir, "README_generated_figures.txt"))

message("\nDONE. Figures written to: ", output_dir)
message("Figure data written to: ", figure_data_dir)
message("Upload the main figure PNGs to Overleaf, then share them back for the updated Results chapter.")
