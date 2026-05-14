# =============================================================================
# data_chapter_checks_FIXED_V5.R
# VERSION: FIXED_V5_MATCH_NEWEST_SCORING_MODEL_MASTER_CSV_FIX_2026_05_13
# =============================================================================
# Uses newest draft scoring model: v34 stop-word-removed dictionary,
# Laplace-smoothed score (H-D)/(H+D+10), and +/-0.05 label threshold.
# Purpose:
#   Rebuild and audit every Chapter 4 (Data) table/figure input from the raw data.
#   The script prints each output to the R console and writes CSV files to:
#       C:/Users/Kajetan/Desktop/final thesis/data check
#
# Outputs:
#   - Table 4.1: textual corpus overview
#   - Figure 4.1: sample coverage timeline data + PNG/PDF figure
#   - Table 4.2: ECB press-conference cleaning diagnostics
#   - Table 4.3: raw-vs-clean cleaning validation using newest draft score (H-D)/(H+D+10)
#   - Table 4.4: FOMC/ECB variable mapping
#   - Table 4.5: rate decision distribution
#   - Table 4.6: sample construction and attrition
#   - Table 4.7: document-length descriptive statistics
#   - Table 4.8: financial-market descriptive statistics
#   - Table 4.9: rate decision distribution by era
#
# Notes:
#   1. This is an audit/check script. It does NOT run the full RF model.
#   2. It extracts the dictionary and rate-decision vectors from the model file
#      v31_old_scoring.R without sourcing/running the full model.
#   3. FOMC rate/yield variables from USMPD are converted from percentage-point
#      changes to basis points by multiplying by 100. Equity/FX variables remain
#      in percent returns. EA-MPD OIS variables are already in basis points.
# =============================================================================

# =============================================================================
# 0. SETUP
# =============================================================================

find_repo_root <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = TRUE)
  repeat {
    if (dir.exists(file.path(current, "data", "raw")) &&
        dir.exists(file.path(current, "code"))) {
      return(current)
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not find repository root containing data/raw and code.")
    }
    current <- parent
  }
}

BASE_DIR <- find_repo_root()
RAW_DIR  <- file.path(BASE_DIR, "data", "raw")
OUT_DIR  <- file.path(BASE_DIR, "reproduction", "data_check")

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

required_pkgs <- c(
  "readr", "readxl", "dplyr", "tidyr", "stringr", "tibble", "purrr",
  "tidytext", "SnowballC", "ggplot2", "scales"
)
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing required packages. Install them with:\n",
    "install.packages(c(", paste(sprintf('"%s"', missing_pkgs), collapse = ", "), "))"
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(tibble)
  library(purrr)
  library(tidytext)
  library(SnowballC)
  library(ggplot2)
  library(scales)
})

# Force dplyr verbs to win over masking
select    <- dplyr::select
filter    <- dplyr::filter
summarise <- dplyr::summarise
mutate    <- dplyr::mutate
arrange   <- dplyr::arrange
rename    <- dplyr::rename

# Registry for the single master CSV and output file registry.
# Each call to write_and_print() appends its output automatically.
.output_registry <- list()
.output_frames   <- list()

log_file <- file.path(OUT_DIR, "data_check_console_log.txt")
log_con <- file(log_file, open = "wt", encoding = "UTF-8")
sink(log_con, split = TRUE)
on.exit({
  try(sink(), silent = TRUE)
  try(close(log_con), silent = TRUE)
}, add = TRUE)

cat("============================================================\n")
cat("CHAPTER 4 DATA CHECK SCRIPT\n")
cat("============================================================\n")
cat(sprintf("Base dir:   %s\n", BASE_DIR))
cat(sprintf("Raw dir:    %s\n", RAW_DIR))
cat(sprintf("Output dir: %s\n", OUT_DIR))
cat(sprintf("Log file:   %s\n\n", log_file))

# =============================================================================
# 1. HELPERS
# =============================================================================

resolve_file <- function(filename) {
  candidates <- c(
    file.path(RAW_DIR, filename),
    file.path(BASE_DIR, "code", filename),
    file.path(BASE_DIR, filename),
    file.path(getwd(), filename),
    filename
  )
  candidates <- unique(candidates)
  hit <- candidates[file.exists(candidates)][1]
  if (is.na(hit)) {
    stop(
      "Could not find file: ", filename, "\nSearched:\n",
      paste0("  - ", candidates, collapse = "\n")
    )
  }
  normalizePath(hit, winslash = "/", mustWork = TRUE)
}

find_optional_file <- function(filename) {
  candidates <- c(
    file.path(RAW_DIR, filename),
    file.path(BASE_DIR, "code", filename),
    file.path(BASE_DIR, filename),
    file.path(getwd(), filename),
    filename
  )
  candidates <- unique(candidates)
  hit <- candidates[file.exists(candidates)][1]
  if (is.na(hit)) return(NA_character_)
  normalizePath(hit, winslash = "/", mustWork = TRUE)
}

register_artifact <- function(output_name, output_title, output_file, output_type = "csv") {
  .output_registry[[length(.output_registry) + 1L]] <<- tibble(
    output_name = output_name,
    output_title = output_title,
    output_file = normalizePath(output_file, winslash = "/", mustWork = FALSE),
    output_type = output_type
  )
}

write_and_print <- function(df, stem, title = stem, n_print = Inf) {
  path <- file.path(OUT_DIR, paste0(stem, ".csv"))
  readr::write_csv(df, path, na = "")

  # Store for the final long-format master CSV.
  .output_frames[[stem]] <<- list(title = title, file = path, data = as_tibble(df))
  register_artifact(stem, title, path, "csv")

  cat("\n------------------------------------------------------------\n")
  cat(title, "\n", sep = "")
  cat("------------------------------------------------------------\n")

  df_print <- as.data.frame(df)
  if (is.infinite(n_print)) {
    print(df_print, row.names = FALSE)
  } else {
    print(utils::head(df_print, n_print), row.names = FALSE)
    if (nrow(df_print) > n_print) {
      cat(sprintf("\n... %d additional rows written to CSV.\n", nrow(df_print) - n_print))
    }
  }

  cat(sprintf("\nWrote: %s\n", path))
  invisible(df)
}

safe_pct <- function(x, digits = 1) {
  ifelse(is.na(x), NA_character_, paste0(round(100 * x, digits), "%"))
}

to_date <- function(x) {
  # Robust date converter for CSV and Excel inputs.
  # Handles Date/POSIX, Excel serial numbers, ISO strings, and EA-MPD
  # mixed-format strings such as "12/09/2024" (dd/mm/YYYY).
  if (inherits(x, "Date")) return(x)
  if (inherits(x, "POSIXt")) return(as.Date(x))

  if (is.numeric(x)) {
    return(as.Date(x, origin = "1899-12-30"))
  }

  x_chr <- trimws(as.character(x))
  x_chr[x_chr %in% c("", "NA", "NaN", "NULL")] <- NA_character_

  out <- rep(as.Date(NA), length(x_chr))

  # Excel serials sometimes arrive as character strings.
  suppressWarnings(x_num <- as.numeric(x_chr))
  is_serial <- !is.na(x_num) & is.na(out) & x_num > 20000 & x_num < 60000
  out[is_serial] <- as.Date(x_num[is_serial], origin = "1899-12-30")

  # Try common date formats. Put day/month/year before month/day/year because
  # EA-MPD uses European dates in its recent rows.
  formats <- c(
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S"
  )

  for (fmt in formats) {
    miss <- is.na(out) & !is.na(x_chr)
    if (!any(miss)) break
    suppressWarnings(parsed <- as.Date(x_chr[miss], format = fmt))
    idx <- which(miss)
    out[idx[!is.na(parsed)]] <- parsed[!is.na(parsed)]
  }

  # Last fallback for strings beginning with ISO date, e.g. "1999-01-07 UTC".
  miss <- is.na(out) & !is.na(x_chr)
  if (any(miss)) {
    iso10 <- substr(x_chr[miss], 1, 10)
    suppressWarnings(parsed <- as.Date(iso10, format = "%Y-%m-%d"))
    idx <- which(miss)
    out[idx[!is.na(parsed)]] <- parsed[!is.na(parsed)]
  }

  bad <- which(is.na(out) & !is.na(x_chr))
  if (length(bad) > 0) {
    warning(
      "Some dates could not be parsed. First bad values: ",
      paste(unique(x_chr[bad])[1:min(10, length(unique(x_chr[bad])))], collapse = ", ")
    )
  }

  out
}

first_existing_col <- function(df, candidates, label = "column") {
  hit <- candidates[candidates %in% names(df)][1]
  if (is.na(hit)) {
    warning(sprintf("No %s column found. Tried: %s", label, paste(candidates, collapse = ", ")))
    return(NA_character_)
  }
  hit
}

extract_vector_assignment <- function(model_file, object_name) {
  lines <- readLines(model_file, warn = FALSE, encoding = "UTF-8")
  start <- grep(paste0("^\\s*", object_name, "\\s*<-\\s*c\\s*\\("), lines)[1]
  if (is.na(start)) stop("Could not find assignment for object: ", object_name)

  block <- character(0)
  depth <- 0L
  for (i in start:length(lines)) {
    line <- lines[i]
    block <- c(block, line)
    no_comment <- sub("#.*$", "", line)
    depth <- depth + stringr::str_count(no_comment, fixed("(")) - stringr::str_count(no_comment, fixed(")"))
    if (i > start && depth <= 0L) break
  }

  env <- new.env(parent = baseenv())
  eval(parse(text = paste(block, collapse = "\n")), envir = env)
  get(object_name, envir = env)
}

label_from_score <- function(x, threshold = 0.10) {
  dplyr::case_when(
    is.na(x) ~ NA_character_,
    x > threshold ~ "HAWKISH",
    x < -threshold ~ "DOVISH",
    TRUE ~ "NEUTRAL"
  )
}

class_from_bps <- function(x) {
  dplyr::case_when(
    is.na(x) ~ NA_character_,
    x > 0 ~ "Hike",
    x < 0 ~ "Cut",
    TRUE ~ "Hold"
  )
}

# =============================================================================
# 2. LOAD RAW DATA
# =============================================================================

cat("\nLoading raw files...\n")

fomc_statements_path <- resolve_file("fomc_statements_econometrics.csv")
fomc_pc_path         <- resolve_file("fomc_press_conferences_econometrics.csv")
ecb_decisions_path   <- resolve_file("ecb_decisions_econometrics.csv")
ecb_pc_path          <- resolve_file("ecb_statements_econometrics.csv")
usmpd_path           <- resolve_file("USMPD.xlsx")
eampd_path           <- resolve_file("EA-MPD.xlsx")
model_file           <- resolve_file("v31_old_scoring.R")
manual_recovery_path <- find_optional_file("manual_text_recovery.csv")

cat(sprintf("  FOMC statements:       %s\n", fomc_statements_path))
cat(sprintf("  FOMC press conf.:      %s\n", fomc_pc_path))
cat(sprintf("  ECB decisions/stmt:    %s\n", ecb_decisions_path))
cat(sprintf("  ECB PC raw file:       %s\n", ecb_pc_path))
cat(sprintf("  USMPD:                 %s\n", usmpd_path))
cat(sprintf("  EA-MPD:                %s\n", eampd_path))
cat(sprintf("  Model file:            %s\n", model_file))
cat(sprintf("  Manual recovery file:  %s\n", ifelse(is.na(manual_recovery_path), "not found", manual_recovery_path)))

fomc_statements <- read_csv(fomc_statements_path, show_col_types = FALSE) %>%
  mutate(
    source = "fomc_statements",
    corpus = "FOMC Statement",
    event_type = "statement",
    meeting_date = to_date(meeting_date)
  )

fomc_press_conferences <- read_csv(fomc_pc_path, show_col_types = FALSE) %>%
  mutate(
    source = "fomc_press_conferences",
    corpus = "FOMC Press Conference",
    event_type = "press_conference",
    meeting_date = to_date(meeting_date)
  )

# In the current pipeline, ecb_decisions_econometrics.csv is mapped to the
# statement/press-release object used in the EA-MPD press-release window.
ecb_statements <- read_csv(ecb_decisions_path, show_col_types = FALSE) %>%
  mutate(
    source = "ecb_statements",
    corpus = "ECB Statement",
    event_type = "statement",
    event_id = str_replace(event_id, "^ecb_dec_", "ecb_stmt_"),
    meeting_date = to_date(meeting_date)
  )

# ecb_statements_econometrics.csv contains the ECB introductory-statement page
# with transcript material and is mapped to the press-conference object for the
# Q&A-window analysis.
ecb_press_conferences <- read_csv(ecb_pc_path, show_col_types = FALSE) %>%
  mutate(
    source = "ecb_press_conferences",
    corpus = "ECB Press Conference",
    event_type = "press_conference",
    event_id = str_replace(event_id, "^ecb_stmt_", "ecb_pressconf_"),
    meeting_date = to_date(meeting_date)
  )

combined_raw <- bind_rows(
  fomc_statements,
  fomc_press_conferences,
  ecb_statements,
  ecb_press_conferences
) %>%
  mutate(
    text_raw = as.character(text_raw),
    original_text_clean = as.character(text_clean)
  )

cat(sprintf("\nLoaded combined raw corpus: %d documents\n", nrow(combined_raw)))
print(table(combined_raw$corpus, useNA = "ifany"))

# Mirror the newest model's ECB HTML-cache correction when the cache exists.
# The model stores this in C:/Users/Kajetan/Desktop/final thesis/output v30/ecb_html_cache.rds.
# This audit script does not scrape the web; it only reuses the local cache if present.
ecb_cache_candidates <- c(
  file.path(BASE_DIR, "output v30", "ecb_html_cache.rds"),
  file.path(OUT_DIR, "ecb_html_cache.rds"),
  file.path(RAW_DIR, "ecb_html_cache.rds")
)
ecb_cache_path <- ecb_cache_candidates[file.exists(ecb_cache_candidates)][1]

ecb_cache_audit <- tibble(
  cache_path = ifelse(is.na(ecb_cache_path), NA_character_, normalizePath(ecb_cache_path, winslash = "/", mustWork = FALSE)),
  cache_found = !is.na(ecb_cache_path),
  cached_keys = NA_integer_,
  rows_replaced = 0L
)

if (!is.na(ecb_cache_path)) {
  html_cache <- readRDS(ecb_cache_path)
  cache_keys <- names(html_cache)
  replace_idx <- which(
    combined_raw$source == "ecb_press_conferences" &
      combined_raw$event_id %in% cache_keys &
      !is.na(vapply(html_cache[combined_raw$event_id], function(x) {
        if (is.null(x)) NA_character_ else as.character(x)[1]
      }, FUN.VALUE = character(1)))
  )

  if (length(replace_idx) > 0) {
    for (i in replace_idx) {
      key <- combined_raw$event_id[i]
      cached_text <- html_cache[[key]]
      if (!is.null(cached_text) && !is.na(cached_text) && nchar(cached_text) > 100) {
        combined_raw$text_raw[i] <- cached_text
      }
    }
  }

  ecb_cache_audit <- tibble(
    cache_path = normalizePath(ecb_cache_path, winslash = "/", mustWork = FALSE),
    cache_found = TRUE,
    cached_keys = length(cache_keys),
    rows_replaced = length(replace_idx)
  )
}
write_and_print(ecb_cache_audit, "ecb_html_cache_audit", "ECB HTML cache audit - reused local cache if available")

# =============================================================================
# 3. CLEAN PRESS CONFERENCE TEXTS FOR AUDIT
# =============================================================================

ECB_SPEAKERS <- c(
  "Duisenberg", "Trichet", "Draghi", "Lagarde", "Noyer", "Papademos",
  "Constancio", "Constâncio", "de Guindos", "Lane"
)

clean_ecb_pc <- function(text) {
  if (is.na(text) || nchar(trimws(text)) < 100) return(text)

  text <- gsub(
    "(Related topics|CONTACT European Central Bank|Disclaimer|SEE ALSO|Media contacts|Reproduction is permitted).*$",
    "", text, perl = TRUE
  )
  text <- trimws(text)

  speaker_alt <- paste(ECB_SPEAKERS, collapse = "|")
  pat <- paste0("(Question[^:]*:|(?:", speaker_alt, ")\\s*:)")
  locs <- gregexpr(pat, text, perl = TRUE)[[1]]
  if (length(locs) == 0 || locs[1] == -1) return(text)

  lens <- attr(locs, "match.length")
  labs <- substring(text, locs, locs + lens - 1)

  intro <- trimws(substr(text, 1, locs[1] - 1))
  intro <- gsub("Jump to the transcript of the questions and answers", "", intro, fixed = TRUE)
  intro <- gsub("\\*\\s*\\*\\s*\\*", "", intro, perl = TRUE)
  intro <- trimws(intro)

  kept <- character(0)
  for (i in seq_along(locs)) {
    s <- locs[i] + lens[i]
    e <- if (i < length(locs)) locs[i + 1] - 1 else nchar(text)
    if (!grepl("^Question", labs[i], ignore.case = TRUE)) {
      kept <- c(kept, trimws(substring(text, s, e)))
    }
  }

  paste(c(intro, kept), collapse = " ")
}

clean_fomc_pc <- function(text) {
  if (is.na(text) || nchar(trimws(text)) < 50) return(text)

  pat <- "(?:QUESTION\\.|[A-Z][A-Z]+(?:[\\s\\-]+[A-Z][A-Z.\\-]+)+\\.)"
  locs <- gregexpr(pat, text, perl = TRUE)[[1]]
  if (length(locs) == 0 || locs[1] == -1) return(text)

  lens <- attr(locs, "match.length")
  labs <- substring(text, locs, locs + lens - 1)
  kept <- character(0)

  for (i in seq_along(locs)) {
    e <- if (i < length(locs)) locs[i + 1] - 1 else nchar(text)
    if (grepl("^CHAIR", labs[i])) {
      kept <- c(kept, trimws(substring(text, locs[i] + lens[i], e)))
    }
  }

  if (length(kept) == 0) return(text)
  paste(kept, collapse = " ")
}

combined <- combined_raw %>%
  mutate(text_clean_audit = text_raw)

combined$text_clean_audit[combined$source == "fomc_press_conferences"] <- vapply(
  combined$text_raw[combined$source == "fomc_press_conferences"],
  clean_fomc_pc,
  FUN.VALUE = character(1)
)

combined$text_clean_audit[combined$source == "ecb_press_conferences"] <- vapply(
  combined$text_raw[combined$source == "ecb_press_conferences"],
  clean_ecb_pc,
  FUN.VALUE = character(1)
)

combined <- combined %>%
  mutate(
    raw_chars = nchar(ifelse(is.na(text_raw), "", text_raw)),
    clean_chars = nchar(ifelse(is.na(text_clean_audit), "", text_clean_audit)),
    retention = if_else(raw_chars > 0, clean_chars / raw_chars, NA_real_)
  )

# =============================================================================
# 4. EXTRACT DICTIONARY + FINAL SCORING FORMULA
# =============================================================================

cat("\nExtracting dictionary vectors from model file...\n")

DICT_STOP_WORDS <- extract_vector_assignment(model_file, "DICT_STOP_WORDS")
policy_hawkish <- extract_vector_assignment(model_file, "policy_hawkish")
policy_dovish  <- extract_vector_assignment(model_file, "policy_dovish")
inflation_hawkish <- extract_vector_assignment(model_file, "inflation_hawkish")
inflation_dovish  <- extract_vector_assignment(model_file, "inflation_dovish")
employment_hawkish <- extract_vector_assignment(model_file, "employment_hawkish")
employment_dovish  <- extract_vector_assignment(model_file, "employment_dovish")
residual_hawkish <- extract_vector_assignment(model_file, "residual_hawkish")
residual_dovish  <- extract_vector_assignment(model_file, "residual_dovish")

cat(sprintf("  Stop words: %d\n", length(DICT_STOP_WORDS)))
cat(sprintf("  Policy:     %d hawkish / %d dovish\n", length(policy_hawkish), length(policy_dovish)))
cat(sprintf("  Inflation:  %d hawkish / %d dovish\n", length(inflation_hawkish), length(inflation_dovish)))
cat(sprintf("  Employment: %d hawkish / %d dovish\n", length(employment_hawkish), length(employment_dovish)))
cat(sprintf("  Residual:   %d hawkish / %d dovish\n", length(residual_hawkish), length(residual_dovish)))

# -----------------------------------------------------------------------------
# Newest draft scoring model check
# -----------------------------------------------------------------------------
# The uploaded model file currently defines the dictionary section as v34:
#   - custom stop words: DICT_STOP_WORDS, length 44
#   - raw dictionary phrases are already stop-word-removed
#   - split_and_stem() stems bigrams/trigrams directly from those phrases
#   - score_text_sep() removes DICT_STOP_WORDS from text before n-gram generation
#   - only bigram/trigram matches enter counts
#   - scores use Laplace smoothing: (H - D) / (H + D + ALPHA), ALPHA = 10
#   - dictionary labels use DICT_LABEL_THRESHOLD = 0.05
# This script mirrors that logic rather than the earlier unsmoothed (H-D)/(H+D)
# check formula.

SCORING_ALPHA <- 10
DICT_LABEL_THRESHOLD <- 0.05

split_and_stem_newest <- function(raw) {
  if (length(raw) == 0) {
    return(list(unigrams = character(0), bigrams = character(0), trigrams = character(0)))
  }
  raw <- tolower(raw)
  n <- stringr::str_count(raw, "\\S+")
  list(
    unigrams = unique(SnowballC::wordStem(raw[n == 1], language = "english")),
    bigrams  = unique(vapply(stringr::str_split(raw[n == 2], " "), function(p) {
      paste(SnowballC::wordStem(p, language = "english"), collapse = " ")
    }, FUN.VALUE = character(1))),
    trigrams = unique(vapply(stringr::str_split(raw[n >= 3], " "), function(p) {
      paste(SnowballC::wordStem(p, language = "english"), collapse = " ")
    }, FUN.VALUE = character(1)))
  )
}

DICT <- list(
  ph = split_and_stem_newest(policy_hawkish),
  pd = split_and_stem_newest(policy_dovish),
  ih = split_and_stem_newest(inflation_hawkish),
  id = split_and_stem_newest(inflation_dovish),
  eh = split_and_stem_newest(employment_hawkish),
  ed = split_and_stem_newest(employment_dovish),
  rh = split_and_stem_newest(residual_hawkish),
  rd = split_and_stem_newest(residual_dovish)
)

scoring_model_audit <- tibble(
  item = c(
    "model_file",
    "dictionary_version_detected_from_comments",
    "score_formula",
    "alpha",
    "label_threshold",
    "stop_words",
    "raw_policy_hawkish",
    "raw_policy_dovish",
    "raw_inflation_hawkish",
    "raw_inflation_dovish",
    "raw_employment_hawkish",
    "raw_employment_dovish",
    "raw_residual_hawkish",
    "raw_residual_dovish",
    "raw_total_phrases",
    "dedup_bigram_trigram_terms_used"
  ),
  value = c(
    model_file,
    "v34 stop-word-removed + pruned <3 hits, as stated in uploaded R file comments",
    "(H - D) / (H + D + 10)",
    as.character(SCORING_ALPHA),
    as.character(DICT_LABEL_THRESHOLD),
    as.character(length(DICT_STOP_WORDS)),
    as.character(length(policy_hawkish)),
    as.character(length(policy_dovish)),
    as.character(length(inflation_hawkish)),
    as.character(length(inflation_dovish)),
    as.character(length(employment_hawkish)),
    as.character(length(employment_dovish)),
    as.character(length(residual_hawkish)),
    as.character(length(residual_dovish)),
    as.character(length(policy_hawkish) + length(policy_dovish) +
                   length(inflation_hawkish) + length(inflation_dovish) +
                   length(employment_hawkish) + length(employment_dovish) +
                   length(residual_hawkish) + length(residual_dovish)),
    as.character(sum(vapply(DICT, function(x) length(x$bigrams) + length(x$trigrams), integer(1))))
  )
)
write_and_print(scoring_model_audit, "data_chapter_scoring_model_audit", "Scoring model audit - newest uploaded draft")

preclean_for_dict_newest <- function(texts) {
  cleaned <- tolower(ifelse(is.na(texts), "", texts))
  vapply(cleaned, function(txt) {
    words <- unlist(stringr::str_split(txt, "\\s+"))
    paste(words[!words %in% DICT_STOP_WORDS], collapse = " ")
  }, FUN.VALUE = character(1), USE.NAMES = FALSE)
}

count_matches <- function(tokens_df, token_col, dict_terms, out_name) {
  if (length(dict_terms) == 0) {
    return(tibble(doc_id = integer(), !!out_name := integer()))
  }
  tokens_df %>%
    filter(.data[[token_col]] %in% dict_terms) %>%
    count(doc_id, name = out_name)
}

score_text_sep_final <- function(texts) {
  # Mirrors score_text_sep() in the newest uploaded v31_old_scoring.R.
  n <- length(texts)
  df <- tibble(
    doc_id = seq_along(texts),
    text = preclean_for_dict_newest(texts)
  )

  bigrams <- df %>%
    tidytext::unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    mutate(stem = vapply(stringr::str_split(bigram, " "), function(p) {
      paste(SnowballC::wordStem(p, language = "english"), collapse = " ")
    }, FUN.VALUE = character(1)))

  trigrams <- df %>%
    tidytext::unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
    mutate(stem = vapply(stringr::str_split(trigram, " "), function(p) {
      paste(SnowballC::wordStem(p, language = "english"), collapse = " ")
    }, FUN.VALUE = character(1)))

  s <- tibble(doc_id = seq_along(texts))

  add_component <- function(base, key, dict_side) {
    b <- count_matches(bigrams, "stem", dict_side$bigrams, paste0(key, "_b"))
    t <- count_matches(trigrams, "stem", dict_side$trigrams, paste0(key, "_t"))
    base %>%
      left_join(b, by = "doc_id") %>%
      left_join(t, by = "doc_id")
  }

  for (key in names(DICT)) {
    s <- add_component(s, key, DICT[[key]])
  }

  count_cols <- setdiff(names(s), "doc_id")
  s <- s %>% mutate(across(all_of(count_cols), ~ replace_na(.x, 0L)))

  s <- s %>%
    mutate(
      ph_total = ph_b + ph_t,
      pd_total = pd_b + pd_t,
      ih_total = ih_b + ih_t,
      id_total = id_b + id_t,
      eh_total = eh_b + eh_t,
      ed_total = ed_b + ed_t,
      rh_total = rh_b + rh_t,
      rd_total = rd_b + rd_t,

      policy_score     = (ph_total - pd_total) / (ph_total + pd_total + SCORING_ALPHA),
      inflation_score  = (ih_total - id_total) / (ih_total + id_total + SCORING_ALPHA),
      employment_score = (eh_total - ed_total) / (eh_total + ed_total + SCORING_ALPHA),
      residual_score   = (rh_total - rd_total) / (rh_total + rd_total + SCORING_ALPHA),

      econ_h_total = ih_total + eh_total + rh_total,
      econ_d_total = id_total + ed_total + rd_total,
      econ_score = (econ_h_total - econ_d_total) / (econ_h_total + econ_d_total + SCORING_ALPHA),

      net_h_total = ph_total + ih_total + eh_total + rh_total,
      net_d_total = pd_total + id_total + ed_total + rd_total,
      net_score = (net_h_total - net_d_total) / (net_h_total + net_d_total + SCORING_ALPHA)
    )

  tibble(
    policy_score = s$policy_score,
    inflation_score = s$inflation_score,
    employment_score = s$employment_score,
    residual_score = s$residual_score,
    econ_score = s$econ_score,
    net_score = s$net_score,
    net_h_total = s$net_h_total,
    net_d_total = s$net_d_total
  )
}

cat("\nScoring cleaned audit text with newest draft formula (H-D)/(H+D+10), threshold +/-0.05...\n")
sc <- score_text_sep_final(combined$text_clean_audit)
combined <- bind_cols(combined, sc)

# =============================================================================
# 5. TOKEN COUNTS FOR DOCUMENT-LENGTH TABLES
# =============================================================================

data("stop_words")

count_non_stop_tokens <- function(texts) {
  n <- length(texts)
  valid <- !is.na(texts) & nchar(trimws(texts)) >= 5
  out <- rep(NA_integer_, n)
  if (!any(valid)) return(out)

  df <- tibble(doc_id = which(valid), text = texts[valid])
  counts <- df %>%
    tidytext::unnest_tokens(word, text, token = "words") %>%
    anti_join(stop_words, by = "word") %>%
    count(doc_id, name = "n_tokens")

  out[valid] <- 0L
  out[counts$doc_id] <- counts$n_tokens
  out
}

combined$non_stop_tokens <- count_non_stop_tokens(combined$text_clean_audit)

# =============================================================================
# 6. TABLE 4.1 + FIGURE 4.1 DATA
# =============================================================================

corpus_levels <- c("FOMC Statement", "FOMC Press Conference", "ECB Statement", "ECB Press Conference")

corpus_overview <- combined %>%
  mutate(corpus = factor(corpus, levels = corpus_levels)) %>%
  group_by(corpus) %>%
  summarise(
    central_bank = first(central_bank),
    document_type = case_when(
      first(event_type) == "statement" ~ "Statement",
      TRUE ~ "Press Conference"
    ),
    period = paste0(format(min(meeting_date, na.rm = TRUE), "%Y"), "--", format(max(meeting_date, na.rm = TRUE), "%Y")),
    n_documents = n(),
    avg_non_stop_tokens = round(mean(non_stop_tokens, na.rm = TRUE), 0),
    first_date = min(meeting_date, na.rm = TRUE),
    last_date = max(meeting_date, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  select(central_bank, document_type, period, n_documents, avg_non_stop_tokens, first_date, last_date)

write_and_print(corpus_overview, "tbl_4_1_corpus_overview", "Table 4.1 - Textual corpus overview")

sample_coverage_points <- combined %>%
  transmute(
    corpus,
    central_bank,
    document_type = if_else(event_type == "statement", "Statement", "Press Conference"),
    meeting_date,
    event_id,
    source,
    is_scheduled = if ("is_scheduled" %in% names(combined)) as.logical(is_scheduled) else NA,
    is_fomc_intermeeting = central_bank == "FOMC" & event_type == "statement" & !is.na(is_scheduled) & !is_scheduled
  ) %>%
  arrange(corpus, meeting_date)

write_and_print(sample_coverage_points, "fig_4_1_sample_coverage_points", "Figure 4.1 - Event-level points used in sample coverage timeline", n_print = 30)

sample_coverage_intervals <- sample_coverage_points %>%
  group_by(corpus, central_bank, document_type) %>%
  summarise(
    start_date = min(meeting_date, na.rm = TRUE),
    end_date = max(meeting_date, na.rm = TRUE),
    n_documents = n(),
    n_intermeeting_fomc = sum(is_fomc_intermeeting, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(corpus = factor(corpus, levels = corpus_levels)) %>%
  arrange(corpus)

write_and_print(sample_coverage_intervals, "fig_4_1_sample_coverage_intervals", "Figure 4.1 - Coverage intervals")

lane_map <- tibble(
  corpus = corpus_levels,
  lane = c(4, 3, 2, 1)
)

plot_df <- sample_coverage_points %>%
  left_join(lane_map, by = "corpus")
interval_df <- sample_coverage_intervals %>%
  left_join(lane_map, by = "corpus")

fig_4_1 <- ggplot() +
  annotate("rect", xmin = as.Date("2008-01-01"), xmax = as.Date("2009-12-31"), ymin = -Inf, ymax = Inf, alpha = 0.10) +
  annotate("rect", xmin = as.Date("2020-01-01"), xmax = as.Date("2021-12-31"), ymin = -Inf, ymax = Inf, alpha = 0.10) +
  geom_segment(
    data = interval_df,
    aes(x = start_date, xend = end_date, y = lane, yend = lane),
    linewidth = 5,
    lineend = "round",
    alpha = 0.22
  ) +
  geom_point(
    data = plot_df,
    aes(x = meeting_date, y = lane),
    shape = "|",
    size = 3.4,
    alpha = 0.45
  ) +
  geom_point(
    data = plot_df %>% filter(is_fomc_intermeeting),
    aes(x = meeting_date, y = lane),
    shape = "|",
    size = 6,
    alpha = 0.95
  ) +
  geom_vline(xintercept = as.numeric(as.Date("2011-04-27")), linetype = "dashed", linewidth = 0.35) +
  geom_vline(xintercept = as.numeric(as.Date("2015-01-01")), linetype = "dashed", linewidth = 0.35) +
  annotate("text", x = as.Date("2008-12-31"), y = 0.55, label = "GFC", size = 3, fontface = "italic") +
  annotate("text", x = as.Date("2020-12-31"), y = 0.55, label = "COVID", size = 3, fontface = "italic") +
  annotate("text", x = as.Date("2011-04-27"), y = 4.55, label = "FOMC PC starts\n(Apr 2011)", size = 3, hjust = 1.05) +
  annotate("text", x = as.Date("2015-01-01"), y = 4.55, label = "ECB switches to\n6-weekly (Jan 2015)", size = 3, hjust = -0.05) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y", limits = c(as.Date("1994-01-01"), as.Date("2026-12-31"))) +
  scale_y_continuous(
    breaks = lane_map$lane,
    labels = lane_map$corpus,
    limits = c(0.45, 4.85),
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  labs(
    title = "Sample coverage timeline, 1994--2026",
    subtitle = "Each tick marks one document/meeting; taller ticks mark FOMC inter-meeting actions.",
    x = NULL,
    y = NULL,
    caption = "Grey bands mark the GFC (2008--2009) and COVID period (2020--2021)."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title.position = "plot"
  )

png_path <- file.path(OUT_DIR, "fig_4_1_sample_coverage_timeline.png")
pdf_path <- file.path(OUT_DIR, "fig_4_1_sample_coverage_timeline.pdf")
ggsave(png_path, fig_4_1, width = 8.5, height = 4.6, dpi = 300)
ggsave(pdf_path, fig_4_1, width = 8.5, height = 4.6)
register_artifact("fig_4_1_sample_coverage_timeline_png", "Figure 4.1 - sample coverage timeline PNG", png_path, "png")
register_artifact("fig_4_1_sample_coverage_timeline_pdf", "Figure 4.1 - sample coverage timeline PDF", pdf_path, "pdf")
cat(sprintf("\nSaved Figure 4.1 PNG: %s\n", png_path))
cat(sprintf("Saved Figure 4.1 PDF: %s\n", pdf_path))

# =============================================================================
# 7. TABLE 4.2 - CLEANING DIAGNOSTICS
# =============================================================================

ecb_pc_diag_docs <- combined %>%
  filter(source == "ecb_press_conferences") %>%
  transmute(
    meeting_date,
    event_id,
    raw_chars,
    clean_chars,
    retention_pct = 100 * retention,
    barely_cleaned_gt95 = retention > 0.95,
    over_cleaned_lt20 = retention < 0.20
  )

write_and_print(ecb_pc_diag_docs, "tbl_4_2_ecb_pc_cleaning_document_diagnostics", "Table 4.2 support - ECB PC document-level cleaning diagnostics", n_print = 25)

ecb_pc_cleaning_diag <- ecb_pc_diag_docs %>%
  summarise(
    transcripts_cleaned = n(),
    mean_pct_text_retained = round(mean(retention_pct, na.rm = TRUE), 1),
    median_pct_text_retained = round(median(retention_pct, na.rm = TRUE), 1),
    documents_gt95_pct_retained = sum(barely_cleaned_gt95, na.rm = TRUE),
    documents_lt20_pct_retained = sum(over_cleaned_lt20, na.rm = TRUE)
  )

write_and_print(ecb_pc_cleaning_diag, "tbl_4_2_ecb_pc_cleaning_diagnostics", "Table 4.2 - ECB press conference cleaning diagnostics")

# =============================================================================
# 8. TABLE 4.3 - CLEANING VALIDATION WITH FINAL SCORE
# =============================================================================

cat("\nScoring raw and cleaned press-conference texts for cleaning validation...\n")
pc_docs <- combined %>%
  filter(source %in% c("fomc_press_conferences", "ecb_press_conferences")) %>%
  mutate(source_short = if_else(source == "fomc_press_conferences", "FOMC PC", "ECB PC"))

raw_scores <- score_text_sep_final(pc_docs$text_raw)$net_score
clean_scores <- score_text_sep_final(pc_docs$text_clean_audit)$net_score

cleaning_validation_docs <- pc_docs %>%
  transmute(
    source = source_short,
    meeting_date,
    event_id,
    raw_score = raw_scores,
    clean_score = clean_scores,
    shift = clean_score - raw_score,
    raw_label = label_from_score(raw_score, threshold = DICT_LABEL_THRESHOLD),
    clean_label = label_from_score(clean_score, threshold = DICT_LABEL_THRESHOLD),
    label_flip = raw_label != clean_label
  )

write_and_print(cleaning_validation_docs, "tbl_4_3_cleaning_validation_document_scores", "Table 4.3 support - document-level raw/clean score shifts", n_print = 30)

cleaning_validation <- cleaning_validation_docs %>%
  group_by(source) %>%
  summarise(
    N = sum(!is.na(raw_score) & !is.na(clean_score)),
    mean_raw = round(mean(raw_score, na.rm = TRUE), 4),
    mean_clean = round(mean(clean_score, na.rm = TRUE), 4),
    mean_shift = round(mean(shift, na.rm = TRUE), 4),
    corr_raw_clean = round(cor(raw_score, clean_score, use = "pairwise.complete.obs"), 3),
    flips_pct = round(100 * mean(label_flip, na.rm = TRUE), 1),
    .groups = "drop"
  )

write_and_print(cleaning_validation, "tbl_4_3_cleaning_validation", "Table 4.3 - Effect of text cleaning on dictionary scores (newest scoring model)")

# =============================================================================
# 9. LOAD MARKET DATA + VARIABLE MAP
# =============================================================================

cat("\nLoading USMPD and EA-MPD market data...\n")

read_us_sheet <- function(sheet_name) {
  read_excel(usmpd_path, sheet = sheet_name) %>%
    mutate(Date = to_date(Date))
}

us_stmt_raw <- read_us_sheet("Statements")
us_pc_raw   <- read_us_sheet("Press Conferences")

eas_pr_raw <- read_excel(eampd_path, sheet = "Press Release Window") %>%
  mutate(date = to_date(date))
eas_pc_raw <- read_excel(eampd_path, sheet = "Press Conference Window") %>%
  mutate(date = to_date(date))

# USMPD rates/yields are percentage-point changes. Convert them to basis points.
us_rate_cols <- c("MP1", "MP2", "FF1", "FF2", "OIS1Y", "OIS2Y", "UST3M", "UST6M", "UST2Y", "UST5Y", "UST10Y", "UST30Y", "TIPS5Y", "TIPS10Y", "TIPS30Y")
convert_us_units <- function(df) {
  df %>% mutate(across(any_of(us_rate_cols), ~ .x * 100))
}

us_stmt <- us_stmt_raw %>%
  convert_us_units() %>%
  transmute(
    meeting_date = Date,
    MP1, MP2, SEP = if ("SEP" %in% names(.)) SEP else NA_real_,
    Unscheduled = if ("Unscheduled" %in% names(.)) Unscheduled else NA_real_,
    SP500, EURUSD, DXY,
    UST2Y, UST5Y, UST10Y, UST30Y,
    .market_match = TRUE
  )

us_pc <- us_pc_raw %>%
  convert_us_units() %>%
  transmute(
    meeting_date = Date,
    MP1, MP2,
    SP500, EURUSD, DXY,
    UST2Y, UST5Y, UST10Y, UST30Y,
    .market_match = TRUE
  )

col_sx5e_pr <- first_existing_col(eas_pr_raw, c("SX5E", "STOXX50", "SX5E50"), "Euro Stoxx 50")
col_sx5e_pc <- first_existing_col(eas_pc_raw, c("SX5E", "STOXX50", "SX5E50"), "Euro Stoxx 50")
col_ois_mp1_pr <- first_existing_col(eas_pr_raw, c("OIS_1M", "OIS_SW", "OIS_1Y"), "ECB OIS_MP1")
col_ois_mp1_pc <- first_existing_col(eas_pc_raw, c("OIS_1M", "OIS_SW", "OIS_1Y"), "ECB OIS_MP1")

eas_pr <- eas_pr_raw %>%
  transmute(
    meeting_date = date,
    OIS_MP1 = .data[[col_ois_mp1_pr]],
    SX5E = .data[[col_sx5e_pr]],
    EURUSD,
    OIS_2Y, OIS_5Y, OIS_10Y,
    .market_match = TRUE
  )

eas_pc <- eas_pc_raw %>%
  transmute(
    meeting_date = date,
    OIS_MP1 = .data[[col_ois_mp1_pc]],
    SX5E = .data[[col_sx5e_pc]],
    EURUSD,
    OIS_2Y, OIS_5Y, OIS_10Y,
    .market_match = TRUE
  )

variable_map <- tibble::tribble(
  ~role, ~variable, ~fomc, ~ecb,
  "Rate surprise (control)", "MP1 / OIS_MP1", "Fed funds futures", "OIS 1M/SW/1Y detected by script",
  "Equity index", "S&P 500 / SX5E", "S&P 500", paste0("Euro Stoxx 50 (", col_sx5e_pr, ")"),
  "Exchange rate", "EUR/USD", "EUR/USD", "EUR/USD",
  "Short-term yield", "UST 2Y / OIS 2Y", "US Treasury 2Y", "Euro area OIS 2Y",
  "Medium-term yield", "UST 5Y / OIS 5Y", "US Treasury 5Y", "Euro area OIS 5Y",
  "Long-term yield", "UST 10Y / OIS 10Y", "US Treasury 10Y", "Euro area OIS 10Y",
  "Very long-term yield", "UST 30Y / ---", "US Treasury 30Y", "Not available"
)
write_and_print(variable_map, "tbl_4_4_variable_map", "Table 4.4 - Variable mapping across FOMC and ECB regressions")

# =============================================================================
# 10. ATTACH RATE DECISIONS FROM MODEL FILE
# =============================================================================

cat("\nExtracting rate-decision vectors from model file...\n")
fomc_rate_decisions <- extract_vector_assignment(model_file, "fomc_rate_decisions")
ecb_rate_decisions  <- extract_vector_assignment(model_file, "ecb_rate_decisions")
cat(sprintf("  FOMC rate decision entries: %d\n", length(fomc_rate_decisions)))
cat(sprintf("  ECB rate decision entries:  %d\n", length(ecb_rate_decisions)))

combined <- combined %>%
  mutate(
    date_key = str_extract(event_id, "\\d{8}$"),
    bps_change_statement_only = case_when(
      source == "fomc_statements" ~ as.numeric(fomc_rate_decisions[date_key]),
      source == "ecb_statements"  ~ as.numeric(ecb_rate_decisions[date_key]),
      TRUE ~ NA_real_
    )
  )

rate_calendar <- combined %>%
  filter(source %in% c("fomc_statements", "ecb_statements")) %>%
  select(central_bank, meeting_date, bps_change_statement_only) %>%
  distinct()

combined <- combined %>%
  left_join(rate_calendar, by = c("central_bank", "meeting_date"), suffix = c("", "_from_calendar")) %>%
  mutate(
    bps_change = case_when(
      source %in% c("fomc_statements", "ecb_statements") ~ bps_change_statement_only,
      event_type == "press_conference" ~ bps_change_statement_only_from_calendar,
      TRUE ~ NA_real_
    ),
    decision_class = class_from_bps(bps_change)
  )

unmatched_rate_docs <- combined %>%
  filter(source %in% c("fomc_statements", "ecb_statements") & is.na(bps_change)) %>%
  select(corpus, meeting_date, event_id)
write_and_print(unmatched_rate_docs, "rate_decision_unmatched_statement_docs", "Audit - statement documents without hand-coded bps_change", n_print = 50)

# =============================================================================
# 11. TABLE 4.5 - RATE DECISION DISTRIBUTION
# =============================================================================

rate_statement_sample <- combined %>%
  filter(source %in% c("fomc_statements", "ecb_statements"), !is.na(bps_change))

rate_distribution <- rate_statement_sample %>%
  group_by(central_bank) %>%
  summarise(
    Cut = sum(bps_change < 0, na.rm = TRUE),
    Hold = sum(bps_change == 0, na.rm = TRUE),
    Hike = sum(bps_change > 0, na.rm = TRUE),
    Total = n(),
    hold_pct = round(100 * Hold / Total, 1),
    median_abs_nonzero_bps = median(abs(bps_change[bps_change != 0]), na.rm = TRUE),
    .groups = "drop"
  )

write_and_print(rate_distribution, "tbl_4_5_rate_decision_distribution", "Table 4.5 - Rate decision distribution by central bank")

# =============================================================================
# 12. REGRESSION FRAMES + TABLE 4.6 SAMPLE ATTRITION
# =============================================================================

make_reg_frame <- function(docs, market) {
  docs %>% left_join(market, by = "meeting_date")
}

fed_stmt_docs <- combined %>% filter(source == "fomc_statements")
fed_pc_docs   <- combined %>% filter(source == "fomc_press_conferences")
ecb_stmt_docs <- combined %>% filter(source == "ecb_statements")
ecb_pc_docs   <- combined %>% filter(source == "ecb_press_conferences")

fed_stmt_reg <- make_reg_frame(fed_stmt_docs, us_stmt)
fed_pc_reg   <- make_reg_frame(fed_pc_docs, us_pc)
ecb_stmt_reg <- make_reg_frame(ecb_stmt_docs, eas_pr)
ecb_pc_reg   <- make_reg_frame(ecb_pc_docs, eas_pc)

count_attrition <- function(df, corpus_name, equity_col, surprise_col, yield10_col) {
  tibble(
    corpus = corpus_name,
    total_documents = nrow(df),
    with_market_data_date_match = sum(!is.na(df$.market_match)),
    naive_sample = sum(!is.na(df$net_score) & !is.na(df[[equity_col]]) & !is.na(df$EURUSD)),
    controlled_sample = sum(!is.na(df$net_score) & !is.na(df[[equity_col]]) & !is.na(df$EURUSD) & !is.na(df[[surprise_col]])),
    yield_10y_sample = sum(!is.na(df$net_score) & !is.na(df[[yield10_col]]))
  )
}

sample_attrition_long <- bind_rows(
  count_attrition(fed_stmt_reg, "FOMC Statement", "SP500", "MP1", "UST10Y"),
  count_attrition(fed_pc_reg,   "FOMC Press Conference", "SP500", "MP1", "UST10Y"),
  count_attrition(ecb_stmt_reg, "ECB Statement", "SX5E", "OIS_MP1", "OIS_10Y"),
  count_attrition(ecb_pc_reg,   "ECB Press Conference", "SX5E", "OIS_MP1", "OIS_10Y")
)

write_and_print(sample_attrition_long, "tbl_4_6_sample_attrition_long", "Table 4.6 support - sample attrition long format")

sample_attrition <- sample_attrition_long %>%
  select(corpus, total_documents, with_market_data_date_match, naive_sample, controlled_sample, yield_10y_sample) %>%
  pivot_longer(-corpus, names_to = "stage", values_to = "N") %>%
  mutate(
    stage = recode(stage,
      total_documents = "Total documents",
      with_market_data_date_match = "With market-data date match",
      naive_sample = "Naive sample",
      controlled_sample = "Controlled sample",
      yield_10y_sample = "10-year yield sample"
    ),
    stage = factor(stage, levels = c("Total documents", "With market-data date match", "Naive sample", "Controlled sample", "10-year yield sample")),
    corpus = factor(corpus, levels = corpus_levels)
  ) %>%
  arrange(stage, corpus) %>%
  pivot_wider(names_from = corpus, values_from = N)

write_and_print(sample_attrition, "tbl_4_6_sample_attrition", "Table 4.6 - Sample construction and attrition")

# =============================================================================
# 13. TABLE 4.7 - DOCUMENT LENGTH DESCRIPTIVE STATISTICS
# =============================================================================

text_desc <- combined %>%
  filter(!is.na(non_stop_tokens)) %>%
  mutate(corpus = factor(corpus, levels = corpus_levels)) %>%
  group_by(corpus) %>%
  summarise(
    N = n(),
    Mean = round(mean(non_stop_tokens, na.rm = TRUE), 0),
    SD = round(sd(non_stop_tokens, na.rm = TRUE), 0),
    Min = min(non_stop_tokens, na.rm = TRUE),
    Max = max(non_stop_tokens, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(corpus)

write_and_print(text_desc, "tbl_4_7_text_descriptive_statistics", "Table 4.7 - Document length descriptive statistics")

# =============================================================================
# 14. TABLE 4.8 - MARKET DESCRIPTIVE STATISTICS
# =============================================================================

summarise_market_vars <- function(df, panel, window, vars) {
  map_dfr(vars, function(v) {
    x <- df[[v]]
    tibble(
      panel = panel,
      variable = v,
      window = window,
      N = sum(!is.na(x)),
      Mean = round(mean(x, na.rm = TRUE), 2),
      SD = round(sd(x, na.rm = TRUE), 2),
      Min = round(min(x, na.rm = TRUE), 2),
      Max = round(max(x, na.rm = TRUE), 2)
    )
  })
}

# Controlled regression frame = non-missing text score and rate surprise.
fed_stmt_ctrl <- fed_stmt_reg %>% filter(!is.na(net_score), !is.na(MP1))
fed_pc_ctrl   <- fed_pc_reg %>% filter(!is.na(net_score), !is.na(MP1))
ecb_stmt_ctrl <- ecb_stmt_reg %>% filter(!is.na(net_score), !is.na(OIS_MP1))
ecb_pc_ctrl   <- ecb_pc_reg %>% filter(!is.na(net_score), !is.na(OIS_MP1))

market_desc <- bind_rows(
  summarise_market_vars(fed_stmt_ctrl, "Panel A: FOMC (USMPD)", "Statement", c("MP1", "SP500", "EURUSD", "UST2Y", "UST10Y", "UST30Y")),
  summarise_market_vars(fed_pc_ctrl,   "Panel A: FOMC (USMPD)", "Press Conf", c("MP1", "SP500", "EURUSD", "UST10Y")),
  summarise_market_vars(ecb_stmt_ctrl, "Panel B: ECB (EA-MPD)", "Pr. Release", c("OIS_MP1", "SX5E", "EURUSD", "OIS_10Y")),
  summarise_market_vars(ecb_pc_ctrl,   "Panel B: ECB (EA-MPD)", "Press Conf", c("OIS_MP1", "SX5E", "EURUSD", "OIS_10Y"))
)

write_and_print(market_desc, "tbl_4_8_market_descriptive_statistics", "Table 4.8 - Financial market descriptive statistics")

# =============================================================================
# 15. TABLE 4.9 - RATE DECISION DISTRIBUTION BY ERA
# =============================================================================

assign_era_fomc <- function(d) {
  case_when(
    d <= as.Date("2007-12-31") ~ "1. Great Moderation",
    d <= as.Date("2009-12-31") ~ "2. GFC",
    d <= as.Date("2015-12-31") ~ "3. Zero Lower Bound",
    d <= as.Date("2019-12-31") ~ "4. Normalisation",
    d <= as.Date("2021-12-31") ~ "5. COVID & Recovery",
    d <= as.Date("2023-12-31") ~ "6. Post-COVID Tight.",
    TRUE ~ "7. Easing Cycle"
  )
}

assign_era_ecb <- function(d) {
  case_when(
    d <= as.Date("2007-12-31") ~ "1. Early ECB",
    d <= as.Date("2009-12-31") ~ "2. GFC",
    d <= as.Date("2013-12-31") ~ "3. Sovereign Debt",
    d <= as.Date("2021-12-31") ~ "4. Neg. Rates & QE",
    d <= as.Date("2023-12-31") ~ "5. Post-COVID Tight.",
    TRUE ~ "6. Easing Cycle"
  )
}

rate_by_era <- rate_statement_sample %>%
  mutate(
    era = if_else(central_bank == "FOMC", assign_era_fomc(meeting_date), assign_era_ecb(meeting_date))
  ) %>%
  group_by(central_bank, era) %>%
  summarise(
    Cut = sum(bps_change < 0, na.rm = TRUE),
    Hold = sum(bps_change == 0, na.rm = TRUE),
    Hike = sum(bps_change > 0, na.rm = TRUE),
    Total = n(),
    .groups = "drop"
  ) %>%
  arrange(central_bank, era)

write_and_print(rate_by_era, "tbl_4_9_rate_decision_distribution_by_era", "Table 4.9 - Rate decision distribution by monetary policy era")

# =============================================================================
# 16. SUMMARY CHECKS AND POTENTIAL FLAGS
# =============================================================================

summary_checks <- tibble(
  check = c(
    "Total corpus documents",
    "FOMC statement documents",
    "FOMC PC documents",
    "ECB statement documents",
    "ECB PC documents",
    "FOMC statements with coded bps_change",
    "ECB statements with coded bps_change",
    "ECB PC mean retention above 70% target?",
    "ECB PC documents >95% retained",
    "FOMC PC market-date matches",
    "ECB statement OIS_10Y non-missing in controlled frame",
    "ECB PC OIS_10Y non-missing in controlled frame"
  ),
  value = c(
    as.character(nrow(combined)),
    as.character(nrow(fed_stmt_docs)),
    as.character(nrow(fed_pc_docs)),
    as.character(nrow(ecb_stmt_docs)),
    as.character(nrow(ecb_pc_docs)),
    as.character(sum(!is.na(fed_stmt_docs$bps_change))),
    as.character(sum(!is.na(ecb_stmt_docs$bps_change))),
    as.character(ecb_pc_cleaning_diag$mean_pct_text_retained > 70),
    as.character(ecb_pc_cleaning_diag$documents_gt95_pct_retained),
    as.character(sum(!is.na(fed_pc_reg$.market_match))),
    as.character(sum(!is.na(ecb_stmt_ctrl$OIS_10Y))),
    as.character(sum(!is.na(ecb_pc_ctrl$OIS_10Y)))
  ),
  interpretation = c(
    "Should equal the raw four-corpus total used in Chapter 4.",
    "Raw FOMC statement count before market/text-score filters.",
    "Raw FOMC press-conference count before market/text-score filters.",
    "Raw ECB statement/press-release count before market/text-score filters.",
    "Raw ECB press-conference count before market/text-score filters.",
    "May be below raw documents if some statement dates are not in the hand-coded vector.",
    "May be below raw documents if some statement dates are not in the hand-coded vector.",
    "TRUE means the cleaning paragraph should explicitly discuss high retention.",
    "High values mean many ECB transcripts were barely altered by regex cleaning.",
    "If below total FOMC PCs, the latest PC likely post-dates the USMPD vintage.",
    "This is the effective long-end ECB statement sample size.",
    "This is the effective long-end ECB press-conference sample size."
  )
)

write_and_print(summary_checks, "data_chapter_check_summary", "Summary checks and flags")

# =============================================================================
# 17. MASTER CSV + OUTPUT REGISTRY
# =============================================================================

registry_path <- file.path(OUT_DIR, "output_file_registry.csv")
master_path   <- file.path(OUT_DIR, "chapter_4_data_outputs_master.csv")

# Register these two files before writing the registry, so the registry itself is complete.
register_artifact("output_file_registry", "Registry of all generated files", registry_path, "csv")
register_artifact("chapter_4_data_outputs_master", "Master long-format CSV containing all Chapter 4 outputs", master_path, "csv")

output_registry <- bind_rows(.output_registry) %>% distinct()
readr::write_csv(output_registry, registry_path, na = "")
cat(sprintf("\nWrote output registry: %s\n", registry_path))

# Robust master CSV construction:
# Earlier versions built the master from the in-memory .output_frames object.
# In some R sessions, one entry could become atomic and break obj$data. The safer
# approach is to rebuild the master from the CSV files already written to disk.
make_master_piece_from_file <- function(output_name, output_title, output_file) {
  output_file_norm <- normalizePath(output_file, winslash = "/", mustWork = FALSE)

  if (!file.exists(output_file)) {
    return(tibble(
      output_name = output_name,
      output_title = output_title,
      output_file = output_file_norm,
      row_id = NA_integer_,
      field = "FILE_NOT_FOUND",
      value = NA_character_
    ))
  }

  df <- tryCatch(
    readr::read_csv(
      output_file,
      col_types = readr::cols(.default = readr::col_character()),
      show_col_types = FALSE,
      progress = FALSE
    ),
    error = function(e) {
      tibble(.read_error = conditionMessage(e))
    }
  )

  if (ncol(df) == 0 || nrow(df) == 0) {
    return(tibble(
      output_name = output_name,
      output_title = output_title,
      output_file = output_file_norm,
      row_id = NA_integer_,
      field = ifelse(ncol(df) == 0, "EMPTY_FILE", paste(names(df), collapse = " | ")),
      value = NA_character_
    ))
  }

  df %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(
      cols = -row_id,
      names_to = "field",
      values_to = "value",
      values_transform = list(value = as.character)
    ) %>%
    mutate(
      output_name = output_name,
      output_title = output_title,
      output_file = output_file_norm,
      .before = 1
    )
}

master_source_registry <- output_registry %>%
  filter(output_type == "csv", output_name != "chapter_4_data_outputs_master")

master_csv <- purrr::pmap_dfr(
  list(
    output_name = master_source_registry$output_name,
    output_title = master_source_registry$output_title,
    output_file = master_source_registry$output_file
  ),
  make_master_piece_from_file
)

readr::write_csv(master_csv, master_path, na = "")
cat(sprintf("Wrote master CSV: %s\n", master_path))

cat("\n============================================================\n")
cat("CHAPTER 4 DATA CHECK COMPLETE\n")
cat("============================================================\n")
cat(sprintf("All outputs written to: %s\n", OUT_DIR))
cat("Key final files:\n")
cat(sprintf("  - %s\n", master_path))
cat(sprintf("  - %s\n", registry_path))
cat(sprintf("  - %s\n", log_file))
