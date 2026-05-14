# =============================================================================
# v30_merged.R   [v30 merged - RQ1 RF + 4 RQ2 expanding-window forecasts]
# =============================================================================
#
# CHANGES VS v25_dm:
#  - ADDED:    expanding-window Lasso+RF for FOUR forecast targets
#                bps_next   (rate decision at meeting t+1)
#                bps_next2  (rate decision at meeting t+2)            <- NEW
#                y2_next    (2Y yield at meeting t+1)
#                y2_next2   (2Y yield at meeting t+2)                  <- NEW
#              All 4 trained as honest expanding-window regressions
#              (fit on docs 1..t-1, predict at meeting t, never refit on test).
#              Run on each of 4 corpora x 2 text inputs (full + fwd).
#
#  - DELETED:  dict RQ2 baseline regressions (old Step 17),
#              ordered probit (old Step 18),
#              in-sample OOB yield comparison.
#
#  - REORG:    PHASE A = ALL model building / score computation
#              PHASE B = ALL regressions, plots, file writes
#              (so scores are produced first, then evaluated cleanly at end).
#
#  - KEPT:     RQ1 Lasso -> balanced ranger RF (descriptive vs dict comparison)
#              Dual-mandate dictionary scoring (Spec A/B/C)
#              FE ladders (chair + era), BERT merge, RQ1 method comparison.
#
# RUNTIME: with all flags ON the 32 expanding-window grids run roughly
#   2-6 hours on a fast laptop (~2,000 RF fits per grid). Toggle the flags
#   below to skip targets or text_fwd if you need to iterate quickly.
# =============================================================================

library(readr)
library(dplyr)
library(stringr)
library(tidytext)
library(SnowballC)
library(tidyr)
library(readxl)
library(lmtest)
library(sandwich)
library(rvest)
library(Matrix)
library(ranger)
library(tibble)
library(glmnet)
library(parallel)
library(fixest)
library(ggplot2)
library(patchwork)
library(scales)

# Force dplyr verbs to win over masking
select    <- dplyr::select
filter    <- dplyr::filter
summarise <- dplyr::summarise
mutate    <- dplyr::mutate
arrange   <- dplyr::arrange
rename    <- dplyr::rename

setwd("C:/Users/Kajetan/Desktop/final thesis")

raw_dir    <- "C:/Users/Kajetan/Desktop/final thesis/raw data"
output_dir <- "C:/Users/Kajetan/Desktop/final thesis/output v30"

stopifnot(dir.exists(raw_dir))
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
cat(sprintf("  Working dir:  %s\n", getwd()))
cat(sprintf("  Raw data dir: %s\n", raw_dir))
cat(sprintf("  Output dir:   %s\n", output_dir))
data("stop_words")
`%||%` <- function(a, b) if (!is.null(a) && !is.na(a)) a else b
set.seed(42)

# -- Global RF / runtime config ----------------------------------------------
RF_THREADS       <- max(1L, parallel::detectCores() - 1L)
RF_NTREE         <- 500L
TUNE_HYPERPARAMS <- FALSE

# -- Which RQ2 expanding-window targets to run -------------------------------
# Each ON target adds ~30-90 min for full text + another ~30-90 min for fwd
# (i.e. ~60-180 min per target with both text inputs). Flip OFF to skip.
RUN_RQ2_BPS_NEXT  <- TRUE
RUN_RQ2_BPS_NEXT2 <- TRUE   # NEW (decision two meetings ahead)
RUN_RQ2_Y2_NEXT   <- TRUE
RUN_RQ2_Y2_NEXT2  <- TRUE   # NEW (2Y yield two meetings ahead)
RUN_RQ2_TEXT_FWD  <- TRUE   # set FALSE to halve runtime (skips fwd grids)

cat("============================================================\n")
cat("  v30_merged PIPELINE - RQ1 RF + 4 expanding-window RQ2 forecasts\n")
cat(sprintf("  Threads: %d  |  Trees per RF: %d\n", RF_THREADS, RF_NTREE))
cat(sprintf("  RQ2 grids enabled: bps_next=%s bps_next2=%s y2_next=%s y2_next2=%s  | text_fwd=%s\n",
            RUN_RQ2_BPS_NEXT, RUN_RQ2_BPS_NEXT2, RUN_RQ2_Y2_NEXT, RUN_RQ2_Y2_NEXT2,
            RUN_RQ2_TEXT_FWD))
cat("============================================================\n\n")

ECB_SPEAKERS <- c("Duisenberg","Trichet","Draghi","Lagarde","Noyer",
                  "Papademos","Constancio","Constâncio","de Guindos","Lane")

# =============================================================================
# STEP 1 - LOAD RAW DATA
# =============================================================================
cat("Step 1: Loading raw data...\n")
fomc_statements <- read_csv(file.path(raw_dir, "fomc_statements_econometrics.csv"),
                            show_col_types = FALSE) |>
  mutate(source = "fomc_statements")
fomc_press_conferences <- read_csv(file.path(raw_dir, "fomc_press_conferences_econometrics.csv"),
                                   show_col_types = FALSE) |>
  mutate(source = "fomc_press_conferences")
ecb_statements <- read_csv(file.path(raw_dir, "ecb_statements_econometrics.csv"),
                           show_col_types = FALSE) |>
  mutate(source     = "ecb_press_conferences",
         event_type = "press_conference",
         event_id   = str_replace(event_id, "^ecb_stmt_", "ecb_pressconf_"))
ecb_decisions <- read_csv(file.path(raw_dir, "ecb_decisions_econometrics.csv"),
                          show_col_types = FALSE) |>
  mutate(source     = "ecb_statements",
         event_type = "statement",
         event_id   = str_replace(event_id, "^ecb_dec_", "ecb_stmt_"))

combined <- bind_rows(fomc_statements, fomc_press_conferences,
                      ecb_statements, ecb_decisions)
cat(sprintf("  Total rows: %d\n", nrow(combined)))
cat("  Event types:\n")
print(table(combined$event_type, useNA = "ifany"))

# =============================================================================
# STEP 2 - HTML RE-SCRAPE FOR 2019+ ECB DOCS (cached)
# =============================================================================
cat("\nStep 2: HTML re-scrape for 2019+ ECB docs...\n")
parse_ecb_html <- function(url) {
  page <- tryCatch(read_html(url), error = function(e) NULL)
  if (is.null(page)) return(NULL)
  paras <- page %>% html_elements("p")
  if (length(paras) == 0) return(NULL)
  kept <- character(0); dropped <- 0L
  for (p in paras) {
    p_text <- html_text(p)
    if (is.na(p_text) || nchar(trimws(p_text)) < 3) next
    se <- html_elements(p, "strong")
    if (length(se) > 0) {
      st <- paste(sapply(se, html_text), collapse = " ")
      if (nchar(st) > 0.8 * nchar(p_text)) { dropped <- dropped + 1L; next }
    }
    kept <- c(kept, trimws(p_text))
  }
  list(text = paste(kept, collapse = " "), dropped = dropped)
}

if ("url" %in% names(combined)) {
  needs_fix <- rep(FALSE, nrow(combined))
  pc_idx <- which(combined$event_type == "press_conference" &
                    combined$source == "ecb_press_conferences")
  for (i in pc_idx) {
    tr <- combined$text_raw[i]; if (is.na(tr)) next
    if (!grepl("Question\\s*:", tr, perl = TRUE) &&
        !is.na(combined$meeting_date[i]) &&
        as.Date(combined$meeting_date[i]) >= as.Date("2019-01-01"))
      needs_fix[i] <- TRUE
  }
  fix_idx <- which(needs_fix)
  cat(sprintf("  Need HTML fix: %d\n", length(fix_idx)))
  if (length(fix_idx) > 0) {
    cache_path <- file.path(output_dir, "ecb_html_cache.rds")
    html_cache <- if (file.exists(cache_path)) readRDS(cache_path) else list()
    n_c <- 0L; n_f <- 0L; n_x <- 0L
    for (i in fix_idx) {
      key <- combined$event_id[i]
      if (!is.null(html_cache[[key]]) && nchar(html_cache[[key]]) > 100) {
        combined$text_raw[i] <- html_cache[[key]]; n_c <- n_c + 1L; next
      }
      url <- combined$url[i]
      if (is.na(url) || nchar(url) < 10) { n_x <- n_x + 1L; next }
      cat(sprintf("  %s: ", as.character(combined$meeting_date[i])))
      r <- parse_ecb_html(url)
      if (is.null(r) || is.na(r$text) || nchar(r$text) < 100) {
        cat("FAILED\n"); n_x <- n_x + 1L
      } else {
        combined$text_raw[i] <- r$text; html_cache[[key]] <- r$text
        n_f <- n_f + 1L; cat(sprintf("OK (%d Q dropped)\n", r$dropped))
        Sys.sleep(1)
      }
    }
    saveRDS(html_cache, cache_path)
    cat(sprintf("  cached: %d  fetched: %d  failed: %d\n", n_c, n_f, n_x))
  }
}

# =============================================================================
# RECOVERY: scrape texts that were NA in raw inputs (run once, then validate)
# =============================================================================
recover_missing_texts <- function(out_path) {
  # Each target: event_id + ordered list of candidate URLs (try until one works)
  targets <- list(
    # FOMC — confirmed scrapable
    list(id = "fomc_stmt_20070628",
         urls = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20070628a.htm"),
    list(id = "fomc_stmt_20080122",
         urls = c("https://www.federalreserve.gov/newsevents/pressreleases/monetary20080122b.htm",
                  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20080122a.htm")),
    # ECB — uncertain whether standalone press releases exist; try common patterns
    list(id = "ecb_dec_19990204", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990204.en.html",
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990204_1.en.html")),
    list(id = "ecb_dec_19990708", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990708.en.html",
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990708_1.en.html")),
    list(id = "ecb_dec_19990805", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990805.en.html",
      "https://www.ecb.europa.eu/press/pr/date/1999/html/pr990805_1.en.html")),
    list(id = "ecb_dec_20001109", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/2000/html/pr001109.en.html",
      "https://www.ecb.europa.eu/press/pr/date/2000/html/pr001109_1.en.html")),
    list(id = "ecb_dec_20010111", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr010111.en.html",
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr010111_1.en.html")),
    list(id = "ecb_dec_20010412", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr010412.en.html",
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr010412_1.en.html")),
    list(id = "ecb_dec_20011008", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr011008.en.html",
      "https://www.ecb.europa.eu/press/pr/date/2001/html/pr011008_1.en.html")),
    list(id = "ecb_dec_20060907", urls = c(
      "https://www.ecb.europa.eu/press/pr/date/2006/html/pr060907.en.html",
      "https://www.ecb.europa.eu/press/pr/date/2006/html/pr060907_1.en.html"))
  )
  cat("\n=== RECOVERY: attempting to fetch missing texts ===\n")
  rows <- list()
  for (tgt in targets) {
    cat(sprintf("  %-25s ", tgt$id))
    got <- NA_character_; chosen_url <- NA_character_
    for (u in tgt$urls) {
      r <- tryCatch({
        Sys.sleep(1)
        pg <- xml2::read_html(u)
        # Try generic content extraction
        ps <- rvest::html_elements(pg, "#article p, .main p, #content p, article p, main p, body p")
        if (length(ps) == 0) ps <- rvest::html_elements(pg, "p")
        txt <- paste(rvest::html_text2(ps), collapse = "\n")
        # Strip nav/footer noise: keep only paragraphs with > 30 chars
        sents <- unlist(strsplit(txt, "\n", fixed = TRUE))
        txt <- paste(sents[nchar(sents) > 30], collapse = "\n")
        if (nchar(txt) < 200) NA_character_ else txt
      }, error = function(e) NA_character_)
      if (!is.na(r)) { got <- r; chosen_url <- u; break }
    }
    if (is.na(got)) cat("FAILED\n")
    else cat(sprintf("OK (%d chars from %s)\n", nchar(got),
                     sub(".*/", "", chosen_url)))
    rows[[length(rows)+1L]] <- data.frame(
      event_id = tgt$id,
      url_used = chosen_url %||% NA_character_,
      n_chars  = if (is.na(got)) 0L else nchar(got),
      text_raw = got,
      stringsAsFactors = FALSE
    )
  }
  res <- dplyr::bind_rows(rows)
  readr::write_csv(res, out_path)
  cat(sprintf("\n  Wrote %s\n  Recovered: %d of %d targets\n",
              out_path, sum(!is.na(res$text_raw)), nrow(res)))
  invisible(res)
}

# Run once — writes manual_text_recovery.csv into raw_dir
recovery <- recover_missing_texts(file.path(output_dir, "manual_text_recovery.csv"))

# =============================================================================

# =============================================================================
# STEP 3 - CLEAN ECB PC TRANSCRIPTS
# =============================================================================
cat("\nStep 3: Cleaning ECB PC transcripts...\n")
clean_ecb_pc <- function(text) {
  if (is.na(text) || nchar(trimws(text)) < 100) return(text)
  text <- gsub("(Related topics|CONTACT European Central Bank|Disclaimer|SEE ALSO|Media contacts|Reproduction is permitted).*$",
               "", text, perl = TRUE)
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
    e <- if (i < length(locs)) locs[i+1] - 1 else nchar(text)
    if (!grepl("^Question", labs[i], ignore.case = TRUE))
      kept <- c(kept, trimws(substring(text, s, e)))
  }
  paste(c(intro, kept), collapse = " ")
}

combined$text_clean <- combined$text_raw
ecb_pc_idx <- which(combined$event_type == "press_conference" &
                      combined$source == "ecb_press_conferences")
for (i in ecb_pc_idx) combined$text_clean[i] <- clean_ecb_pc(combined$text_raw[i])

# =============================================================================
# STEP 4 - CLEAN FOMC PC TRANSCRIPTS
# =============================================================================
cat("\nStep 4: Cleaning FOMC PC transcripts...\n")
clean_fomc_pc <- function(text) {
  if (is.na(text) || nchar(trimws(text)) < 50) return(text)
  pat <- "(?:QUESTION\\.|[A-Z][A-Z]+(?:[\\s\\-]+[A-Z][A-Z.\\-]+)+\\.)"
  locs <- gregexpr(pat, text, perl = TRUE)[[1]]
  if (length(locs) == 0 || locs[1] == -1) return(text)
  lens <- attr(locs, "match.length")
  labs <- substring(text, locs, locs + lens - 1)
  kept <- character(0)
  for (i in seq_along(locs)) {
    e <- if (i < length(locs)) locs[i+1] - 1 else nchar(text)
    if (grepl("^CHAIR", labs[i]))
      kept <- c(kept, trimws(substring(text, locs[i] + lens[i], e)))
  }
  if (length(kept) == 0) return(text)
  paste(kept, collapse = " ")
}

fomc_pc_idx <- which(combined$event_type == "press_conference" &
                       combined$source == "fomc_press_conferences")
for (i in fomc_pc_idx) combined$text_clean[i] <- clean_fomc_pc(combined$text_clean[i])

# === END OF PART 1 ============================================================
# =============================================================================
# STEP 5 - ATTACH RATE DECISIONS (bps_change)
# =============================================================================
cat("\nStep 5: Attaching rate decisions...\n")
fomc_rate_decisions <- c(
  "19940204"=+25,"19940322"=+25,"19940418"=+25,"19940517"=+50,"19940706"=0,
  "19940816"=+50,"19940927"=0,"19941115"=+75,"19941220"=0,"19950201"=+50,
  "19950328"=0,"19950523"=0,"19950706"=-25,"19950822"=0,"19950926"=0,
  "19951115"=0,"19951219"=-25,"19960131"=-25,"19960326"=0,"19960521"=0,
  "19960703"=0,"19960820"=0,"19960924"=0,"19961113"=0,"19961217"=0,
  "19970205"=0,"19970325"=+25,"19970520"=0,"19970702"=0,"19970819"=0,
  "19970930"=0,"19971112"=0,"19971216"=0,"19980211"=0,"19980331"=0,
  "19980519"=0,"19980701"=0,"19980818"=0,"19980929"=-25,"19981015"=-25,
  "19981117"=-25,"19990203"=0,"19990330"=0,"19990518"=0,"19990630"=+25,
  "19990824"=+25,"19991005"=0,"19991116"=+25,"19991221"=0,"20000202"=+25,
  "20000321"=+25,"20000516"=+50,"20000628"=0,"20000822"=0,"20001003"=0,
  "20001115"=0,"20001219"=0,"20010103"=-50,"20010131"=-50,"20010320"=-50,
  "20010418"=-50,"20010515"=-50,"20010627"=-25,"20010821"=-25,"20010917"=-50,
  "20011002"=-50,"20011106"=-50,"20011211"=-25,"20020130"=0,"20020319"=0,
  "20020507"=0,"20020626"=0,"20020813"=0,"20020924"=0,"20021106"=-50,
  "20021210"=0,"20030129"=0,"20030318"=0,"20030506"=0,"20030625"=-25,
  "20030812"=0,"20030916"=0,"20031028"=0,"20031209"=0,"20040128"=0,
  "20040316"=0,"20040504"=0,"20040630"=+25,"20040810"=+25,"20040921"=+25,
  "20041110"=+25,"20041214"=+25,"20050202"=+25,"20050322"=+25,"20050503"=+25,
  "20050630"=+25,"20050809"=+25,"20050920"=+25,"20051101"=+25,"20051213"=+25,
  "20060131"=+25,"20060328"=+25,"20060510"=+25,"20060629"=+25,"20060808"=0,
  "20060920"=0,"20061025"=0,"20061212"=0,"20070131"=0,"20070321"=0,
  "20070509"=0,"20070618"=0,"20070807"=0,"20070918"=-50,"20071031"=-25,
  "20071211"=-25,"20080122"=-75,"20080130"=-50,"20080318"=-75,"20080430"=-25,
  "20080625"=0,"20080805"=0,"20080916"=0,"20081008"=-50,"20081029"=-50,
  "20081216"=-100,"20090128"=0,"20090318"=0,"20090429"=0,"20090624"=0,
  "20090812"=0,"20090923"=0,"20091104"=0,"20091216"=0,"20100127"=0,
  "20100316"=0,"20100428"=0,"20100623"=0,"20100810"=0,"20100921"=0,
  "20101103"=0,"20101214"=0,"20110126"=0,"20110315"=0,"20110427"=0,
  "20110622"=0,"20110809"=0,"20110921"=0,"20111102"=0,"20111213"=0,
  "20120125"=0,"20120313"=0,"20120425"=0,"20120620"=0,"20120801"=0,
  "20120913"=0,"20121024"=0,"20121212"=0,"20130130"=0,"20130320"=0,
  "20130501"=0,"20130619"=0,"20130731"=0,"20130918"=0,"20131030"=0,
  "20131218"=0,"20140129"=0,"20140319"=0,"20140430"=0,"20140618"=0,
  "20140730"=0,"20140917"=0,"20141029"=0,"20141217"=0,"20150128"=0,
  "20150318"=0,"20150429"=0,"20150617"=0,"20150729"=0,"20150917"=0,
  "20151028"=0,"20151216"=+25,"20160127"=0,"20160316"=0,"20160427"=0,
  "20160615"=0,"20160727"=0,"20160921"=0,"20161102"=0,"20161214"=+25,
  "20170201"=0,"20170315"=+25,"20170503"=0,"20170614"=+25,"20170726"=0,
  "20170920"=0,"20171101"=0,"20171213"=+25,"20180131"=0,"20180321"=+25,
  "20180502"=0,"20180613"=+25,"20180801"=0,"20180926"=+25,"20181108"=0,
  "20181219"=+25,"20190130"=0,"20190320"=0,"20190501"=0,"20190619"=0,
  "20190731"=-25,"20190918"=-25,"20191030"=-25,"20191211"=0,"20200129"=0,
  "20200303"=-50,"20200315"=-100,"20200429"=0,"20200610"=0,"20200729"=0,
  "20200916"=0,"20201105"=0,"20201216"=0,"20210127"=0,"20210317"=0,
  "20210428"=0,"20210616"=0,"20210728"=0,"20210922"=0,"20211103"=0,
  "20211215"=0,"20220126"=0,"20220316"=+25,"20220504"=+50,"20220615"=+75,
  "20220727"=+75,"20220921"=+75,"20221102"=+75,"20221214"=+50,"20230201"=+25,
  "20230322"=+25,"20230503"=+25,"20230614"=0,"20230726"=+25,"20230920"=0,
  "20231101"=0,"20231213"=0,"20240131"=0,"20240320"=0,"20240501"=0,
  "20240612"=0,"20240731"=0,"20240918"=-50,"20241107"=-25,"20241218"=-25,
  "20250129"=0,"20250319"=0,"20250507"=0,"20250618"=0,"20250730"=0,
  "20250917"=-25,"20251029"=-25,"20251210"=-25,"20260128"=0,"20260318"=0
)
ecb_rate_decisions <- c(
  "19990107"=0,"19990204"=0,"19990304"=0,"19990408"=-50,"19990506"=0,
  "19990603"=0,"19990708"=0,"19990805"=0,"19990903"=0,"19991104"=+50,
  "19991202"=0,"20000203"=+25,"20000302"=+25,"20000413"=+25,"20000511"=+50,
  "20000608"=+25,"20000706"=0,"20000803"=0,"20000831"=+25,"20001005"=0,
  "20001102"=0,"20001207"=0,"20010118"=0,"20010201"=0,"20010301"=0,
  "20010412"=0,"20010510"=-25,"20010607"=0,"20010712"=0,"20010830"=0,
  "20010918"=-50,"20011004"=0,"20011108"=-50,"20011206"=0,"20020207"=0,
  "20020307"=0,"20020404"=0,"20020502"=0,"20020606"=0,"20020704"=0,
  "20020801"=0,"20020905"=0,"20021003"=0,"20021107"=0,"20021205"=-50,
  "20030206"=0,"20030306"=-25,"20030403"=0,"20030508"=-50,"20030605"=0,
  "20030703"=0,"20030807"=0,"20030904"=0,"20031002"=0,"20031106"=0,
  "20031204"=0,"20040108"=0,"20040205"=0,"20040304"=0,"20040401"=0,
  "20040506"=0,"20040603"=0,"20040701"=0,"20040805"=0,"20040902"=0,
  "20041007"=0,"20041104"=0,"20041202"=0,"20050203"=0,"20050303"=0,
  "20050407"=0,"20050505"=0,"20050602"=0,"20050707"=0,"20050804"=0,
  "20050901"=0,"20051006"=0,"20051103"=0,"20051201"=+25,"20060202"=+25,
  "20060302"=0,"20060406"=0,"20060504"=+25,"20060601"=0,"20060706"=0,
  "20060803"=+25,"20060901"=0,"20061005"=+25,"20061102"=0,"20061207"=+25,
  "20070111"=0,"20070208"=+25,"20070308"=0,"20070412"=0,"20070510"=+25,
  "20070607"=0,"20070705"=0,"20070802"=0,"20070906"=0,"20071004"=0,
  "20071108"=0,"20071206"=0,"20080110"=0,"20080207"=0,"20080306"=0,
  "20080403"=0,"20080508"=0,"20080605"=0,"20080703"=+25,"20080807"=0,
  "20080904"=0,"20081008"=-50,"20081106"=-50,"20081204"=-75,"20090115"=-50,
  "20090205"=-50,"20090305"=-50,"20090402"=-25,"20090507"=-25,"20090604"=0,
  "20090702"=0,"20090806"=0,"20090903"=0,"20091008"=0,"20091105"=0,
  "20091203"=0,"20100114"=0,"20100204"=0,"20100304"=0,"20100408"=0,
  "20100506"=0,"20100603"=0,"20100708"=0,"20100805"=0,"20100902"=0,
  "20101007"=0,"20101104"=0,"20101202"=0,"20110113"=0,"20110203"=0,
  "20110303"=0,"20110407"=+25,"20110505"=0,"20110609"=0,"20110707"=+25,
  "20110804"=0,"20110901"=0,"20111006"=-25,"20111103"=-25,"20111208"=-25,
  "20120112"=0,"20120209"=0,"20120308"=0,"20120405"=0,"20120503"=0,
  "20120606"=0,"20120705"=-25,"20120802"=0,"20120906"=0,"20121004"=0,
  "20121108"=0,"20121206"=0,"20130110"=0,"20130207"=0,"20130307"=0,
  "20130404"=0,"20130502"=-25,"20130606"=0,"20130704"=0,"20130801"=0,
  "20130905"=0,"20131003"=0,"20131107"=-25,"20131205"=0,"20140109"=0,
  "20140206"=0,"20140306"=0,"20140403"=0,"20140508"=-10,"20140605"=-10,
  "20140703"=0,"20140807"=0,"20140904"=-10,"20141002"=0,"20141106"=0,
  "20141204"=0,"20150122"=0,"20150305"=0,"20150415"=0,"20150603"=0,
  "20150716"=0,"20150903"=0,"20151022"=-10,"20151203"=-10,"20160121"=0,
  "20160310"=-5,"20160421"=0,"20160602"=0,"20160721"=0,"20160908"=0,
  "20161020"=0,"20161208"=0,"20170119"=0,"20170309"=0,"20170427"=0,
  "20170608"=0,"20170720"=0,"20170907"=0,"20171026"=0,"20171214"=0,
  "20180125"=0,"20180308"=0,"20180426"=0,"20180614"=0,"20180726"=0,
  "20180913"=0,"20181025"=0,"20181213"=0,"20190124"=0,"20190307"=0,
  "20190410"=0,"20190606"=0,"20190725"=0,"20190912"=-10,"20191024"=0,
  "20191212"=0,"20200123"=0,"20200312"=0,"20200430"=0,"20200604"=0,
  "20200716"=0,"20200910"=0,"20201029"=0,"20201210"=0,"20210121"=0,
  "20210311"=0,"20210422"=0,"20210610"=0,"20210722"=0,"20210909"=0,
  "20211028"=0,"20211216"=0,"20220203"=0,"20220310"=0,"20220414"=0,
  "20220609"=+25,"20220721"=+50,"20220908"=+75,"20221027"=+75,"20221215"=+50,
  "20230202"=+50,"20230316"=+50,"20230504"=+25,"20230615"=+25,"20230727"=+25,
  "20230914"=+25,"20231026"=0,"20231214"=0,"20240125"=0,"20240307"=0,
  "20240411"=0,"20240606"=-25,"20240718"=0,"20240912"=-25,"20241017"=-25,
  "20241212"=-25,"20250130"=-25,"20250306"=-25,"20250417"=-25,"20250605"=-25,
  "20250724"=0,"20250911"=0,"20251030"=0,"20251218"=0,"20260205"=0,
  "19990617"=0,"19990909"=0,"19991007"=0,"20000120"=0,"20000316"=0,
  "20000905"=0,"20001109"=0,"20001214"=0,"20010111"=0,"20010315"=0,
  "20010705"=0,"20010913"=0,"20010917"=0,"20011008"=0,"20020110"=0,
  "20020912"=0,"20021010"=0,"20030123"=0,"20030710"=0,"20050113"=0,
  "20050504"=0,"20060112"=0,"20060608"=0,"20060907"=0,"20070606"=0,
  "20080410"=0,"20081002"=0,"20081015"=0,"20100610"=0,"20110908"=0,
  "20120404"=0,"20131002"=0,"20260319"=0
)
date_str <- str_extract(combined$event_id, "\\d{8}$")
is_fomc  <- str_starts(combined$event_id, "fomc")
is_stmt  <- combined$source %in% c("fomc_statements", "ecb_statements")
combined$bps_change <- as.integer(ifelse(!is_stmt, NA,
                                         ifelse(is_fomc,
                                                fomc_rate_decisions[date_str],
                                                ecb_rate_decisions[date_str])))
cat(sprintf("  Statements with bps_change: %d\n", sum(!is.na(combined$bps_change))))


# =============================================================================
# DICTIONARY + SCORING (v33 - with stop word removal)
# =============================================================================
# WHAT CHANGED:
#   1. Custom stop word list (44 words): articles, copula/aux verbs, pronouns,
#      safe prepositions/connectors. Deliberately EXCLUDES directional words
#      (above, below, up, down, further, more, very, too, still, well, than,
#      not, no) to prevent hawk/dove collisions.
#
#   2. Dictionary phrases are written with stop words already stripped.
#      - 7 phrases that collapsed to unigrams were dropped (covered by others).
#      - Duplicates after stop removal (e.g. "raise interest rates" and
#        "raising interest rates" both stem to "rais interest rate") are kept
#        in the raw list — split_and_stem() deduplicates via unique().
#      - Former trigrams that lost a stop word became bigrams (more flexible).
#
#   3. Scoring function score_text_sep() now strips the same stop words from
#      the text BEFORE building n-grams, so dictionary and text match.
# =============================================================================

cat("\nBuilding 8-vector dual-mandate dictionary (v33 stop-word-removed)...\n")

# -- Custom stop word list --------------------------------------------------
# Conservative: only truly non-directional function words.
# KEEPS: above, below, up, down, further, more, very, too, still, well,
#        than, not, no, about, back, only, so, just
DICT_STOP_WORDS <- c(
  # Articles
  "a", "an", "the",
  # Copula / auxiliary verbs
  "is", "are", "was", "were", "be", "been", "being",
  "has", "have", "had", "having",
  "do", "does", "did",
  # Pronouns
  "it", "its", "our", "their", "we", "they",
  "i", "me", "my", "he", "she", "him", "her", "his",
  # Safe prepositions / connectors
  "of", "to", "at", "by", "with", "on", "in", "for",
  "and", "or", "but", "that", "which"
)
cat(sprintf("  Custom stop list: %d words\n", length(DICT_STOP_WORDS)))

# -- Helper: remove stop words from a character vector of words -------------
remove_stops <- function(words) {
  words[!words %in% DICT_STOP_WORDS]
}

# =============================================================================
# DICTIONARY - 8 VECTORS (v34: stop-word-removed + pruned <3 hits)
# =============================================================================
# v34 vs v33: removed 8 entries with <3 total hits across all 870 docs.
# Total: 289 phrases (was 297).
# =============================================================================

policy_hawkish <- c(
  "decided raise",
  "raise interest",
  "raise interest rates",
  "raising interest rates",
  "raised interest rates",
  "raise target",
  "raising rates",
  "decided increase",
  "increase rates",
  "increasing rates",
  "point increase", "ongoing increases",
  "rate increase", "rate hike",
  "raise rates",
  "ongoing rate increases",
  "further rate increases",
  # REMOVED: "additional rate increases"  (1 hit)
  "pace increases",
  "further increases", "additional increases",
  "sufficiently restrictive",
  "restrictive policy", "restrictive stance",
  "restrictive territory",
  "restrictive monetary policy",
  "restrictive policy stance",
  "more restrictive",
  "policy tightening", "monetary tightening",
  "further tightening",
  "tighten monetary policy",
  "tightening monetary policy",
  "additional policy firming",
  "further policy firming",
  "cumulative tightening", "quantitative tightening",
  "balance sheet reduction",
  "tightening cycle", "remove accommodation",
  "faster pace",
  # REMOVED: "determined return"  (2 hits)
  "restore price stability",
  "bring inflation down",
  "bring inflation back",
  "returning inflation",
  "appropriate raise",
  "remain vigilant", "vigilant about",
  "maintain price stability",
  "strongly committed", "firmly committed",
  "policy normalization", "normalize policy",
  "committed returning"
)

policy_dovish <- c(
  "lower target",
  "lower rate",
  "lower interest rates",
  "lowering interest rates",
  "decided lower",
  "decided reduce",
  "reduce key",
  "cut rates",
  "point reduction", "point decrease",
  "rate cut", "point cut", "rate reduction",
  "appropriate lower",
  "appropriate reduce",
  "pace reductions",
  # REMOVED: "pace cuts"  (1 hit)
  "slower pace",
  "ease monetary policy",
  "policy easing", "monetary easing", "further easing",
  "policy accommodation", "monetary accommodation",
  "accommodative policy", "accommodative stance",
  "highly accommodative", "remains accommodative",
  "ample degree",
  "asset purchases",
  "purchase programme", "purchase program",
  "net asset purchases", "net purchases",
  "quantitative easing",
  "forward guidance",
  "reinvesting principal", "reinvestment policy",
  "targeted longer", "pandemic emergency",
  "favourable financing", "preserve favourable",
  "ample liquidity", "smooth transmission",
  "support economy",
  "support recovery",
  "support economic",
  "extended period",
  "least until",
  "until substantial further",
  "lower bound", "neutral rate",
  "monetary stimulus", "policy stimulus",
  "additional stimulus",
  "further stimulus",
  # REMOVED: "provide stimulus"  (1 hit)
  "generate economic",
  "stabilises sustainably"
)

inflation_hawkish <- c(
  "high inflation", "higher inflation", "rising inflation",
  "inflation remains elevated",
  "inflation high",
  "inflation elevated",
  "elevated inflation",
  "inflation remains high",
  "still too high",
  "inflation running",
  "far too high",
  "remain well above",
  "significantly above",
  "above target",
  # REMOVED: "above aim"  (1 hit)
  "considerably above",
  "higher prices", "rising prices", "high prices",
  "prices rose", "inflation rose",
  "price pressures",
  "price pressures remain",
  "core inflation", "underlying inflation",
  "services inflation",
  "domestic price", "producer prices",
  "upward trend",
  "wage growth", "wage pressures",
  "nominal wage", "labour costs",
  "unit labour cost", "unit labor cost",
  "second round effects", "wage price spiral",
  "cost pressures", "pass through",
  "tight labor", "tight labour",
  "upward pressure",
  "inflationary pressures", "inflationary risks",
  "inflationary expectations",
  "entrenched inflation",
  "inflation overshoot", "pricing power"
)

inflation_dovish <- c(
  "low inflation", "lower inflation",
  "subdued inflation", "muted inflation",
  "falling inflation",
  "inflation declined", "inflation fell", "inflation dropped",
  "inflation eased",
  "inflation fallen",
  "continued decline",
  "moving down",
  "moving sustainably toward",
  "ongoing disinflation", "disinflation process",
  "disinflationary pressures",
  "prices fell",
  "projected decline",
  "lower prices",
  "transitory factors", "temporary factors",
  "below target", "below aim",
  "significantly below",
  "price stability objective",
  "closer target",
  "back target",
  "deflation risk", "deflationary pressure", "deflationary risk",
  "well anchored", "firmly anchored"
)

employment_hawkish <- c(
  "strong growth", "robust growth", "higher growth", "high growth",
  "rapid growth", "above potential",
  "solid growth", "solid pace", "strong pace",
  "strong economic", "strong demand",
  "economic expansion",
  "continued expand",
  "strong labor market", "strong labour market",
  "tight labor market", "tight labour market",
  "very tight",
  "very strong labor",
  "low unemployment", "job gains",
  "rising wages", "higher wages", "strong wages",
  "supply constraints", "capacity constraints",
  "labour shortage", "labor shortage"
  # REMOVED: "exceeds supply"  (1 hit)
)

employment_dovish <- c(
  "economic weakness",
  "weak growth", "low growth", "lower growth",
  "weaker growth", "slower growth", "modest growth",
  "weak demand", "weaker demand", "lower demand",
  "growth slowed",
  "drag growth",
  "economic slowdown", "economic downturn",
  "weak economic", "weaker economic",
  "sharp decline", "negative growth", "deep recession",
  "high unemployment", "rising unemployment", "higher unemployment",
  "labor market slack", "job losses",
  "loss momentum"
  # REMOVED: "rising layoffs"  (2 hits)
)

residual_hawkish <- c(
  "upside risk", "tail risks",
  "strong vigilance", "continued vigilance",
  "remain alert", "closely monitor",
  "upward revision", "revised upward", "revised up",
  "stronger than expected",
  "higher than expected",
  "supply shock", "supply disruption",
  "energy shock", "second round"
)

residual_dovish <- c(
  "downside risk",
  "risks growth",
  "path inflation",
  "negative territory", "economic headwinds",
  "credit tightening",
  # REMOVED: "financial tightening"  (1 hit)
  "tighter credit", "tighter financing",
  "tightening financial",
  "credit standards", "market turmoil",
  "systemic risk", "financial vulnerabilities",
  "liquidity trap",
  "zero lower bound", "effective lower bound",
  "downward revision", "revised downward", "revised down",
  "weaker than expected", "lower than expected",
  "geopolitical risks", "geopolitical tensions",
  "trade tensions",
  "sovereign debt", "debt crisis", "fiscal consolidation",
  "high uncertainty", "heightened uncertainty", "elevated uncertainty",
  "financial fragmentation"
)

cat(sprintf("  policy:     %d hawk / %d dove\n", length(policy_hawkish), length(policy_dovish)))
cat(sprintf("  inflation:  %d hawk / %d dove\n", length(inflation_hawkish), length(inflation_dovish)))
cat(sprintf("  employment: %d hawk / %d dove\n", length(employment_hawkish), length(employment_dovish)))
cat(sprintf("  residual:   %d hawk / %d dove\n", length(residual_hawkish), length(residual_dovish)))
cat(sprintf("  TOTAL:      %d phrases\n",
            length(policy_hawkish) + length(policy_dovish) +
              length(inflation_hawkish) + length(inflation_dovish) +
              length(employment_hawkish) + length(employment_dovish) +
              length(residual_hawkish) + length(residual_dovish)))

# =============================================================================
# split_and_stem — unchanged logic, operates on the cleaned phrases above
# =============================================================================
split_and_stem <- function(raw) {
  if (length(raw) == 0) return(list(unigrams = character(0),
                                    bigrams  = character(0),
                                    trigrams = character(0)))
  n <- str_count(raw, "\\S+")
  list(
    unigrams = unique(wordStem(raw[n == 1], language = "english")),
    bigrams  = unique(sapply(str_split(raw[n == 2], " "),
                             function(p) paste(wordStem(p, "english"), collapse = " "))),
    trigrams = unique(sapply(str_split(raw[n >= 3], " "),
                             function(p) paste(wordStem(p, "english"), collapse = " ")))
  )
}
ph <- split_and_stem(policy_hawkish);     pd <- split_and_stem(policy_dovish)
ih <- split_and_stem(inflation_hawkish);  id <- split_and_stem(inflation_dovish)
eh <- split_and_stem(employment_hawkish); ed <- split_and_stem(employment_dovish)
rh <- split_and_stem(residual_hawkish);   rd <- split_and_stem(residual_dovish)

# =============================================================================
# SCORING FUNCTION — with stop word removal from text
# =============================================================================
# CHANGE vs v30-v32: added a stop-word removal step after lowercasing and
# BEFORE building n-grams. Uses the same DICT_STOP_WORDS list so that
# dictionary stems and text stems align.
# =============================================================================
score_text_sep <- function(texts) {
  # Preprocess: lowercase, then remove stop words from each document
  cleaned <- tolower(ifelse(is.na(texts), "", texts))
  cleaned <- sapply(cleaned, function(txt) {
    words <- unlist(str_split(txt, "\\s+"))
    paste(words[!words %in% DICT_STOP_WORDS], collapse = " ")
  }, USE.NAMES = FALSE)
  
  df <- data.frame(doc_id = seq_along(texts),
                   text   = cleaned,
                   stringsAsFactors = FALSE)
  ALPHA <- 10  # Laplace smoothing constant
  
  s <- data.frame(doc_id = seq_along(texts))
  
  bi <- df %>% unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    mutate(bs = sapply(str_split(bigram, " "),
                       function(p) paste(wordStem(p, "english"), collapse = " ")))
  cb <- function(s) if (length(s) == 0) NULL else
    bi %>% filter(bs %in% s) %>% count(doc_id, name = "n")
  
  tri <- df %>% unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
    mutate(ts = sapply(str_split(trigram, " "),
                       function(p) paste(wordStem(p, "english"), collapse = " ")))
  ct <- function(s) if (length(s) == 0) NULL else
    tri %>% filter(ts %in% s) %>% count(doc_id, name = "n")
  
  sj <- function(base, add, col) {
    if (is.null(add)) base else base %>% left_join(add %>% rename(!!col := n), by = "doc_id")
  }
  s <- sj(s, cb(ph$bigrams), "ph_b"); s <- sj(s, ct(ph$trigrams), "ph_t")
  s <- sj(s, cb(pd$bigrams), "pd_b"); s <- sj(s, ct(pd$trigrams), "pd_t")
  s <- sj(s, cb(ih$bigrams), "ih_b"); s <- sj(s, ct(ih$trigrams), "ih_t")
  s <- sj(s, cb(id$bigrams), "id_b"); s <- sj(s, ct(id$trigrams), "id_t")
  s <- sj(s, cb(eh$bigrams), "eh_b"); s <- sj(s, ct(eh$trigrams), "eh_t")
  s <- sj(s, cb(ed$bigrams), "ed_b"); s <- sj(s, ct(ed$trigrams), "ed_t")
  s <- sj(s, cb(rh$bigrams), "rh_b"); s <- sj(s, ct(rh$trigrams), "rh_t")
  s <- sj(s, cb(rd$bigrams), "rd_b"); s <- sj(s, ct(rd$trigrams), "rd_t")
  
  cc <- c("ph_b","ph_t","pd_b","pd_t","ih_b","ih_t","id_b","id_t",
          "eh_b","eh_t","ed_b","ed_t","rh_b","rh_t","rd_b","rd_t")
  for (k in cc) if (!k %in% names(s)) s[[k]] <- 0L
  
  s <- s %>% mutate(across(all_of(cc), ~ replace_na(.x, 0))) %>%
    mutate(
      ph_total = ph_b + ph_t, pd_total = pd_b + pd_t,
      ih_total = ih_b + ih_t, id_total = id_b + id_t,
      eh_total = eh_b + eh_t, ed_total = ed_b + ed_t,
      rh_total = rh_b + rh_t, rd_total = rd_b + rd_t,
      # --- Laplace-smoothed: (H - D) / (H + D + α) ---
      policy_score     = (ph_total - pd_total) / (ph_total + pd_total + ALPHA),
      inflation_score  = (ih_total - id_total) / (ih_total + id_total + ALPHA),
      employment_score = (eh_total - ed_total) / (eh_total + ed_total + ALPHA),
      residual_score   = (rh_total - rd_total) / (rh_total + rd_total + ALPHA),
      econ_h_total = ih_total + eh_total + rh_total,
      econ_d_total = id_total + ed_total + rd_total,
      econ_score   = (econ_h_total - econ_d_total) / (econ_h_total + econ_d_total + ALPHA),
      net_h_total  = ph_total + ih_total + eh_total + rh_total,
      net_d_total  = pd_total + id_total + ed_total + rd_total,
      net_score    = (net_h_total - net_d_total) / (net_h_total + net_d_total + ALPHA)
    )
  
  out <- data.frame(policy_score     = rep(NA_real_, length(texts)),
                    inflation_score  = rep(NA_real_, length(texts)),
                    employment_score = rep(NA_real_, length(texts)),
                    residual_score   = rep(NA_real_, length(texts)),
                    econ_score       = rep(NA_real_, length(texts)),
                    net_score        = rep(NA_real_, length(texts)))
  out$policy_score[s$doc_id]     <- s$policy_score
  out$inflation_score[s$doc_id]  <- s$inflation_score
  out$employment_score[s$doc_id] <- s$employment_score
  out$residual_score[s$doc_id]   <- s$residual_score
  out$econ_score[s$doc_id]       <- s$econ_score
  out$net_score[s$doc_id]        <- s$net_score
  out
}
score_text_net <- function(texts) score_text_sep(texts)$net_score

# =============================================================================
# STEP 6 - SCORE ALL DOCS + FORWARD/CURRENT SPLIT
#  v30 change: also save text_fwd and text_curr columns for downstream RF use
# =============================================================================
cat("\nStep 6: Scoring documents (5 primitive + 2 derived)...\n")
sc <- score_text_sep(combined$text_clean)
combined$policy_score     <- sc$policy_score
combined$inflation_score  <- sc$inflation_score
combined$employment_score <- sc$employment_score
combined$residual_score   <- sc$residual_score
combined$econ_score       <- sc$econ_score
combined$net_score        <- sc$net_score

DICT_LABEL_THRESHOLD <- 0.05
combined <- combined %>% mutate(
  label_text       = case_when(net_score        >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               net_score        < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL"),
  policy_label     = case_when(policy_score     >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               policy_score     < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL"),
  inflation_label  = case_when(inflation_score  >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               inflation_score  < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL"),
  employment_label = case_when(employment_score >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               employment_score < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL"),
  residual_label   = case_when(residual_score   >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               residual_score   < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL"),
  econ_label       = case_when(econ_score       >  DICT_LABEL_THRESHOLD ~ "HAWKISH",
                               econ_score       < -DICT_LABEL_THRESHOLD ~ "DOVISH",
                               TRUE ~ "NEUTRAL")
)

FORWARD_RE <- paste0(
  "\\bwill\\b|\\bexpected?\\b|\\banticipat|\\bintend|",
  "\\bat least until|\\bas long as|\\bgoing forward|",
  "\\bwould be appropriate|\\bwill be appropriate|\\bappropriate to |",
  "\\bforthcoming|\\bfuture\\b|\\bperiod ahead|\\bprospect|\\boutlook|",
  "\\bbalance of risks|\\brisks to the outlook|\\blikely to\\b|",
  "\\bmonitor|\\bwatchful|\\bremain alert|\\bstand ready|\\bprepared to|",
  "\\bin coming months|\\bover the medium|\\bsubsequent|",
  "\\bproject|\\bforecast|\\bforesee|\\bforeseen|\\benvisage|",
  "\\bon track to|\\btrajectory|\\bpath of\\b|",
  "\\baim to|\\bseek to|\\blooking ahead|\\bover the coming|",
  "\\bnext meeting|\\bdata dependent|\\bmeeting by meeting|",
  "\\bconditional|\\bcontingent|\\bif warranted|",
  "\\bremains to be seen|\\btoo early to|\\bshould remain|",
  "\\bcommit to|\\bin due course|\\beventually\\b"
)
split_fwd_curr <- function(text) {
  if (is.na(text) || nchar(trimws(text)) < 10) return(list(fwd="", curr=""))
  sents <- unlist(strsplit(text, "(?<=[.!?])\\s+", perl = TRUE))
  is_fwd <- grepl(FORWARD_RE, sents, ignore.case = TRUE)
  list(fwd = paste(sents[is_fwd], collapse = " "),
       curr = paste(sents[!is_fwd], collapse = " "))
}
cat("  Forward/current split (saving text_fwd / text_curr for RF)...\n")
spl    <- lapply(combined$text_clean, split_fwd_curr)
fwd_t  <- sapply(spl, function(x) x$fwd)
curr_t <- sapply(spl, function(x) x$curr)

# v30: keep the actual fwd/curr text on the dataframe so the
# expanding-window RFs can pick it up via .data[[text_col]].
combined$text_fwd       <- fwd_t
combined$text_curr      <- curr_t
combined$net_score_fwd  <- score_text_net(fwd_t)
combined$net_score_curr <- score_text_net(curr_t)

# =============================================================================
# STEP 7 - SORT, SAVE, SPLIT INTO 4 CORPORA
# =============================================================================
cat("\nStep 7: Sorting, saving, splitting...\n")
combined_final <- combined %>%
  mutate(
    event_type       = str_trim(tolower(event_type)),
    event_type_order = factor(event_type,
                              levels = c("statement", "press_conference"),
                              ordered = TRUE)
  ) %>%
  arrange(meeting_date, event_type_order) %>%
  select(-event_type_order)
write_csv(combined_final, file.path(output_dir, "combined_final_v2.csv"))
cat("  Saved -> combined_final_v2.csv\n")

fed_statements        <- combined_final %>% filter(central_bank == "FOMC", event_type == "statement")
fed_press_conferences <- combined_final %>% filter(central_bank == "FOMC", event_type == "press_conference")
ecb_statements_df     <- combined_final %>% filter(central_bank == "ECB",  event_type == "statement")
ecb_press_conferences <- combined_final %>% filter(central_bank == "ECB",  event_type == "press_conference")

cat(sprintf("  fed_statements:        %d rows\n", nrow(fed_statements)))
cat(sprintf("  fed_press_conferences: %d rows\n", nrow(fed_press_conferences)))
cat(sprintf("  ecb_statements:        %d rows\n", nrow(ecb_statements_df)))
cat(sprintf("  ecb_press_conferences: %d rows\n", nrow(ecb_press_conferences)))
cat("\nSteps 1-7 complete.\n")

# =============================================================================
# STEP 8 - BIGRAM MATRICES FOR 4 CORPORA  (descriptive output, kept from v25)
# =============================================================================
cat("\nStep 8: Building bigram matrices (no View() in v30)...\n")
make_bigram_matrix <- function(df, corpus_name, text_col = "text_clean",
                               min_doc_freq = 1, max_terms = Inf) {
  docs <- df %>%
    mutate(
      row_id = row_number(),
      doc_id = if ("event_id" %in% names(.)) as.character(event_id) else as.character(row_id),
      text = tolower(ifelse(is.na(.data[[text_col]]), "", .data[[text_col]]))
    ) %>%
    select(row_id, doc_id, meeting_date, text)
  bigrams <- docs %>%
    unnest_tokens(bigram_raw, text, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram_raw)) %>%
    separate(bigram_raw, into = c("w1", "w2"), sep = " ", remove = FALSE) %>%
    filter(str_detect(w1, "^[a-z]+$"), str_detect(w2, "^[a-z]+$")) %>%
    mutate(bigram = paste(wordStem(w1, language = "english"),
                          wordStem(w2, language = "english"))) %>%
    count(row_id, doc_id, meeting_date, bigram, name = "count")
  term_keep <- bigrams %>%
    distinct(row_id, bigram) %>%
    count(bigram, name = "doc_freq") %>%
    filter(doc_freq >= min_doc_freq) %>%
    arrange(desc(doc_freq), bigram) %>%
    slice_head(n = max_terms)
  bigrams <- bigrams %>% filter(bigram %in% term_keep$bigram)
  mat_wide <- bigrams %>%
    select(doc_id, meeting_date, bigram, count) %>%
    pivot_wider(names_from = bigram, values_from = count, values_fill = 0) %>%
    arrange(meeting_date, doc_id)
  cat(sprintf("  %s matrix: %d documents x %d bigram columns\n",
              corpus_name, nrow(mat_wide), ncol(mat_wide) - 2))
  mat_wide
}
bigram_mat_fomc_stmt <- make_bigram_matrix(fed_statements,        "FOMC statements",        min_doc_freq = 1)
bigram_mat_fomc_pc   <- make_bigram_matrix(fed_press_conferences, "FOMC press conferences", min_doc_freq = 1)
bigram_mat_ecb_stmt  <- make_bigram_matrix(ecb_statements_df,     "ECB statements",         min_doc_freq = 1)
bigram_mat_ecb_pc    <- make_bigram_matrix(ecb_press_conferences, "ECB press conferences",  min_doc_freq = 1)

# =============================================================================
# STEP 9 - MERGE FOMC WITH USMPD (incl. UST yields)
# =============================================================================
cat("\nStep 9: Merging FOMC with USMPD...\n")
usmpd_stmt <- read_excel(file.path(raw_dir, "USMPD.xlsx"), sheet = "Statements") %>%
  mutate(Date = as.Date(Date))
usmpd_pc   <- read_excel(file.path(raw_dir, "USMPD.xlsx"), sheet = "Press Conferences") %>%
  mutate(Date = as.Date(Date))
usmpd_stmt_small <- usmpd_stmt %>%
  select(Date, MP1, MP2, SP500, EURUSD, DXY, SEP, Unscheduled,
         UST2Y, UST5Y, UST10Y, UST30Y)
usmpd_pc_small <- usmpd_pc %>%
  select(Date, MP1, MP2, SP500, EURUSD, DXY,
         UST2Y, UST5Y, UST10Y, UST30Y)
fed_stmt_reg <- fed_statements %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  left_join(usmpd_stmt_small, by = c("meeting_date" = "Date"))
fed_pc_reg <- fed_press_conferences %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  left_join(usmpd_pc_small, by = c("meeting_date" = "Date"))
cat(sprintf("  FOMC stmt matched: %d / %d  (UST10Y non-NA: %d)\n",
            sum(!is.na(fed_stmt_reg$MP1)), nrow(fed_stmt_reg),
            sum(!is.na(fed_stmt_reg$UST10Y))))
cat(sprintf("  FOMC PC matched:   %d / %d  (UST10Y non-NA: %d)\n",
            sum(!is.na(fed_pc_reg$MP1)),  nrow(fed_pc_reg),
            sum(!is.na(fed_pc_reg$UST10Y))))

# =============================================================================
# STEP 10 - MERGE ECB WITH EA-MPD (incl. OIS yields)
# =============================================================================
cat("\nStep 10: Merging ECB with EA-MPD...\n")
eampd_pr <- read_excel(file.path(raw_dir, "EA-MPD.xlsx"), sheet = "Press Release Window") %>%
  mutate(date = as.Date(as.numeric(date), origin = "1899-12-30"))
eampd_pc_eampd <- read_excel(file.path(raw_dir, "EA-MPD.xlsx"), sheet = "Press Conference Window") %>%
  mutate(date = as.Date(as.numeric(date), origin = "1899-12-30"))
find_col <- function(df, pattern) {
  m <- names(df)[str_detect(names(df), regex(pattern, ignore_case = TRUE))]
  if (length(m) == 0) { cat(sprintf("  WARNING: '%s' not found\n", pattern)); return(NULL) }
  m[1]
}
stoxx_col  <- find_col(eampd_pr, "STOXX50|SX5|stoxx50|EuroStoxx")
eurusd_col <- find_col(eampd_pr, "EURUSD|EUR_USD|eur.usd")
ois1_col   <- find_col(eampd_pr, "OIS_1M|OIS_1Y|OIS.1[MY]")
ois2_col   <- find_col(eampd_pr, "OIS_2Y|OIS.2Y")
ois5_col   <- find_col(eampd_pr, "OIS_5Y|OIS.5Y")
ois10_col  <- find_col(eampd_pr, "OIS_10Y|OIS.10Y")
build_eampd <- function(df) {
  out <- dplyr::select(df, date)
  if (!is.null(stoxx_col))  out$SX5E    <- df[[stoxx_col]]
  if (!is.null(eurusd_col)) out$EURUSD  <- df[[eurusd_col]]
  if (!is.null(ois1_col))   out$OIS_MP1 <- df[[ois1_col]]
  if (!is.null(ois2_col))   out$OIS_MP2 <- df[[ois2_col]]
  if (!is.null(ois5_col))   out$OIS_5Y  <- df[[ois5_col]]
  if (!is.null(ois10_col))  out$OIS_10Y <- df[[ois10_col]]
  out
}
eampd_pr_small       <- build_eampd(eampd_pr)
eampd_pc_eampd_small <- build_eampd(eampd_pc_eampd)
ecb_stmt_reg <- ecb_statements_df %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  left_join(eampd_pr_small, by = c("meeting_date" = "date"))
ecb_pc_reg <- ecb_press_conferences %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  left_join(eampd_pc_eampd_small, by = c("meeting_date" = "date"))
cat(sprintf("  ECB stmt matched: %d / %d  (OIS_10Y non-NA: %d)\n",
            sum(!is.na(ecb_stmt_reg$SX5E)), nrow(ecb_stmt_reg),
            sum(!is.na(ecb_stmt_reg$OIS_10Y))))
cat(sprintf("  ECB PC matched:   %d / %d  (OIS_10Y non-NA: %d)\n",
            sum(!is.na(ecb_pc_reg$SX5E)),   nrow(ecb_pc_reg),
            sum(!is.na(ecb_pc_reg$OIS_10Y))))

# =============================================================================
# STEP 10b - PROPAGATE bps_change FROM STATEMENTS TO PCs
# =============================================================================
cat("\nStep 10b: Propagating bps_change from statements to press conferences...\n")
fomc_bps_lookup <- fed_stmt_reg %>% filter(!is.na(bps_change)) %>%
  distinct(meeting_date, bps_change) %>% rename(bps_change_from_stmt = bps_change)
ecb_bps_lookup <- ecb_stmt_reg %>% filter(!is.na(bps_change)) %>%
  distinct(meeting_date, bps_change) %>% rename(bps_change_from_stmt = bps_change)
fed_pc_reg <- fed_pc_reg %>%
  left_join(fomc_bps_lookup, by = "meeting_date") %>%
  mutate(bps_change = coalesce(bps_change, bps_change_from_stmt)) %>%
  select(-bps_change_from_stmt)
ecb_pc_reg <- ecb_pc_reg %>%
  left_join(ecb_bps_lookup, by = "meeting_date") %>%
  mutate(bps_change = coalesce(bps_change, bps_change_from_stmt)) %>%
  select(-bps_change_from_stmt)
cat(sprintf("  FOMC stmt with bps_change: %d / %d\n",
            sum(!is.na(fed_stmt_reg$bps_change)), nrow(fed_stmt_reg)))
cat(sprintf("  FOMC PC   with bps_change: %d / %d  (propagated)\n",
            sum(!is.na(fed_pc_reg$bps_change)),   nrow(fed_pc_reg)))
cat(sprintf("  ECB  stmt with bps_change: %d / %d\n",
            sum(!is.na(ecb_stmt_reg$bps_change)), nrow(ecb_stmt_reg)))
cat(sprintf("  ECB  PC   with bps_change: %d / %d  (propagated)\n",
            sum(!is.na(ecb_pc_reg$bps_change)),   nrow(ecb_pc_reg)))

# =============================================================================
# STEP 10c - CHAIR + ERA CONSTRUCTION
# =============================================================================
cat("\nStep 10c: Constructing chair + era...\n")
add_chair_era <- function(df) {
  df %>%
    mutate(
      meeting_date = as.Date(meeting_date),
      chair = case_when(
        central_bank == "FOMC" & meeting_date <  as.Date("2006-02-01") ~ "Greenspan",
        central_bank == "FOMC" & meeting_date <  as.Date("2014-02-03") ~ "Bernanke",
        central_bank == "FOMC" & meeting_date <  as.Date("2018-02-05") ~ "Yellen",
        central_bank == "FOMC" & meeting_date >= as.Date("2018-02-05") ~ "Powell",
        central_bank == "ECB"  & meeting_date <  as.Date("2003-11-01") ~ "Duisenberg",
        central_bank == "ECB"  & meeting_date <  as.Date("2011-11-01") ~ "Trichet",
        central_bank == "ECB"  & meeting_date <  as.Date("2019-11-01") ~ "Draghi",
        central_bank == "ECB"  & meeting_date >= as.Date("2019-11-01") ~ "Lagarde",
        TRUE ~ NA_character_
      ),
      era = case_when(
        meeting_date <  as.Date("2008-09-15") ~ "pre_gfc",
        meeting_date <  as.Date("2022-01-01") ~ "zlb_qe",
        meeting_date >= as.Date("2022-01-01") ~ "post_zlb",
        TRUE ~ NA_character_
      )
    )
}
fed_stmt_reg <- add_chair_era(fed_stmt_reg)
fed_pc_reg   <- add_chair_era(fed_pc_reg)
ecb_stmt_reg <- add_chair_era(ecb_stmt_reg)
ecb_pc_reg   <- add_chair_era(ecb_pc_reg)
cat("  chair/era columns added\n")

write_csv(fed_stmt_reg, file.path(output_dir, "fed_stmt_reg_v2.csv"))
write_csv(fed_pc_reg, file.path(output_dir, "fed_pc_reg_v2.csv"))
write_csv(ecb_stmt_reg, file.path(output_dir, "ecb_stmt_reg_v2.csv"))
write_csv(ecb_pc_reg, file.path(output_dir, "ecb_pc_reg_v2.csv"))
cat("  Saved fed/ecb_stmt_reg_v2.csv and fed/ecb_pc_reg_v2.csv\n")

# =============================================================================
# STEP 11 - HELPER FUNCTIONS (used in PHASE B)
# =============================================================================
# run_ols          : OLS with HC1 robust SEs, prints summary
# run_fe_ladder_table : FE ladder (base / +chair / +chair+era) via fixest
# run_dual_mandate : dual-mandate Spec A vs B vs C
# These are all defined now so we can call them later in PHASE B without
# re-loading; they don't side-effect the data.
# =============================================================================
cat("\nStep 11: Defining regression helpers (run_ols / FE ladder / dual mandate)...\n")

run_ols <- function(formula, data, label = "") {
  m <- lm(formula, data = data)
  cat("\n", strrep("-", 65), "\n", label, "\n", sep = "")
  print(coeftest(m, vcov = vcovHC(m, type = "HC1")))
  cat(sprintf("  N = %d   |   R^2 = %.4f   |   Adj-R^2 = %.4f\n",
              nobs(m), summary(m)$r.squared, summary(m)$adj.r.squared))
  invisible(m)
}

run_fe_ladder_table <- function(outcome, base_rhs, data, label_prefix) {
  full_formula_str <- sprintf("%s ~ %s", outcome, base_rhs)
  fA <- as.formula(full_formula_str)
  fB <- as.formula(paste(full_formula_str, "| chair"))
  fC <- as.formula(paste(full_formula_str, "| chair + era"))
  rhs_vars <- all.vars(as.formula(paste("~", base_rhs)))
  required <- intersect(c(outcome, rhs_vars, "chair", "era"), names(data))
  d <- data[stats::complete.cases(data[, required, drop = FALSE]), ]
  if (nrow(d) < 20) {
    cat(sprintf("\n## %s -- TOO FEW OBS (%d), skipping\n", label_prefix, nrow(d)))
    return(invisible(NULL))
  }
  safe_fit <- function(f, tag) tryCatch(
    fixest::feols(f, data = d, vcov = "hetero"),
    error = function(e) { cat(sprintf("  %s failed: %s\n", tag, e$message)); NULL }
  )
  mA <- safe_fit(fA, "A"); mB <- safe_fit(fB, "B"); mC <- safe_fit(fC, "C")
  cat(sprintf("\n## %s | N=%d | A=base | B=+chair FE | C=+chair+era FE\n",
              label_prefix, nrow(d)))
  models <- Filter(Negate(is.null), list(A = mA, B = mB, C = mC))
  if (length(models) == 0) { cat("  All specs failed.\n"); return(invisible(NULL)) }
  tryCatch(
    print(fixest::etable(
      models, vcov = "hetero",
      signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
      fitstat = c("n", "r2", "ar2"))),
    error = function(e) {
      cat(sprintf("  etable failed: %s\n", e$message))
      for (nm in names(models)) {
        cat(sprintf("\n  >>> Spec %s <<<\n", nm)); print(summary(models[[nm]]))
      }
    }
  )
  control_terms <- c("MP1","MP2","OIS_MP1","OIS_MP2","bps_curr","bps_lag1",
                     "y2_curr","y2_lag1","(Intercept)")
  score_terms   <- setdiff(rhs_vars, control_terms)
  rows <- data.frame()
  for (sp in names(models)) {
    m  <- models[[sp]]
    ct <- coef(m)
    se_m <- tryCatch(fixest::se(m),     error = function(e) setNames(rep(NA, length(ct)), names(ct)))
    pv_m <- tryCatch(fixest::pvalue(m), error = function(e) setNames(rep(NA, length(ct)), names(ct)))
    r2_v <- tryCatch(fixest::fitstat(m, "r2", simplify = TRUE), error = function(e) NA_real_)
    for (st in score_terms) {
      if (st %in% names(ct)) {
        rows <- rbind(rows, data.frame(
          label = label_prefix, outcome = outcome,
          score_term = st, fe_spec = sp,
          estimate = unname(ct[st]),
          se = unname(se_m[st]),
          pvalue = unname(pv_m[st]),
          n = nobs(m), r2 = r2_v,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  invisible(list(models = models, rows = rows))
}

run_dual_mandate <- function(outcome, data, control = NULL, label_prefix = "") {
  cat(sprintf("\n=== %s [Spec A vs B vs C] ===\n", label_prefix))
  d <- data %>% filter(!is.na(.data[[outcome]]),
                       !is.na(net_score),
                       !is.na(policy_score), !is.na(econ_score),
                       !is.na(inflation_score), !is.na(employment_score),
                       !is.na(residual_score))
  if (!is.null(control)) d <- d %>% filter(!is.na(.data[[control]]))
  if (nrow(d) < 10) { cat(sprintf("  SKIP (N=%d)\n", nrow(d))); return(invisible(NULL)) }
  ctrl <- if (is.null(control)) "" else sprintf(" + %s", control)
  fA <- as.formula(sprintf("%s ~ net_score%s", outcome, ctrl))
  fB <- as.formula(sprintf("%s ~ policy_score + econ_score%s", outcome, ctrl))
  fC <- as.formula(sprintf("%s ~ policy_score + inflation_score + employment_score + residual_score%s",
                           outcome, ctrl))
  run_ols(fA, d, sprintf("%s | Spec A: net_score%s", label_prefix, ctrl))
  run_ols(fB, d, sprintf("%s | Spec B: policy + econ%s", label_prefix, ctrl))
  run_ols(fC, d, sprintf("%s | Spec C: policy + inflation + employment + residual%s",
                         label_prefix, ctrl))
}

# =============================================================================
# STEP 19 - TF-IDF infrastructure (blocklist + builder)
# =============================================================================
cat("\nStep 19: Defining TF-IDF builder + blocklist...\n")
BLOCKLIST_BIGRAMS <- c(
  # FOMC regional banks
  "cleveland richmond", "richmond atlanta", "atlanta boston",
  "chicago st", "chicago minneapoli", "loui minneapoli",
  "st loui", "kansa citi", "san francisco", "new york",
  "reserv bank", "feder reserv",
  "york philadelphia", "philadelphia cleveland", "cleveland atlanta",
  "atlanta chicago", "minneapoli kansa", "dalla san",
  # FOMC discount-rate boilerplate
  "discount rate", "request submit", "board approv",
  "relat action", "committe continu", "submit board",
  "approv request", "governor approv", "request discount",
  "charg depositori", "depositori institut",
  # ECB header/footer
  "press confer", "confer start", "relat content",
  "consider under", "polici measur", "polici decis",
  "press releas", "monetari polici", "polici transmiss",
  "communic reproduct", "reproduct permit", "permit provid",
  "council steer", "pre commit",
  # Operational language
  "minimum bid", "bid rate", "futur roll", "avoid interfer",
  "time return",
  # PC idioms
  "american famili", "social partner", "mandat goal"
)
BLOCKLIST_NAMES_AND_PLACES <- c(
  "alan greenspan", "greenspan chairman", "chairman greenspan",
  "ben bernank",    "bernank chairman",    "chairman bernank",
  "janet yellen",   "yellen chairman",     "chairman yellen",
  "jerom powel",    "powel chairman",      "chairman powel",
  "chairman susan", "susan bie",           "bie roger",
  "roger fergus",   "frederic mishkin",    "donald kohn",
  "randal kroszner","kroszner freder",     "kevin warsh",
  "wim duisenberg", "jean claud",       "claud trichet",
  "mario draghi",   "christin lagard",  "lagard christin",
  "luca papademo",  "vitor constanci",  "constanci vitor",
  "de guindo",      "philip lane",      "isabel schnabel",
  "piero cipollon",
  "washington dc",  "constitut avenu", "20th street",
  "frankfurt main", "sonnemannstrass 20",
  "european central", "central bank",
  "centuri low", "new centuri"
)
BLOCKLIST_BIGRAMS <- c(BLOCKLIST_BIGRAMS, BLOCKLIST_NAMES_AND_PLACES)
BLOCKLIST_ALL     <- BLOCKLIST_BIGRAMS
cat(sprintf("  Blocklist: %d bigrams total (incl. %d names/places)\n",
            length(BLOCKLIST_BIGRAMS), length(BLOCKLIST_NAMES_AND_PLACES)))

MAX_FEATURES <- 2000
MIN_DOC_FREQ <- 3
build_tfidf <- function(df, text_col = "text_clean",
                        max_features = MAX_FEATURES,
                        min_doc_freq = MIN_DOC_FREQ,
                        blocklist    = BLOCKLIST_ALL) {
  if (!(text_col %in% names(df))) stop(sprintf("Column '%s' not found in data", text_col))
  docs <- data.frame(doc_id = seq_len(nrow(df)),
                     text   = tolower(ifelse(is.na(df[[text_col]]), "", df[[text_col]])),
                     stringsAsFactors = FALSE)
  bi <- docs %>%
    unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram)) %>%
    separate(bigram, c("w1", "w2"), sep = " ", remove = FALSE) %>%
    anti_join(stop_words, by = c("w1" = "word")) %>%
    anti_join(stop_words, by = c("w2" = "word")) %>%
    filter(str_detect(w1, "^[a-z]+$"), str_detect(w2, "^[a-z]+$"),
           nchar(w1) >= 2, nchar(w2) >= 2) %>%
    mutate(term = paste(wordStem(w1, "english"),
                        wordStem(w2, "english"))) %>%
    count(doc_id, term, name = "tf")
  all_terms <- bi
  if (length(blocklist) > 0) all_terms <- all_terms %>% filter(!term %in% blocklist)
  df_counts <- all_terms %>%
    distinct(doc_id, term) %>%
    count(term, name = "doc_freq") %>%
    filter(doc_freq >= min_doc_freq) %>%
    arrange(desc(doc_freq)) %>%
    slice_head(n = max_features)
  all_terms <- all_terms %>% filter(term %in% df_counts$term)
  tfidf <- all_terms %>% bind_tf_idf(term, doc_id, tf)
  term_levels <- sort(unique(tfidf$term))
  X <- sparseMatrix(
    i = tfidf$doc_id,
    j = match(tfidf$term, term_levels),
    x = tfidf$tf_idf,
    dims = c(nrow(df), length(term_levels)),
    dimnames = list(NULL, term_levels)
  )
  list(X = as.matrix(X), terms = term_levels)
}

# =============================================================================
# STEP 20 - RQ1 RF: Lasso -> balanced ranger (single spec, descriptive)
# =============================================================================
# Why this exists: descriptive comparison of dictionary vs RF on RQ1 only.
# Target = sign(bps_change) in {-1, 0, +1} = {cut, hold, hike}.
# Same-meeting prediction (in-sample / OOB), NOT a forecast.
# Output: rf_score = P(hike) - P(cut) from OOB predictions.
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 20: RQ1 RF training - Lasso -> balanced ranger (single spec)\n")
cat("==================================================================\n")

make_balanced_weights <- function(y) {
  tab <- table(y); tab <- tab[tab > 0]
  per_obs <- 1 / as.numeric(tab[as.character(y)])
  per_obs * length(y) / sum(per_obs)
}

train_rf_lasso <- function(df, corpus_name,
                           ntree        = RF_NTREE,
                           holdout_frac = 0.20) {
  t0 <- Sys.time()
  cat(sprintf("\n-- %s [Lasso -> balanced RF] -- %s\n",
              corpus_name, format(t0, "%H:%M:%S")))
  
  df <- df %>%
    filter(!is.na(bps_change), !is.na(text_clean),
           nchar(text_clean) > 100) %>%
    arrange(meeting_date)
  df$decision_sign <- factor(sign(df$bps_change),
                             levels = c(-1, 0, 1),
                             labels = c("cut", "hold", "hike"))
  cat(sprintf("  Training sample: %d docs\n", nrow(df)))
  cat("  Class distribution: "); print(table(df$decision_sign))
  
  y_full <- df$decision_sign
  tfidf  <- build_tfidf(df, text_col = "text_clean")
  X_full <- tfidf$X
  cat(sprintf("  TF-IDF matrix: %d docs x %d bigrams\n", nrow(X_full), ncol(X_full)))
  
  cv_fit <- tryCatch(
    cv.glmnet(x = X_full, y = y_full, family = "multinomial",
              alpha = 1, nfolds = 10, type.measure = "class",
              standardize = TRUE),
    error = function(e) { cat(sprintf("  cv.glmnet failed: %s\n", e$message)); NULL }
  )
  if (is.null(cv_fit)) return(NULL)
  
  best_lambda <- cv_fit$lambda.1se
  coefs <- coef(cv_fit, s = best_lambda)
  lasso_terms <- character(0)
  for (cl in names(coefs)) {
    cmat <- as.matrix(coefs[[cl]])
    cmat <- cmat[rownames(cmat) != "(Intercept)", , drop = FALSE]
    lasso_terms <- union(lasso_terms, rownames(cmat)[cmat[, 1] != 0])
  }
  if (length(lasso_terms) < 5) {
    cat("  WARNING: <5 features at lambda.1se - falling back to lambda.min\n")
    best_lambda <- cv_fit$lambda.min
    coefs <- coef(cv_fit, s = best_lambda)
    lasso_terms <- character(0)
    for (cl in names(coefs)) {
      cmat <- as.matrix(coefs[[cl]])
      cmat <- cmat[rownames(cmat) != "(Intercept)", , drop = FALSE]
      lasso_terms <- union(lasso_terms, rownames(cmat)[cmat[, 1] != 0])
    }
  }
  cat(sprintf("  Lasso selected: %d features (feature/doc ratio = %.2f)\n",
              length(lasso_terms), length(lasso_terms) / nrow(df)))
  if (length(lasso_terms) < 2) {
    cat("  STOP: <2 Lasso features - aborting this corpus\n")
    return(NULL)
  }
  
  keep_idx <- which(tfidf$terms %in% lasso_terms)
  X        <- X_full[, keep_idx, drop = FALSE]
  terms    <- tfidf$terms[keep_idx]
  clean_names <- make.names(terms, unique = TRUE)
  colnames(X) <- clean_names
  y  <- y_full
  cw <- make_balanced_weights(y)
  cat(sprintf("  RF feature matrix: %d docs x %d features  (case.weights %.3f-%.3f)\n",
              nrow(X), ncol(X), min(cw), max(cw)))
  
  n         <- nrow(df)
  n_train   <- floor((1 - holdout_frac) * n)
  train_idx <- seq_len(n_train)
  test_idx  <- setdiff(seq_len(n), train_idx)
  
  rf_full <- ranger(x = X, y = y, num.trees = ntree, probability = TRUE,
                    case.weights = cw, importance = "impurity",
                    num.threads = RF_THREADS, seed = 42, verbose = FALSE)
  
  oob_probs   <- rf_full$predictions
  class_names <- colnames(oob_probs)
  oob_pred    <- factor(class_names[max.col(oob_probs, ties.method = "first")],
                        levels = levels(y))
  oob_acc <- mean(oob_pred == y, na.rm = TRUE)
  cm_full <- table(actual = y, predicted = oob_pred)
  per_class_recall <- diag(cm_full) / rowSums(cm_full)
  per_class_recall[is.nan(per_class_recall)] <- NA
  balanced_acc <- mean(per_class_recall, na.rm = TRUE)
  cat(sprintf("  OOB accuracy: %.3f  |  Balanced accuracy: %.3f\n",
              oob_acc, balanced_acc))
  cat("  OOB confusion matrix:\n"); print(cm_full)
  
  holdout_acc <- NA
  if (length(test_idx) >= 5) {
    y_tr <- y[train_idx]
    nne_tr <- table(y_tr); nne_tr <- nne_tr[nne_tr > 0]
    if (length(nne_tr) >= 2) {
      cw_tr <- make_balanced_weights(y_tr)
      rf_train <- ranger(x = X[train_idx, , drop = FALSE], y = y_tr,
                         num.trees = ntree, probability = TRUE,
                         case.weights = cw_tr, importance = "none",
                         num.threads = RF_THREADS, seed = 42, verbose = FALSE)
      pred_prob <- predict(rf_train, X[test_idx, , drop = FALSE])$predictions
      pred_test <- factor(colnames(pred_prob)[max.col(pred_prob, ties.method = "first")],
                          levels = levels(y))
      holdout_acc <- mean(pred_test == y[test_idx])
      cat(sprintf("  Holdout accuracy: %.3f\n", holdout_acc))
    }
  }
  
  get_col <- function(m, col_name)
    if (col_name %in% colnames(m)) m[, col_name] else rep(0, nrow(m))
  df$rf_p_cut  <- get_col(oob_probs, "cut")
  df$rf_p_hold <- get_col(oob_probs, "hold")
  df$rf_p_hike <- get_col(oob_probs, "hike")
  df$rf_score  <- df$rf_p_hike - df$rf_p_cut
  df$rf_label  <- factor(class_names[max.col(oob_probs, ties.method = "first")],
                         levels = c("cut", "hold", "hike"))
  
  imp_vec <- rf_full$variable.importance
  imp_df  <- data.frame(term_stem = names(imp_vec),
                        MeanDecreaseGini = as.numeric(imp_vec),
                        stringsAsFactors = FALSE)
  imp_df$term_original <- terms[match(imp_df$term_stem, clean_names)]
  imp_df <- imp_df %>% arrange(desc(MeanDecreaseGini))
  
  cat(sprintf("-- %s DONE in %.1fs\n", corpus_name,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  
  list(df = df, model = rf_full, terms = terms, clean_names = clean_names,
       importance = imp_df, lasso_terms = lasso_terms, lasso_lambda = best_lambda,
       oob_acc = oob_acc, balanced_acc = balanced_acc,
       per_class_recall = per_class_recall, holdout_acc = holdout_acc,
       spec = "Lasso + balanced RF")
}

T_RF_START <- Sys.time()
rf_fomc_stmt <- train_rf_lasso(fed_stmt_reg, "FOMC STATEMENTS")
rf_ecb_stmt  <- train_rf_lasso(ecb_stmt_reg, "ECB STATEMENTS")
rf_fomc_pc   <- train_rf_lasso(fed_pc_reg,   "FOMC PRESS CONFERENCES")
rf_ecb_pc    <- train_rf_lasso(ecb_pc_reg,   "ECB PRESS CONFERENCES")
cat(sprintf("\n  All RQ1 RFs done in %.1fs\n",
            as.numeric(difftime(Sys.time(), T_RF_START, units = "secs"))))

# =============================================================================
# STEP 21 - Feature importance: top hawkish / dovish bigrams
# =============================================================================
cat("\nStep 21: Computing directional feature importance...\n")
compute_directional_importance <- function(rf_result, corpus_name, top_n = 30) {
  df    <- rf_result$df
  terms <- rf_result$terms
  tfidf <- build_tfidf(df, text_col = "text_clean")
  X     <- tfidf$X
  common <- intersect(tfidf$terms, terms)
  X_aligned <- X[, common, drop = FALSE]
  y <- df$decision_sign
  class_means <- sapply(levels(y), function(cl) {
    idx <- which(y == cl)
    if (length(idx) == 0) return(rep(0, ncol(X_aligned)))
    colMeans(X_aligned[idx, , drop = FALSE])
  })
  rownames(class_means) <- colnames(X_aligned)
  if (!("hike" %in% colnames(class_means))) {
    direction <- rep(0, nrow(class_means)); names(direction) <- rownames(class_means)
  } else if (!("cut" %in% colnames(class_means))) {
    direction <- class_means[, "hike"]
  } else {
    direction <- class_means[, "hike"] - class_means[, "cut"]
  }
  imp <- rf_result$importance
  imp$direction <- direction[match(imp$term_original, names(direction))]
  top_hawkish <- imp %>% filter(!is.na(direction), direction > 0) %>%
    arrange(desc(MeanDecreaseGini * pmax(direction, 0))) %>% slice_head(n = top_n) %>%
    select(term = term_original, importance_gini = MeanDecreaseGini, direction)
  top_dovish <- imp %>% filter(!is.na(direction), direction < 0) %>%
    arrange(desc(MeanDecreaseGini * pmax(-direction, 0))) %>% slice_head(n = top_n) %>%
    select(term = term_original, importance_gini = MeanDecreaseGini, direction)
  cat(sprintf("\n-- %s - TOP %d HAWKISH bigrams --\n", corpus_name, top_n))
  print(as.data.frame(top_hawkish) %>%
          mutate(importance_gini = round(importance_gini, 3),
                 direction = round(direction, 5)), row.names = FALSE)
  cat(sprintf("\n-- %s - TOP %d DOVISH bigrams --\n", corpus_name, top_n))
  print(as.data.frame(top_dovish) %>%
          mutate(importance_gini = round(importance_gini, 3),
                 direction = round(direction, 5)), row.names = FALSE)
  list(hawkish = top_hawkish, dovish = top_dovish, all = imp)
}
imp_fomc_stmt <- compute_directional_importance(rf_fomc_stmt, "FOMC STMT")
imp_fomc_pc   <- compute_directional_importance(rf_fomc_pc,   "FOMC PC")
imp_ecb_stmt  <- compute_directional_importance(rf_ecb_stmt,  "ECB STMT")
imp_ecb_pc    <- compute_directional_importance(rf_ecb_pc,    "ECB PC")
all_imp <- bind_rows(
  imp_fomc_stmt$all %>% mutate(corpus = "fomc_stmt"),
  imp_fomc_pc$all   %>% mutate(corpus = "fomc_pc"),
  imp_ecb_stmt$all  %>% mutate(corpus = "ecb_stmt"),
  imp_ecb_pc$all    %>% mutate(corpus = "ecb_pc")
)
write_csv(all_imp, file.path(output_dir, "rf_feature_importance_v30.csv"))
cat("  Saved -> rf_feature_importance_v30.csv\n")

# Attach RQ1 RF scores back to per-corpus dataframes for later use ---------
attach_rf_scores <- function(rf_obj, corpus_name) {
  if (is.null(rf_obj)) {
    cat(sprintf("  %s: rf_obj is NULL - skipping\n", corpus_name)); return(NULL)
  }
  df <- rf_obj$df
  df$spec_used <- rf_obj$spec
  cat(sprintf("  %s: %d rows scored with rf_score\n", corpus_name, nrow(df)))
  df
}
cat("\n  Attaching RQ1 RF scores per corpus:\n")
fed_stmt_rf <- attach_rf_scores(rf_fomc_stmt, "FOMC STMT")
fed_pc_rf   <- attach_rf_scores(rf_fomc_pc,   "FOMC PC  ")
ecb_stmt_rf <- attach_rf_scores(rf_ecb_stmt,  "ECB STMT ")
ecb_pc_rf   <- attach_rf_scores(rf_ecb_pc,    "ECB PC   ")

# Re-add chair/era (some merge ops may drop them)
fed_stmt_rf <- add_chair_era(fed_stmt_rf)
fed_pc_rf   <- add_chair_era(fed_pc_rf)
ecb_stmt_rf <- add_chair_era(ecb_stmt_rf)
ecb_pc_rf   <- add_chair_era(ecb_pc_rf)

# RQ1 RF performance summary
perf_rq1 <- data.frame(
  corpus      = c("FOMC stmt", "FOMC PC", "ECB stmt", "ECB PC"),
  spec_used   = c(rf_fomc_stmt$spec, rf_fomc_pc$spec, rf_ecb_stmt$spec, rf_ecb_pc$spec),
  n           = c(nrow(rf_fomc_stmt$df), nrow(rf_fomc_pc$df),
                  nrow(rf_ecb_stmt$df),  nrow(rf_ecb_pc$df)),
  oob_acc     = round(c(rf_fomc_stmt$oob_acc, rf_fomc_pc$oob_acc,
                        rf_ecb_stmt$oob_acc,  rf_ecb_pc$oob_acc), 3),
  bal_acc     = round(c(rf_fomc_stmt$balanced_acc, rf_fomc_pc$balanced_acc,
                        rf_ecb_stmt$balanced_acc,  rf_ecb_pc$balanced_acc), 3),
  holdout_acc = round(c(rf_fomc_stmt$holdout_acc, rf_fomc_pc$holdout_acc,
                        rf_ecb_stmt$holdout_acc,  rf_ecb_pc$holdout_acc), 3)
)
cat("\n-- RQ1 RF performance --\n"); print(perf_rq1, row.names = FALSE)
write_csv(perf_rq1, file.path(output_dir, "rf_rq1_performance_v30.csv"))
saveRDS(list(fomc_stmt = rf_fomc_stmt, fomc_pc = rf_fomc_pc,
             ecb_stmt = rf_ecb_stmt, ecb_pc = rf_ecb_pc),
        "rf_rq1_models_v30.rds")
cat("  Saved -> rf_rq1_performance_v30.csv  +  rf_rq1_models_v30.rds\n")

# =============================================================================
# STEP 22 - RQ2 CALENDARS  (NEW: bps_next2, y2_next2 in addition to t+1)
# =============================================================================
# For each central bank, build a per-meeting-date calendar with:
#   bps_curr   = bps_change at meeting t
#   bps_next   = bps_change at meeting t+1   (decision one meeting ahead)
#   bps_next2  = bps_change at meeting t+2   (decision two meetings ahead) NEW
#   bps_lag1   = bps_change at meeting t-1
#   y2_curr    = 2Y yield (UST2Y / OIS_MP2) at meeting t
#   y2_next    = 2Y yield at meeting t+1
#   y2_next2   = 2Y yield at meeting t+2     (NEW)
#   y2_lag1    = 2Y yield at meeting t-1
#
# We build the calendar from the STATEMENT-window dataframes (one row per
# meeting), then JOIN it onto every per-corpus dataframe by meeting_date.
# That way FOMC PC / ECB PC inherit the same calendar as their statements.
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 22: RQ2 calendars (bps_next, bps_next2, y2_next, y2_next2)\n")
cat("==================================================================\n")

fomc_cal <- fed_stmt_reg %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  filter(!is.na(bps_change)) %>%
  distinct(meeting_date, bps_change, UST2Y) %>%
  arrange(meeting_date) %>%
  transmute(
    meeting_date,
    bps_curr  = bps_change,
    bps_next  = lead(bps_change, 1L),
    bps_next2 = lead(bps_change, 2L),
    bps_lag1  = lag(bps_change, 1L),
    y2_curr   = UST2Y,
    y2_next   = lead(UST2Y, 1L),
    y2_next2  = lead(UST2Y, 2L),
    y2_lag1   = lag(UST2Y, 1L)
  )

ecb_cal <- ecb_stmt_reg %>%
  mutate(meeting_date = as.Date(meeting_date)) %>%
  filter(!is.na(bps_change)) %>%
  distinct(meeting_date, bps_change, OIS_MP2) %>%
  arrange(meeting_date) %>%
  transmute(
    meeting_date,
    bps_curr  = bps_change,
    bps_next  = lead(bps_change, 1L),
    bps_next2 = lead(bps_change, 2L),
    bps_lag1  = lag(bps_change, 1L),
    y2_curr   = OIS_MP2,
    y2_next   = lead(OIS_MP2, 1L),
    y2_next2  = lead(OIS_MP2, 2L),
    y2_lag1   = lag(OIS_MP2, 1L)
  )

cat(sprintf("  FOMC calendar: %d meetings (bps_next non-NA: %d, bps_next2: %d, y2_next: %d, y2_next2: %d)\n",
            nrow(fomc_cal),
            sum(!is.na(fomc_cal$bps_next)),  sum(!is.na(fomc_cal$bps_next2)),
            sum(!is.na(fomc_cal$y2_next)),   sum(!is.na(fomc_cal$y2_next2))))
cat(sprintf("  ECB  calendar: %d meetings (bps_next non-NA: %d, bps_next2: %d, y2_next: %d, y2_next2: %d)\n",
            nrow(ecb_cal),
            sum(!is.na(ecb_cal$bps_next)),  sum(!is.na(ecb_cal$bps_next2)),
            sum(!is.na(ecb_cal$y2_next)),   sum(!is.na(ecb_cal$y2_next2))))

# Join calendars onto per-corpus reg dataframes -> these are the "RQ2"
# input frames the expanding-window RFs will train on.
attach_calendar <- function(reg_df, cal_df, label) {
  # If columns from a previous run exist, drop them first to avoid .x/.y suffixes
  cal_cols <- setdiff(names(cal_df), "meeting_date")
  to_drop  <- intersect(cal_cols, names(reg_df))
  if (length(to_drop) > 0) reg_df <- reg_df %>% select(-all_of(to_drop))
  out <- reg_df %>%
    mutate(meeting_date = as.Date(meeting_date)) %>%
    left_join(cal_df, by = "meeting_date")
  cat(sprintf("  %s: bps_next non-NA = %d / %d  |  y2_next non-NA = %d / %d\n",
              label,
              sum(!is.na(out$bps_next)), nrow(out),
              sum(!is.na(out$y2_next)),  nrow(out)))
  out
}
fed_stmt_rq2 <- attach_calendar(fed_stmt_reg, fomc_cal, "FOMC stmt")
fed_pc_rq2   <- attach_calendar(fed_pc_reg,   fomc_cal, "FOMC PC  ")
ecb_stmt_rq2 <- attach_calendar(ecb_stmt_reg, ecb_cal,  "ECB stmt ")
ecb_pc_rq2   <- attach_calendar(ecb_pc_reg,   ecb_cal,  "ECB PC   ")

# Sanity check: each rq2 frame must carry text_clean AND text_fwd
for (df_name in c("fed_stmt_rq2","fed_pc_rq2","ecb_stmt_rq2","ecb_pc_rq2")) {
  d <- get(df_name)
  cat(sprintf("    %s: text_clean ok=%s, text_fwd ok=%s, n>100 chars: clean=%d / fwd=%d\n",
              df_name,
              "text_clean" %in% names(d),
              "text_fwd"   %in% names(d),
              sum(!is.na(d$text_clean) & nchar(d$text_clean) > 100),
              sum(!is.na(d$text_fwd)   & nchar(d$text_fwd)   > 100)))
}

# =============================================================================
# STEP 23 - GENERALIZED EXPANDING-WINDOW LASSO->RF FUNCTION
# =============================================================================
# Identical structure to v25 train_rf_y2_rq2_expanding, but the target column
# is now a parameter so we can use it for bps_next, bps_next2, y2_next, y2_next2.
#
# At each meeting t in the sorted dataframe:
#   1. Take training docs 1..(t-1) where target_col is NOT NA.
#   2. Every `lasso_refit_every` meetings, refit cv.glmnet on TF-IDF features
#      built on the training docs only, get the active feature set.
#      (Caching keeps total runtime sane; RF still refits every meeting.)
#   3. Build TF-IDF on training+test together restricted to those features.
#   4. Fit a regression ranger on training rows, predict the test row.
#   5. Store the prediction; the model is discarded after.
#
# Output: column rf_pred_<target> on the returned dataframe + perf summary.
# =============================================================================
cat("\nStep 23: Defining generalized expanding-window Lasso->RF function...\n")

train_rf_xw <- function(df, corpus_name, target_col,
                        text_col          = "text_clean",
                        min_train         = 30,
                        lasso_refit_every = 4,
                        ntree             = RF_NTREE,
                        lambda_choice     = "1se") {
  t_start <- Sys.time()
  cat(sprintf("\n  XW: %s | target=%s | text=%s\n",
              corpus_name, target_col, text_col))
  cat(sprintf("    min_train=%d, lasso_refit_every=%d, ntree=%d\n",
              min_train, lasso_refit_every, ntree))
  
  d <- df %>%
    filter(!is.na(.data[[target_col]]),
           !is.na(.data[[text_col]]),
           nchar(.data[[text_col]]) > 100) %>%
    arrange(meeting_date)
  N <- nrow(d)
  cat(sprintf("    Total usable docs (target+text non-NA): %d\n", N))
  if (N < min_train + 5) {
    cat(sprintf("    Too few docs (%d) - skipping\n", N))
    return(NULL)
  }
  
  pred_col <- paste0("rf_pred_", target_col)
  d[[pred_col]] <- NA_real_
  log_rows <- vector("list", N)
  cached_lasso_terms <- character(0)
  last_lasso_step    <- -Inf
  
  for (t in seq.int(min_train + 1, N)) {
    train_d  <- d[seq_len(t - 1L), , drop = FALSE]
    test_row <- d[t, , drop = FALSE]
    y_train  <- train_d[[target_col]]
    actual_t <- test_row[[target_col]]
    
    refit_lasso <- (t == min_train + 1L) ||
      ((t - last_lasso_step) >= lasso_refit_every) ||
      (length(cached_lasso_terms) < 5)
    
    if (refit_lasso) {
      tfidf_train <- build_tfidf(train_d, text_col = text_col)
      X_tr <- tfidf_train$X
      if (ncol(X_tr) < 5 || nrow(X_tr) < 20) {
        log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                    n_features = NA, pred = NA, actual = actual_t,
                                    refit = TRUE, status = "degenerate_X",
                                    stringsAsFactors = FALSE)
        next
      }
      cv_fit <- tryCatch(
        cv.glmnet(x = X_tr, y = y_train, family = "gaussian",
                  alpha = 1, nfolds = 10, type.measure = "mse",
                  standardize = TRUE),
        error = function(e) NULL
      )
      if (is.null(cv_fit)) {
        log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                    n_features = NA, pred = NA, actual = actual_t,
                                    refit = TRUE, status = "lasso_failed",
                                    stringsAsFactors = FALSE)
        next
      }
      best_lambda <- if (lambda_choice == "1se") cv_fit$lambda.1se else cv_fit$lambda.min
      coefs <- as.matrix(coef(cv_fit, s = best_lambda))
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      lasso_terms <- rownames(coefs)[coefs[, 1] != 0]
      if (length(lasso_terms) < 5) {
        coefs <- as.matrix(coef(cv_fit, s = cv_fit$lambda.min))
        coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
        lasso_terms <- rownames(coefs)[coefs[, 1] != 0]
      }
      cached_lasso_terms <- lasso_terms
      last_lasso_step    <- t
    }
    
    if (length(cached_lasso_terms) < 2) {
      log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                  n_features = length(cached_lasso_terms),
                                  pred = NA, actual = actual_t,
                                  refit = refit_lasso, status = "no_features",
                                  stringsAsFactors = FALSE)
      next
    }
    
    combined_t <- rbind(train_d, test_row)
    tfidf_all  <- build_tfidf(combined_t, text_col = text_col)
    X_all      <- tfidf_all$X
    terms_all  <- tfidf_all$terms
    keep_idx   <- which(terms_all %in% cached_lasso_terms)
    if (length(keep_idx) < 2) {
      log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                  n_features = length(keep_idx),
                                  pred = NA, actual = actual_t,
                                  refit = refit_lasso, status = "no_match",
                                  stringsAsFactors = FALSE)
      next
    }
    X_all_lasso <- X_all[, keep_idx, drop = FALSE]
    colnames(X_all_lasso) <- make.names(terms_all[keep_idx], unique = TRUE)
    X_train <- X_all_lasso[seq_len(nrow(train_d)), , drop = FALSE]
    X_test  <- X_all_lasso[nrow(combined_t), , drop = FALSE]
    
    rf_t <- tryCatch(
      ranger(x = X_train, y = y_train, num.trees = ntree,
             importance = "none", num.threads = RF_THREADS,
             seed = 42, verbose = FALSE),
      error = function(e) NULL
    )
    if (is.null(rf_t)) {
      log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                  n_features = ncol(X_train),
                                  pred = NA, actual = actual_t,
                                  refit = refit_lasso, status = "rf_failed",
                                  stringsAsFactors = FALSE)
      next
    }
    pred_t <- predict(rf_t, X_test)$predictions
    d[[pred_col]][t] <- pred_t
    log_rows[[t]] <- data.frame(t = t, train_n = nrow(train_d),
                                n_features = ncol(X_train),
                                pred = pred_t, actual = actual_t,
                                refit = refit_lasso, status = "ok",
                                stringsAsFactors = FALSE)
    
    if (t %% 25 == 0 || t == N) {
      elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
      cat(sprintf("      [%d / %d]  train_n=%d  n_feat=%d  pred=%.4f  actual=%.4f  (%.0fs)\n",
                  t, N, nrow(train_d), ncol(X_train), pred_t, actual_t, elapsed))
    }
  }
  
  log_df <- bind_rows(log_rows)
  scored <- d %>% filter(!is.na(.data[[pred_col]]))
  n_fc   <- nrow(scored)
  if (n_fc >= 5) {
    y_act    <- scored[[target_col]]
    y_pred   <- scored[[pred_col]]
    ss_tot   <- sum((y_act - mean(y_act))^2)
    ss_res   <- sum((y_act - y_pred)^2)
    xw_r2    <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA
    xw_rmse  <- sqrt(mean((y_act - y_pred)^2))
    naive_rmse <- sd(y_act)
  } else {
    xw_r2 <- NA; xw_rmse <- NA; naive_rmse <- NA
  }
  total_time <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("    XW DONE %s [target=%s, text=%s]: n=%d  XW R^2=%.3f  XW RMSE=%.4f  naive RMSE=%.4f  (%.1fs)\n",
              corpus_name, target_col, text_col, n_fc, xw_r2, xw_rmse, naive_rmse, total_time))
  
  perf <- data.frame(
    corpus      = corpus_name,
    target      = target_col,
    text_input  = text_col,
    n_forecasts = n_fc,
    xw_r2       = round(xw_r2, 3),
    xw_rmse     = round(xw_rmse, 4),
    naive_rmse  = round(naive_rmse, 4),
    stringsAsFactors = FALSE
  )
  list(df = d, perf = perf, log = log_df,
       target_col = target_col, text_col = text_col,
       pred_col = pred_col)
}

# =============================================================================
# STEP 24 - RUN ALL EXPANDING-WINDOW RQ2 GRIDS
# =============================================================================
# Up to 4 targets x 4 corpora x 2 text inputs = 32 grids.
# Each ON target with both text inputs takes ~30-90 min per corpus.
# All results are stored in nested list xw_results so we can merge back.
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 24: Running expanding-window RQ2 grids\n")
cat("==================================================================\n")

# Targets to run (driven by the RUN_RQ2_* flags at top of file)
target_flags <- list(
  bps_next  = RUN_RQ2_BPS_NEXT,
  bps_next2 = RUN_RQ2_BPS_NEXT2,
  y2_next   = RUN_RQ2_Y2_NEXT,
  y2_next2  = RUN_RQ2_Y2_NEXT2
)
active_targets <- names(target_flags)[unlist(target_flags)]
cat(sprintf("  Active targets: %s\n", paste(active_targets, collapse = ", ")))

text_inputs <- c("text_clean")
if (RUN_RQ2_TEXT_FWD) text_inputs <- c(text_inputs, "text_fwd")
cat(sprintf("  Text inputs: %s\n", paste(text_inputs, collapse = ", ")))

corpora <- list(
  fomc_stmt = list(df = fed_stmt_rq2, label = "FOMC STMT"),
  fomc_pc   = list(df = fed_pc_rq2,   label = "FOMC PC"),
  ecb_stmt  = list(df = ecb_stmt_rq2, label = "ECB STMT"),
  ecb_pc    = list(df = ecb_pc_rq2,   label = "ECB PC")
)

# xw_results[[corpus_key]][[target]][[text_input]] = result list from train_rf_xw
xw_results <- list()
T_GRID_START <- Sys.time()
total_grids  <- length(corpora) * length(active_targets) * length(text_inputs)
cat(sprintf("  Total grids to run: %d\n\n", total_grids))

grid_idx <- 0L
for (cn in names(corpora)) {
  xw_results[[cn]] <- list()
  for (tg in active_targets) {
    xw_results[[cn]][[tg]] <- list()
    for (txt in text_inputs) {
      grid_idx <- grid_idx + 1L
      cat(sprintf("\n[grid %d / %d]  %s | target=%s | text=%s\n",
                  grid_idx, total_grids, corpora[[cn]]$label, tg, txt))
      res <- tryCatch(
        train_rf_xw(corpora[[cn]]$df,
                    corpus_name = corpora[[cn]]$label,
                    target_col  = tg,
                    text_col    = txt),
        error = function(e) {
          cat(sprintf("  ERROR in grid: %s\n", e$message)); NULL
        }
      )
      xw_results[[cn]][[tg]][[txt]] <- res
    }
  }
}
cat(sprintf("\n==================================================================\n"))
cat(sprintf("  ALL XW GRIDS DONE in %.1f min\n",
            as.numeric(difftime(Sys.time(), T_GRID_START, units = "mins"))))
cat("==================================================================\n")

# Performance summary across all grids
xw_perf_all <- bind_rows(lapply(names(xw_results), function(cn) {
  bind_rows(lapply(names(xw_results[[cn]]), function(tg) {
    bind_rows(lapply(names(xw_results[[cn]][[tg]]), function(txt) {
      r <- xw_results[[cn]][[tg]][[txt]]
      if (is.null(r)) return(NULL)
      r$perf %>% mutate(corpus_key = cn)
    }))
  }))
}))
xw_perf_all <- xw_perf_all %>%
  select(corpus_key, corpus, target, text_input, n_forecasts, xw_r2, xw_rmse, naive_rmse)
cat("\n-- XW PERFORMANCE SUMMARY --\n")
print(xw_perf_all, row.names = FALSE)
write_csv(xw_perf_all, file.path(output_dir, "rf_rq2_xw_performance_v30.csv"))

# Concatenate all per-step logs into one file (helpful for diagnosing skips)
xw_log_all <- bind_rows(lapply(names(xw_results), function(cn) {
  bind_rows(lapply(names(xw_results[[cn]]), function(tg) {
    bind_rows(lapply(names(xw_results[[cn]][[tg]]), function(txt) {
      r <- xw_results[[cn]][[tg]][[txt]]
      if (is.null(r) || is.null(r$log)) return(NULL)
      r$log %>% mutate(corpus = cn, target = tg, text_input = txt)
    }))
  }))
}))
write_csv(xw_log_all, file.path(output_dir, "rf_rq2_xw_log_v30.csv"))
cat("  Saved -> rf_rq2_xw_performance_v30.csv  +  rf_rq2_xw_log_v30.csv\n")

# Persist heavy result list (without docs to keep size sane)
xw_results_compact <- lapply(xw_results, function(by_corpus) {
  lapply(by_corpus, function(by_target) {
    lapply(by_target, function(r) {
      if (is.null(r)) return(NULL)
      list(perf = r$perf, log = r$log,
           target_col = r$target_col, text_col = r$text_col,
           pred_col = r$pred_col)
    })
  })
})
saveRDS(xw_results_compact, file.path(output_dir, "rf_rq2_xw_results_v30.rds"))
cat("  Saved -> rf_rq2_xw_results_v30.rds (compact, perf+log only)\n")

# =============================================================================
# STEP 25 - MERGE EXPANDING-WINDOW SCORES BACK INTO PER-CORPUS DATAFRAMES
# =============================================================================
# For each corpus, append columns named:
#    rf_pred_<target>_full   (from text_clean grid)
#    rf_pred_<target>_fwd    (from text_fwd   grid, if run)
# So the analyst can run regressions like:
#    bps_next ~ rf_pred_bps_next_full + rf_pred_bps_next_fwd + bps_curr
# directly on the per-corpus dataframe.
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 25: Merging XW scores back into per-corpus dataframes\n")
cat("==================================================================\n")

extract_pred_df <- function(res, suffix) {
  if (is.null(res)) return(NULL)
  pred_col <- res$pred_col
  out_col  <- paste0(pred_col, "_", suffix)
  res$df %>%
    mutate(event_id_merge = as.character(event_id)) %>%
    select(event_id_merge, !!out_col := all_of(pred_col))
}

merge_xw_into_corpus <- function(base_df, results_for_corpus, label) {
  out <- base_df %>% mutate(event_id_merge = as.character(event_id))
  for (tg in names(results_for_corpus)) {
    full_res <- results_for_corpus[[tg]][["text_clean"]]
    fwd_res  <- results_for_corpus[[tg]][["text_fwd"]]
    pf <- extract_pred_df(full_res, "full")
    if (!is.null(pf)) {
      drop_cols <- intersect(setdiff(names(pf), "event_id_merge"), names(out))
      if (length(drop_cols) > 0) out <- out %>% select(-all_of(drop_cols))
      out <- out %>% left_join(pf, by = "event_id_merge")
    }
    pf2 <- extract_pred_df(fwd_res, "fwd")
    if (!is.null(pf2)) {
      drop_cols <- intersect(setdiff(names(pf2), "event_id_merge"), names(out))
      if (length(drop_cols) > 0) out <- out %>% select(-all_of(drop_cols))
      out <- out %>% left_join(pf2, by = "event_id_merge")
    }
  }
  out <- out %>% select(-event_id_merge)
  cat(sprintf("  %s: merged %d new rf_pred_* columns\n",
              label, sum(grepl("^rf_pred_", names(out)))))
  out
}

fed_stmt_rq2_xw <- merge_xw_into_corpus(fed_stmt_rq2, xw_results$fomc_stmt, "FOMC stmt")
fed_pc_rq2_xw   <- merge_xw_into_corpus(fed_pc_rq2,   xw_results$fomc_pc,   "FOMC PC  ")
ecb_stmt_rq2_xw <- merge_xw_into_corpus(ecb_stmt_rq2, xw_results$ecb_stmt,  "ECB stmt ")
ecb_pc_rq2_xw   <- merge_xw_into_corpus(ecb_pc_rq2,   xw_results$ecb_pc,    "ECB PC   ")

# Pull rf_score (RQ1) and ALL the RQ2 rf_pred_* cols onto the rf-augmented frames
# so PHASE B has every score available on one frame per corpus.
attach_rf_score_from_rq1 <- function(rq2_xw_df, rq1_rf_df, label) {
  if (is.null(rq1_rf_df) || nrow(rq1_rf_df) == 0) {
    cat(sprintf("  %s: no RQ1 RF df - skipping rf_score join\n", label)); return(rq2_xw_df)
  }
  rf_small <- rq1_rf_df %>%
    mutate(event_id_merge = as.character(event_id)) %>%
    select(event_id_merge, rf_score, rf_p_cut, rf_p_hold, rf_p_hike, rf_label)
  out <- rq2_xw_df %>%
    mutate(event_id_merge = as.character(event_id)) %>%
    select(-any_of(c("rf_score","rf_p_cut","rf_p_hold","rf_p_hike","rf_label"))) %>%
    left_join(rf_small, by = "event_id_merge") %>%
    select(-event_id_merge)
  cat(sprintf("  %s: rf_score attached to %d / %d rows\n",
              label, sum(!is.na(out$rf_score)), nrow(out)))
  out
}
fed_stmt_rq2_xw <- attach_rf_score_from_rq1(fed_stmt_rq2_xw, fed_stmt_rf, "FOMC stmt")
fed_pc_rq2_xw   <- attach_rf_score_from_rq1(fed_pc_rq2_xw,   fed_pc_rf,   "FOMC PC  ")
ecb_stmt_rq2_xw <- attach_rf_score_from_rq1(ecb_stmt_rq2_xw, ecb_stmt_rf, "ECB stmt ")
ecb_pc_rq2_xw   <- attach_rf_score_from_rq1(ecb_pc_rq2_xw,   ecb_pc_rf,   "ECB PC   ")

# Re-add chair/era after all joins (some can drop them through factor casts)
fed_stmt_rq2_xw <- add_chair_era(fed_stmt_rq2_xw)
fed_pc_rq2_xw   <- add_chair_era(fed_pc_rq2_xw)
ecb_stmt_rq2_xw <- add_chair_era(ecb_stmt_rq2_xw)
ecb_pc_rq2_xw   <- add_chair_era(ecb_pc_rq2_xw)

write_csv(fed_stmt_rq2_xw, file.path(output_dir, "fed_stmt_reg_v2_rq2_xw_v30.csv"))
write_csv(fed_pc_rq2_xw, file.path(output_dir, "fed_pc_reg_v2_rq2_xw_v30.csv"))
write_csv(ecb_stmt_rq2_xw, file.path(output_dir, "ecb_stmt_reg_v2_rq2_xw_v30.csv"))
write_csv(ecb_pc_rq2_xw, file.path(output_dir, "ecb_pc_reg_v2_rq2_xw_v30.csv"))
cat("  Saved 4x *_rq2_xw_v30.csv\n")

# =============================================================================
# ============================  PHASE B  ======================================
#  All regressions, plots, and output files live below.
#  Up to here the pipeline only built MODELS / SCORES.
# =============================================================================
cat("\n\n##################################################################\n")
cat("##                    PHASE B - REGRESSIONS / OUTPUTS              ##\n")
cat("##################################################################\n\n")

# Accumulator for FE ladder rows across all calls below
fe_results <- list()
add_fe <- function(x) { if (!is.null(x)) fe_results[[length(fe_results) + 1L]] <<- x$rows }

# =============================================================================
# STEP 26 - MERGE BERT SCORES
# =============================================================================
# Expects bert_scores_<corpus>.csv files in working dir, with columns:
#   meeting_date, n_sentences, n_stance, bert_p_hawk, bert_p_dove, bert_p_neutral,
#   bert_score, bert_score_dir, bert_label
# and optionally _fwd variants. Missing files are skipped quietly.
# =============================================================================
cat("Step 26: Merging BERT scores...\n")

load_and_merge_bert <- function(reg_df, bert_path, label) {
  if (!file.exists(bert_path)) {
    cat(sprintf("  %s: %s not found - skipping\n", label, bert_path))
    return(reg_df)
  }
  bd <- tryCatch(read_csv(bert_path, show_col_types = FALSE),
                 error = function(e) { cat(sprintf("  %s: read failed (%s)\n",
                                                   label, e$message)); NULL })
  if (is.null(bd) || nrow(bd) == 0) return(reg_df)
  if (!"meeting_date" %in% names(bd)) {
    cat(sprintf("  %s: no meeting_date col in %s - skipping\n", label, bert_path))
    return(reg_df)
  }
  bd <- bd %>% mutate(meeting_date = as.Date(meeting_date))
  bert_cols <- setdiff(names(bd), c("meeting_date", "event_id", "event_type",
                                    "central_bank", "source", "url", "text_raw",
                                    "text_clean", "text_fwd", "text_curr"))
  bd <- bd %>% select(meeting_date, all_of(bert_cols)) %>%
    distinct(meeting_date, .keep_all = TRUE)
  drop_cols <- intersect(bert_cols, names(reg_df))
  if (length(drop_cols) > 0) reg_df <- reg_df %>% select(-all_of(drop_cols))
  out <- reg_df %>%
    mutate(meeting_date = as.Date(meeting_date)) %>%
    left_join(bd, by = "meeting_date")
  bert_score_present <- "bert_score" %in% names(out)
  cat(sprintf("  %s: BERT cols added (%d); bert_score non-NA: %d / %d\n",
              label, length(bert_cols),
              if (bert_score_present) sum(!is.na(out$bert_score)) else 0L,
              nrow(out)))
  out
}

fed_stmt_rq2_xw <- load_and_merge_bert(fed_stmt_rq2_xw, file.path(output_dir, "bert_scores_fomc_stmt.csv"), "FOMC stmt")
fed_pc_rq2_xw   <- load_and_merge_bert(fed_pc_rq2_xw,   "bert_scores_fomc_pc.csv",   "FOMC PC ")
ecb_stmt_rq2_xw <- load_and_merge_bert(ecb_stmt_rq2_xw, file.path(output_dir, "bert_scores_ecb_stmt.csv"),  "ECB stmt ")
ecb_pc_rq2_xw   <- load_and_merge_bert(ecb_pc_rq2_xw,   "bert_scores_ecb_pc.csv",    "ECB PC   ")

# Convenience aliases for the RQ1-only frames (no XW cols, but with rf_score+BERT)
fed_stmt_with_bert <- fed_stmt_rq2_xw
fed_pc_with_bert   <- fed_pc_rq2_xw
ecb_stmt_with_bert <- ecb_stmt_rq2_xw
ecb_pc_with_bert   <- ecb_pc_rq2_xw

write_csv(fed_stmt_with_bert, file.path(output_dir, "fed_stmt_with_bert_v30.csv"))
write_csv(fed_pc_with_bert, file.path(output_dir, "fed_pc_with_bert_v30.csv"))
write_csv(ecb_stmt_with_bert, file.path(output_dir, "ecb_stmt_with_bert_v30.csv"))
write_csv(ecb_pc_with_bert, file.path(output_dir, "ecb_pc_with_bert_v30.csv"))
cat("  Saved 4x *_with_bert_v30.csv\n")

# Quick BERT diagnostic: correlation between bert_score and rf_score / net_score
diag_rows <- list()
for (lab in c("fomc_stmt", "fomc_pc", "ecb_stmt", "ecb_pc")) {
  d <- get(switch(lab,
                  fomc_stmt = "fed_stmt_with_bert",
                  fomc_pc   = "fed_pc_with_bert",
                  ecb_stmt  = "ecb_stmt_with_bert",
                  ecb_pc    = "ecb_pc_with_bert"))
  if (!"bert_score" %in% names(d)) next
  d2 <- d %>% filter(!is.na(bert_score), !is.na(net_score))
  if (nrow(d2) < 10) next
  rho_rf  <- if ("rf_score" %in% names(d2) && sum(!is.na(d2$rf_score)) >= 10)
    cor(d2$bert_score, d2$rf_score,  use = "pairwise.complete.obs") else NA
  rho_dic <- cor(d2$bert_score, d2$net_score, use = "pairwise.complete.obs")
  diag_rows[[lab]] <- data.frame(corpus = lab, n = nrow(d2),
                                 cor_bert_dict = round(rho_dic, 3),
                                 cor_bert_rf   = round(rho_rf,  3))
}
if (length(diag_rows) > 0) {
  bert_corr <- bind_rows(diag_rows)
  cat("\n-- BERT vs dict / RF correlations --\n")
  print(bert_corr, row.names = FALSE)
  write_csv(bert_corr, file.path(output_dir, "bert_diagnostic_correlations.csv"))
}

# =============================================================================
# STEP 27 - RQ1 DICTIONARY REGRESSIONS  (naive, controlled, FE, fwd/curr, DM)
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 27: RQ1 dictionary regressions\n")
cat("==================================================================\n")

# -- 27a) Naive: outcome ~ net_score ----------------------------------------
cat("\n-- 27a) NAIVE: outcome ~ net_score --\n")
run_ols(SP500  ~ net_score, fed_stmt_reg, "FOMC stmt | SP500  ~ net_score")
run_ols(EURUSD ~ net_score, fed_stmt_reg, "FOMC stmt | EURUSD ~ net_score")
run_ols(UST2Y  ~ net_score, fed_stmt_reg, "FOMC stmt | UST2Y  ~ net_score")
run_ols(UST10Y ~ net_score, fed_stmt_reg, "FOMC stmt | UST10Y ~ net_score")
run_ols(SX5E   ~ net_score, ecb_stmt_reg, "ECB  stmt | SX5E   ~ net_score")
run_ols(EURUSD ~ net_score, ecb_stmt_reg, "ECB  stmt | EURUSD ~ net_score")
run_ols(OIS_MP2  ~ net_score, ecb_stmt_reg, "ECB stmt | OIS_MP2  ~ net_score")
run_ols(OIS_10Y  ~ net_score, ecb_stmt_reg, "ECB stmt | OIS_10Y  ~ net_score")

# -- 27b) Controlled: + monetary surprise ------------------------------------
cat("\n-- 27b) CONTROLLED: outcome ~ net_score + MP/OIS surprise --\n")
run_ols(SP500  ~ net_score + MP1, fed_stmt_reg, "FOMC stmt | SP500  ~ net_score + MP1")
run_ols(EURUSD ~ net_score + MP1, fed_stmt_reg, "FOMC stmt | EURUSD ~ net_score + MP1")
run_ols(UST2Y  ~ net_score + MP1, fed_stmt_reg, "FOMC stmt | UST2Y  ~ net_score + MP1")
run_ols(UST10Y ~ net_score + MP1, fed_stmt_reg, "FOMC stmt | UST10Y ~ net_score + MP1")
run_ols(SX5E   ~ net_score + OIS_MP1, ecb_stmt_reg, "ECB stmt | SX5E   ~ net_score + OIS_MP1")
run_ols(EURUSD ~ net_score + OIS_MP1, ecb_stmt_reg, "ECB stmt | EURUSD ~ net_score + OIS_MP1")
run_ols(OIS_MP2  ~ net_score + OIS_MP1, ecb_stmt_reg, "ECB stmt | OIS_MP2  ~ net_score + OIS_MP1")
run_ols(OIS_10Y  ~ net_score + OIS_MP1, ecb_stmt_reg, "ECB stmt | OIS_10Y  ~ net_score + OIS_MP1")

# -- 27c) FE ladder for net_score on key outcomes ---------------------------
cat("\n-- 27c) FE ladder: outcome ~ net_score [+ chair / + chair+era] --\n")
add_fe(run_fe_ladder_table("SP500",  "net_score + MP1",       fed_stmt_reg, "FOMC stmt | SP500"))
add_fe(run_fe_ladder_table("EURUSD", "net_score + MP1",       fed_stmt_reg, "FOMC stmt | EURUSD"))
add_fe(run_fe_ladder_table("UST2Y",  "net_score + MP1",       fed_stmt_reg, "FOMC stmt | UST2Y"))
add_fe(run_fe_ladder_table("UST10Y", "net_score + MP1",       fed_stmt_reg, "FOMC stmt | UST10Y"))
add_fe(run_fe_ladder_table("SX5E",   "net_score + OIS_MP1",   ecb_stmt_reg, "ECB stmt | SX5E"))
add_fe(run_fe_ladder_table("EURUSD", "net_score + OIS_MP1",   ecb_stmt_reg, "ECB stmt | EURUSD"))
add_fe(run_fe_ladder_table("OIS_MP2",  "net_score + OIS_MP1", ecb_stmt_reg, "ECB stmt | OIS_MP2"))
add_fe(run_fe_ladder_table("OIS_10Y",  "net_score + OIS_MP1", ecb_stmt_reg, "ECB stmt | OIS_10Y"))

# -- 27d) Forward/current split ---------------------------------------------
cat("\n-- 27d) FWD/CURR split: outcome ~ net_score_fwd + net_score_curr --\n")
run_ols(SP500  ~ net_score_fwd + net_score_curr + MP1, fed_stmt_reg, "FOMC stmt | SP500  ~ fwd+curr +MP1")
run_ols(UST2Y  ~ net_score_fwd + net_score_curr + MP1, fed_stmt_reg, "FOMC stmt | UST2Y  ~ fwd+curr +MP1")
run_ols(UST10Y ~ net_score_fwd + net_score_curr + MP1, fed_stmt_reg, "FOMC stmt | UST10Y ~ fwd+curr +MP1")
run_ols(SX5E   ~ net_score_fwd + net_score_curr + OIS_MP1, ecb_stmt_reg, "ECB stmt | SX5E   ~ fwd+curr +OIS_MP1")
run_ols(OIS_MP2  ~ net_score_fwd + net_score_curr + OIS_MP1, ecb_stmt_reg, "ECB stmt | OIS_MP2  ~ fwd+curr +OIS_MP1")
run_ols(OIS_10Y  ~ net_score_fwd + net_score_curr + OIS_MP1, ecb_stmt_reg, "ECB stmt | OIS_10Y  ~ fwd+curr +OIS_MP1")
add_fe(run_fe_ladder_table("UST2Y",  "net_score_fwd + net_score_curr + MP1",     fed_stmt_reg, "FOMC stmt | UST2Y FE  fwd+curr"))
add_fe(run_fe_ladder_table("UST10Y", "net_score_fwd + net_score_curr + MP1",     fed_stmt_reg, "FOMC stmt | UST10Y FE fwd+curr"))
add_fe(run_fe_ladder_table("OIS_MP2",  "net_score_fwd + net_score_curr + OIS_MP1", ecb_stmt_reg, "ECB stmt | OIS_MP2 FE  fwd+curr"))
add_fe(run_fe_ladder_table("OIS_10Y",  "net_score_fwd + net_score_curr + OIS_MP1", ecb_stmt_reg, "ECB stmt | OIS_10Y FE fwd+curr"))

# -- 27e) Dual-mandate Spec A vs B vs C -------------------------------------
cat("\n-- 27e) DUAL-MANDATE: Spec A vs B vs C on each outcome --\n")
run_dual_mandate("SP500",  fed_stmt_reg, control = "MP1",     label_prefix = "FOMC stmt | SP500")
run_dual_mandate("UST2Y",  fed_stmt_reg, control = "MP1",     label_prefix = "FOMC stmt | UST2Y")
run_dual_mandate("UST10Y", fed_stmt_reg, control = "MP1",     label_prefix = "FOMC stmt | UST10Y")
run_dual_mandate("EURUSD", fed_stmt_reg, control = "MP1",     label_prefix = "FOMC stmt | EURUSD")
run_dual_mandate("SX5E",   ecb_stmt_reg, control = "OIS_MP1", label_prefix = "ECB stmt | SX5E")
run_dual_mandate("OIS_MP2",  ecb_stmt_reg, control = "OIS_MP1", label_prefix = "ECB stmt | OIS_MP2")
run_dual_mandate("OIS_10Y",  ecb_stmt_reg, control = "OIS_MP1", label_prefix = "ECB stmt | OIS_10Y")
run_dual_mandate("EURUSD", ecb_stmt_reg, control = "OIS_MP1", label_prefix = "ECB stmt | EURUSD (ECB)")

# -- 27f) Save R^2 comparison: net (Spec A) vs DM (Spec C) ------------------
build_r2_comparison <- function(outcome, data, control, label) {
  d <- data %>% filter(!is.na(.data[[outcome]]),
                       !is.na(net_score),
                       !is.na(policy_score), !is.na(inflation_score),
                       !is.na(employment_score), !is.na(residual_score),
                       !is.na(.data[[control]]))
  if (nrow(d) < 10) return(NULL)
  fA <- as.formula(sprintf("%s ~ net_score + %s", outcome, control))
  fB <- as.formula(sprintf("%s ~ policy_score + econ_score + %s", outcome, control))
  fC <- as.formula(sprintf("%s ~ policy_score + inflation_score + employment_score + residual_score + %s",
                           outcome, control))
  mA <- lm(fA, d); mB <- lm(fB, d); mC <- lm(fC, d)
  data.frame(
    corpus = label, outcome = outcome, n = nrow(d),
    r2_A_net    = round(summary(mA)$r.squared, 4),
    r2_B_pol_ec = round(summary(mB)$r.squared, 4),
    r2_C_full   = round(summary(mC)$r.squared, 4),
    delta_C_minus_A = round(summary(mC)$r.squared - summary(mA)$r.squared, 4),
    stringsAsFactors = FALSE
  )
}
r2_rows <- list()
for (oc in c("SP500", "EURUSD", "UST2Y", "UST10Y"))
  r2_rows[[length(r2_rows) + 1L]] <- build_r2_comparison(oc, fed_stmt_reg, "MP1", "FOMC stmt")
for (oc in c("SX5E",  "EURUSD", "OIS_MP2",  "OIS_10Y"))
  r2_rows[[length(r2_rows) + 1L]] <- build_r2_comparison(oc, ecb_stmt_reg, "OIS_MP1", "ECB stmt")
r2_table <- bind_rows(Filter(Negate(is.null), r2_rows))
cat("\n-- R^2 comparison (net vs dual-mandate) --\n")
print(r2_table, row.names = FALSE)
write_csv(r2_table, file.path(output_dir, "r2_dual_mandate_comparison_v30.csv"))

# =============================================================================
# STEP 28 - RQ1 RF REGRESSIONS  (rf_score baseline + FE ladder + horse race)
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 28: RQ1 RF regressions\n")
cat("==================================================================\n")

# -- 28a) rf_score on key outcomes ------------------------------------------
cat("\n-- 28a) RF: outcome ~ rf_score + control --\n")
run_ols(SP500  ~ rf_score + MP1, fed_stmt_with_bert, "FOMC stmt | SP500  ~ rf_score + MP1")
run_ols(EURUSD ~ rf_score + MP1, fed_stmt_with_bert, "FOMC stmt | EURUSD ~ rf_score + MP1")
run_ols(UST2Y  ~ rf_score + MP1, fed_stmt_with_bert, "FOMC stmt | UST2Y  ~ rf_score + MP1")
run_ols(UST10Y ~ rf_score + MP1, fed_stmt_with_bert, "FOMC stmt | UST10Y ~ rf_score + MP1")
run_ols(SX5E    ~ rf_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | SX5E    ~ rf_score + OIS_MP1")
run_ols(EURUSD  ~ rf_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | EURUSD  ~ rf_score + OIS_MP1")
run_ols(OIS_MP2 ~ rf_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | OIS_MP2 ~ rf_score + OIS_MP1")
run_ols(OIS_10Y ~ rf_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | OIS_10Y ~ rf_score + OIS_MP1")

# -- 28b) FE ladder for rf_score --------------------------------------------
cat("\n-- 28b) FE ladder: outcome ~ rf_score [+ chair / + chair+era] --\n")
add_fe(run_fe_ladder_table("SP500",  "rf_score + MP1",     fed_stmt_with_bert, "FOMC stmt | SP500  rf"))
add_fe(run_fe_ladder_table("UST2Y",  "rf_score + MP1",     fed_stmt_with_bert, "FOMC stmt | UST2Y  rf"))
add_fe(run_fe_ladder_table("UST10Y", "rf_score + MP1",     fed_stmt_with_bert, "FOMC stmt | UST10Y rf"))
add_fe(run_fe_ladder_table("SX5E",   "rf_score + OIS_MP1", ecb_stmt_with_bert, "ECB stmt | SX5E    rf"))
add_fe(run_fe_ladder_table("OIS_MP2",  "rf_score + OIS_MP1", ecb_stmt_with_bert, "ECB stmt | OIS_MP2 rf"))
add_fe(run_fe_ladder_table("OIS_10Y",  "rf_score + OIS_MP1", ecb_stmt_with_bert, "ECB stmt | OIS_10Y rf"))

# -- 28c) Horse race rf + dict ----------------------------------------------
cat("\n-- 28c) HORSE RACE: outcome ~ rf_score + net_score + control --\n")
run_ols(SP500  ~ rf_score + net_score + MP1, fed_stmt_with_bert, "FOMC stmt | SP500  rf+dict")
run_ols(UST2Y  ~ rf_score + net_score + MP1, fed_stmt_with_bert, "FOMC stmt | UST2Y  rf+dict")
run_ols(UST10Y ~ rf_score + net_score + MP1, fed_stmt_with_bert, "FOMC stmt | UST10Y rf+dict")
run_ols(SX5E   ~ rf_score + net_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | SX5E    rf+dict")
run_ols(OIS_MP2  ~ rf_score + net_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | OIS_MP2 rf+dict")
run_ols(OIS_10Y  ~ rf_score + net_score + OIS_MP1, ecb_stmt_with_bert, "ECB stmt | OIS_10Y rf+dict")

# =============================================================================
# STEP 29 - RQ1 METHOD COMPARISON  (dict-DM vs RF vs BERT vs horse race)
# =============================================================================
# On the COMMON sample where rf_score, net_score, AND bert_score are non-NA
# (so R^2 values are directly comparable). Reports R^2 of:
#   M1 = dict (Spec A net + control)
#   M2 = RF  (rf_score + control)
#   M3 = BERT (bert_score + control)
#   M4 = horse race (dict + RF + BERT + control)
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 29: RQ1 method comparison (dict vs RF vs BERT vs horse race)\n")
cat("==================================================================\n")

method_compare <- function(outcome, data, control, label) {
  vars_needed <- c(outcome, "net_score", "rf_score", "bert_score", control)
  has_all <- all(vars_needed %in% names(data))
  if (!has_all) {
    cat(sprintf("  %s | %s SKIP - missing cols (%s)\n",
                label, outcome,
                paste(setdiff(vars_needed, names(data)), collapse = ", ")))
    return(NULL)
  }
  d <- data[stats::complete.cases(data[, vars_needed, drop = FALSE]), ]
  if (nrow(d) < 15) {
    cat(sprintf("  %s | %s SKIP - common-sample N too small (%d)\n",
                label, outcome, nrow(d)))
    return(NULL)
  }
  f1 <- as.formula(sprintf("%s ~ net_score + %s",                                outcome, control))
  f2 <- as.formula(sprintf("%s ~ rf_score + %s",                                 outcome, control))
  f3 <- as.formula(sprintf("%s ~ bert_score + %s",                               outcome, control))
  f4 <- as.formula(sprintf("%s ~ net_score + rf_score + bert_score + %s",        outcome, control))
  m1 <- lm(f1, d); m2 <- lm(f2, d); m3 <- lm(f3, d); m4 <- lm(f4, d)
  cat(sprintf("\n%s | %s | common-sample N=%d\n", label, outcome, nrow(d)))
  cat(sprintf("  M1 dict R^2  = %.4f  (adj %.4f)\n",
              summary(m1)$r.squared, summary(m1)$adj.r.squared))
  cat(sprintf("  M2 RF   R^2  = %.4f  (adj %.4f)\n",
              summary(m2)$r.squared, summary(m2)$adj.r.squared))
  cat(sprintf("  M3 BERT R^2  = %.4f  (adj %.4f)\n",
              summary(m3)$r.squared, summary(m3)$adj.r.squared))
  cat(sprintf("  M4 all  R^2  = %.4f  (adj %.4f)\n",
              summary(m4)$r.squared, summary(m4)$adj.r.squared))
  data.frame(
    corpus = label, outcome = outcome, n = nrow(d),
    r2_dict   = round(summary(m1)$r.squared, 4),
    r2_rf     = round(summary(m2)$r.squared, 4),
    r2_bert   = round(summary(m3)$r.squared, 4),
    r2_all    = round(summary(m4)$r.squared, 4),
    adj_dict  = round(summary(m1)$adj.r.squared, 4),
    adj_rf    = round(summary(m2)$adj.r.squared, 4),
    adj_bert  = round(summary(m3)$adj.r.squared, 4),
    adj_all   = round(summary(m4)$adj.r.squared, 4),
    stringsAsFactors = FALSE
  )
}

mc_rows <- list()
for (oc in c("SP500", "UST2Y", "UST10Y", "EURUSD"))
  mc_rows[[length(mc_rows)+1L]] <- method_compare(oc, fed_stmt_with_bert, "MP1", "FOMC stmt")
for (oc in c("SX5E",  "OIS_MP2",  "OIS_10Y", "EURUSD"))
  mc_rows[[length(mc_rows)+1L]] <- method_compare(oc, ecb_stmt_with_bert, "OIS_MP1", "ECB stmt")
mc_table <- bind_rows(Filter(Negate(is.null), mc_rows))
if (nrow(mc_table) > 0) {
  cat("\n-- RQ1 method comparison (R^2 on common sample) --\n")
  print(mc_table, row.names = FALSE)
  write_csv(mc_table, file.path(output_dir, "rq1_method_comparison_v30.csv"))
}

# =============================================================================
# STEP 30 - RQ2 EXPANDING-WINDOW FORECAST REGRESSIONS
# =============================================================================
# For each (corpus, target) pair: regress the target on rf_pred_<target>_full,
# then add the _fwd score, then add dictionary as a horse race.
#   M1 = target ~ rf_pred_full + control
#   M2 = target ~ rf_pred_full + rf_pred_fwd + control
#   M3 = target ~ rf_pred_full + rf_pred_fwd + net_score_fwd + net_score_curr + control
# Control is bps_curr for bps_* targets, y2_curr for y2_* targets.
# Done for all four targets (bps_next, bps_next2, y2_next, y2_next2) on all
# four corpora that have non-NA RF predictions for that target.
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 30: RQ2 expanding-window forecast regressions\n")
cat("==================================================================\n")

run_xw_reg <- function(target, full_col, fwd_col, control, data, label) {
  vars_needed <- c(target, full_col, control, "net_score_fwd", "net_score_curr")
  if (!(full_col %in% names(data))) {
    cat(sprintf("\n  %s | target=%s SKIP - %s not in data\n", label, target, full_col))
    return(NULL)
  }
  d <- data %>% filter(!is.na(.data[[target]]), !is.na(.data[[full_col]]),
                       !is.na(.data[[control]]))
  if (nrow(d) < 15) {
    cat(sprintf("\n  %s | target=%s SKIP - N too small (%d)\n", label, target, nrow(d)))
    return(NULL)
  }
  cat(sprintf("\n%s | target=%s | N=%d\n", label, target, nrow(d)))
  
  f1 <- as.formula(sprintf("%s ~ %s + %s", target, full_col, control))
  m1 <- lm(f1, d)
  cat(sprintf("  M1 [full only]  R^2 = %.4f  (adj %.4f)\n",
              summary(m1)$r.squared, summary(m1)$adj.r.squared))
  print(coeftest(m1, vcov = vcovHC(m1, type = "HC1")))
  
  m2 <- NULL; m3 <- NULL
  has_fwd <- (fwd_col %in% names(d)) && (sum(!is.na(d[[fwd_col]])) >= 15)
  if (has_fwd) {
    d2 <- d %>% filter(!is.na(.data[[fwd_col]]))
    f2 <- as.formula(sprintf("%s ~ %s + %s + %s", target, full_col, fwd_col, control))
    m2 <- lm(f2, d2)
    cat(sprintf("\n  M2 [+ fwd]      R^2 = %.4f  (adj %.4f, N=%d)\n",
                summary(m2)$r.squared, summary(m2)$adj.r.squared, nobs(m2)))
    print(coeftest(m2, vcov = vcovHC(m2, type = "HC1")))
    if (all(c("net_score_fwd", "net_score_curr") %in% names(d2))) {
      d3 <- d2 %>% filter(!is.na(net_score_fwd), !is.na(net_score_curr))
      if (nrow(d3) >= 15) {
        f3 <- as.formula(sprintf("%s ~ %s + %s + net_score_fwd + net_score_curr + %s",
                                 target, full_col, fwd_col, control))
        m3 <- lm(f3, d3)
        cat(sprintf("\n  M3 [+ dict]     R^2 = %.4f  (adj %.4f, N=%d)\n",
                    summary(m3)$r.squared, summary(m3)$adj.r.squared, nobs(m3)))
        print(coeftest(m3, vcov = vcovHC(m3, type = "HC1")))
      }
    }
  } else {
    f3 <- as.formula(sprintf("%s ~ %s + net_score_fwd + net_score_curr + %s",
                             target, full_col, control))
    d3 <- d %>% filter(!is.na(net_score_fwd), !is.na(net_score_curr))
    if (nrow(d3) >= 15) {
      m3 <- lm(f3, d3)
      cat(sprintf("\n  M3 [+ dict, no fwd-RF available]  R^2 = %.4f  (adj %.4f, N=%d)\n",
                  summary(m3)$r.squared, summary(m3)$adj.r.squared, nobs(m3)))
      print(coeftest(m3, vcov = vcovHC(m3, type = "HC1")))
    }
  }
  
  data.frame(
    corpus  = label, target = target, n = nrow(d),
    r2_full        = round(summary(m1)$r.squared, 4),
    adj_r2_full    = round(summary(m1)$adj.r.squared, 4),
    r2_full_fwd    = if (!is.null(m2)) round(summary(m2)$r.squared, 4)        else NA_real_,
    adj_r2_full_fwd= if (!is.null(m2)) round(summary(m2)$adj.r.squared, 4)    else NA_real_,
    r2_horse       = if (!is.null(m3)) round(summary(m3)$r.squared, 4)        else NA_real_,
    adj_r2_horse   = if (!is.null(m3)) round(summary(m3)$adj.r.squared, 4)    else NA_real_,
    stringsAsFactors = FALSE
  )
}

target_map <- list(
  bps_next  = list(full = "rf_pred_bps_next_full",  fwd = "rf_pred_bps_next_fwd",  control = "bps_curr"),
  bps_next2 = list(full = "rf_pred_bps_next2_full", fwd = "rf_pred_bps_next2_fwd", control = "bps_curr"),
  y2_next   = list(full = "rf_pred_y2_next_full",   fwd = "rf_pred_y2_next_fwd",   control = "y2_curr"),
  y2_next2  = list(full = "rf_pred_y2_next2_full",  fwd = "rf_pred_y2_next2_fwd",  control = "y2_curr")
)
corpora_xw <- list(
  fomc_stmt = list(df = fed_stmt_rq2_xw, label = "FOMC stmt"),
  fomc_pc   = list(df = fed_pc_rq2_xw,   label = "FOMC PC  "),
  ecb_stmt  = list(df = ecb_stmt_rq2_xw, label = "ECB stmt "),
  ecb_pc    = list(df = ecb_pc_rq2_xw,   label = "ECB PC   ")
)
xw_reg_rows <- list()
for (cn in names(corpora_xw)) {
  d   <- corpora_xw[[cn]]$df
  lab <- corpora_xw[[cn]]$label
  for (tg in names(target_map)) {
    if (!target_flags[[tg]]) next  # respect the RUN_RQ2_* flags
    spec <- target_map[[tg]]
    res <- run_xw_reg(target = tg, full_col = spec$full, fwd_col = spec$fwd,
                      control = spec$control, data = d, label = lab)
    if (!is.null(res)) xw_reg_rows[[length(xw_reg_rows) + 1L]] <- res
  }
}
xw_reg_table <- bind_rows(xw_reg_rows)
if (nrow(xw_reg_table) > 0) {
  cat("\n-- RQ2 forecast regressions summary --\n")
  print(xw_reg_table, row.names = FALSE)
  write_csv(xw_reg_table, file.path(output_dir, "rq2_method_comparison_v30.csv"))
}

# Save accumulated FE ladder rows
fe_ladder_results <- bind_rows(fe_results)
if (nrow(fe_ladder_results) > 0) {
  write_csv(fe_ladder_results, file.path(output_dir, "fe_ladder_results_v30.csv"))
  cat(sprintf("\n  Saved %d FE-ladder rows -> fe_ladder_results_v30.csv\n",
              nrow(fe_ladder_results)))
}

# =============================================================================
# STEP 31 - PLOTS
# =============================================================================
# 8 single plots (4 corpora x 2 score types: dict net_score and rf_score)
# 4 comparison plots (one per corpus, dict + RF overlaid)
# Saved as both .pdf (cairo_pdf) and .png (300 dpi)
# Recession bands: NBER for FOMC corpora, CEPR for ECB corpora
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 31: Plots\n")
cat("==================================================================\n")

theme_paper <- function() {
  theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      plot.title         = element_text(size = 12, face = "bold"),
      plot.subtitle      = element_text(size = 9,  colour = "grey35"),
      plot.caption       = element_text(size = 8,  colour = "grey50",
                                        hjust = 0, margin = margin(t = 5)),
      axis.title         = element_text(size = 10),
      axis.text          = element_text(size = 9),
      legend.position    = "top",
      legend.title       = element_blank(),
      legend.text        = element_text(size = 9),
      strip.text         = element_text(size = 10, face = "bold")
    )
}

us_recessions <- tribble(
  ~start,        ~end,
  "1990-07-01", "1991-03-01",
  "2001-03-01", "2001-11-01",
  "2007-12-01", "2009-06-01",
  "2020-02-01", "2020-04-01"
) %>% mutate(start = as.Date(start), end = as.Date(end))

ea_recessions <- tribble(
  ~start,        ~end,
  "2008-04-01", "2009-06-01",
  "2011-09-01", "2013-03-01",
  "2020-01-01", "2020-06-01"
) %>% mutate(start = as.Date(start), end = as.Date(end))

plot_score_vs_decisions <- function(reg_df, score_col, score_label, title_lab,
                                    recessions = us_recessions) {
  d <- reg_df %>%
    mutate(meeting_date = as.Date(meeting_date)) %>%
    filter(!is.na(meeting_date), !is.na(.data[[score_col]]))
  if (nrow(d) == 0) {
    cat(sprintf("  %s | %s: no data, skipping\n", title_lab, score_col))
    return(NULL)
  }
  rec_band <- recessions %>%
    filter(end   >= min(d$meeting_date, na.rm = TRUE),
           start <= max(d$meeting_date, na.rm = TRUE))
  decisions_df <- d %>% filter(!is.na(bps_change)) %>%
    mutate(decision = case_when(bps_change >  0 ~ "hike",
                                bps_change <  0 ~ "cut",
                                bps_change == 0 ~ "hold")) %>%
    mutate(decision = factor(decision, levels = c("cut", "hold", "hike")))
  p <- ggplot()
  if (nrow(rec_band) > 0)
    p <- p + geom_rect(data = rec_band,
                       aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
                       fill = "grey85", alpha = 0.45)
  p <- p +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.3) +
    geom_line(data = d, aes(x = meeting_date, y = .data[[score_col]]),
              colour = "#2C3E50", linewidth = 0.4, alpha = 0.85) +
    geom_point(data = d, aes(x = meeting_date, y = .data[[score_col]]),
               colour = "#2C3E50", size = 1.0, alpha = 0.9)
  if (nrow(decisions_df) > 0)
    p <- p + geom_point(data = decisions_df,
                        aes(x = meeting_date,
                            y = min(d[[score_col]], na.rm = TRUE) -
                              0.05 * diff(range(d[[score_col]], na.rm = TRUE)),
                            colour = decision, shape = decision),
                        size = 1.6, alpha = 0.9) +
    scale_colour_manual(values = c(cut = "#D8453E", hold = "grey55", hike = "#2C7BB6")) +
    scale_shape_manual(values = c(cut = 25, hold = 21, hike = 24))
  p +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(title = title_lab, y = score_label, x = NULL,
         caption = "Shaded bands = recessions. Bottom row = rate decisions.") +
    theme_paper()
}

# Build plot list
plots <- list()
for (cb in c("FOMC", "ECB")) {
  for (etype in c("stmt", "pc")) {
    df_name <- paste0(if (cb == "FOMC") "fed_" else "ecb_",
                      if (etype == "stmt") "stmt" else "pc",
                      "_with_bert")
    if (!exists(df_name)) next
    d   <- get(df_name)
    rec <- if (cb == "FOMC") us_recessions else ea_recessions
    lab <- sprintf("%s %s", cb, ifelse(etype == "stmt", "statements", "press conferences"))
    if ("net_score" %in% names(d))
      plots[[paste0(df_name, "_dict")]] <- plot_score_vs_decisions(
        d, "net_score", "Dictionary net_score (hawkish > 0)",
        sprintf("%s -- dictionary net_score", lab), recessions = rec)
    if ("rf_score" %in% names(d))
      plots[[paste0(df_name, "_rf")]] <- plot_score_vs_decisions(
        d, "rf_score",  "RF rf_score (P(hike) - P(cut))",
        sprintf("%s -- RF rf_score", lab), recessions = rec)
  }
}

# Save singles
for (nm in names(plots)) {
  p <- plots[[nm]]
  if (is.null(p)) next
  out_pdf <- sprintf("plot_%s_v30.pdf", nm)
  out_png <- sprintf("plot_%s_v30.png", nm)
  tryCatch({
    ggsave(out_pdf, p, device = cairo_pdf, width = 9, height = 3.6)
    ggsave(out_png, p, dpi = 300, width = 9, height = 3.6)
    cat(sprintf("  saved %s + %s\n", out_pdf, out_png))
  }, error = function(e) cat(sprintf("  FAILED to save %s: %s\n", nm, e$message)))
}

# Build 4 comparison plots (dict + RF on the same axes)
make_comparison_plot <- function(df, label, recessions) {
  if (!all(c("net_score","rf_score") %in% names(df))) return(NULL)
  d <- df %>%
    mutate(meeting_date = as.Date(meeting_date)) %>%
    filter(!is.na(meeting_date), !is.na(net_score) | !is.na(rf_score)) %>%
    select(meeting_date, net_score, rf_score) %>%
    pivot_longer(c(net_score, rf_score), names_to = "method", values_to = "score") %>%
    mutate(method = factor(method, levels = c("net_score","rf_score"),
                           labels = c("Dictionary","RF (OOB)")))
  if (nrow(d) == 0) return(NULL)
  rec <- recessions %>% filter(end >= min(d$meeting_date, na.rm = TRUE),
                               start <= max(d$meeting_date, na.rm = TRUE))
  p <- ggplot()
  if (nrow(rec) > 0)
    p <- p + geom_rect(data = rec,
                       aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
                       fill = "grey85", alpha = 0.45)
  p +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.3) +
    geom_line(data = d, aes(x = meeting_date, y = score, colour = method),
              linewidth = 0.45, alpha = 0.85) +
    scale_colour_manual(values = c(Dictionary = "#2C3E50", `RF (OOB)` = "#D8843E")) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(title = sprintf("%s -- dictionary vs RF score", label),
         y = "Score (positive = hawkish)", x = NULL,
         caption = "Shaded = recessions. Both methods rescaled to comparable units.") +
    theme_paper()
}
comp_specs <- list(
  list(df = fed_stmt_with_bert, lab = "FOMC statements",        rec = us_recessions),
  list(df = fed_pc_with_bert,   lab = "FOMC press conferences", rec = us_recessions),
  list(df = ecb_stmt_with_bert, lab = "ECB statements",         rec = ea_recessions),
  list(df = ecb_pc_with_bert,   lab = "ECB press conferences",  rec = ea_recessions)
)
for (spec in comp_specs) {
  p <- make_comparison_plot(spec$df, spec$lab, spec$rec)
  if (is.null(p)) next
  fkey <- gsub(" ", "_", tolower(spec$lab))
  tryCatch({
    ggsave(sprintf("plot_compare_%s_v30.pdf", fkey), p, device = cairo_pdf, width = 9, height = 3.6)
    ggsave(sprintf("plot_compare_%s_v30.png", fkey), p, dpi = 300, width = 9, height = 3.6)
    cat(sprintf("  saved comparison plot for %s\n", spec$lab))
  }, error = function(e) cat(sprintf("  FAILED comparison plot %s: %s\n", spec$lab, e$message)))
}

# =============================================================================
# STEP 32 - FINAL SUMMARY
# =============================================================================
cat("\n==================================================================\n")
cat("  STEP 32: Pipeline complete\n")
cat("==================================================================\n")

cat("\nKey output files in working directory:\n")
expected_files <- c(
  "combined_final_v2.csv",
  "fed_stmt_reg_v2.csv", "fed_pc_reg_v2.csv",
  "ecb_stmt_reg_v2.csv", "ecb_pc_reg_v2.csv",
  "rf_rq1_performance_v30.csv",
  "rf_rq1_models_v30.rds",
  "rf_feature_importance_v30.csv",
  "fed_stmt_reg_v2_rq2_xw_v30.csv", "fed_pc_reg_v2_rq2_xw_v30.csv",
  "ecb_stmt_reg_v2_rq2_xw_v30.csv", "ecb_pc_reg_v2_rq2_xw_v30.csv",
  "rf_rq2_xw_performance_v30.csv",
  "rf_rq2_xw_log_v30.csv",
  "rf_rq2_xw_results_v30.rds",
  "fed_stmt_with_bert_v30.csv", "fed_pc_with_bert_v30.csv",
  "ecb_stmt_with_bert_v30.csv", "ecb_pc_with_bert_v30.csv",
  "bert_diagnostic_correlations.csv",
  "r2_dual_mandate_comparison_v30.csv",
  "rq1_method_comparison_v30.csv",
  "rq2_method_comparison_v30.csv",
  "fe_ladder_results_v30.csv"
)
for (f in expected_files) {
  if (file.exists(file.path(output_dir, f))) {
    sz <- file.info(file.path(output_dir, f))$size
    cat(sprintf("  [OK]   %-45s  %d bytes\n", f, sz))
  } else {
    cat(sprintf("  [miss] %-45s  (not produced - check earlier output)\n", f))
  }
}

cat("\nPlots: plot_*_v30.pdf and .png in working directory.\n")
cat("\nDone.\n")

# === END OF v30_merged.R ===================================================