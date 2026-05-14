# =============================================================================
# verify_dictionary_coverage.R
# Checks every dictionary phrase against the 4 corpus CSVs using the same
# SnowballC stemming + bigram/trigram matching as the main pipeline.
# Outputs a table of (section, phrase, stemmed_form, n_gram, n_hits)
# and flags any phrase with 0 hits for removal.
# =============================================================================

library(readr)
library(dplyr)
library(tidytext)
library(SnowballC)
library(stringr)
library(tidyr)

cat("=== Dictionary Coverage Verification ===\n\n")

# -- Point this at your raw data folder -------------------------------------
raw_dir <- "C:/Users/Kajetan/Desktop/final thesis/raw data"

# -- Load the 4 corpora ----------------------------------------------------
cat("Loading corpora...\n")
fomc_stmt <- read_csv(file.path(raw_dir, "fomc_statements_econometrics.csv"),
                      show_col_types = FALSE)
fomc_pc   <- read_csv(file.path(raw_dir, "fomc_press_conferences_econometrics.csv"),
                      show_col_types = FALSE)
ecb_stmt  <- read_csv(file.path(raw_dir, "ecb_statements_econometrics.csv"),
                      show_col_types = FALSE)
ecb_dec   <- read_csv(file.path(raw_dir, "ecb_decisions_econometrics.csv"),
                      show_col_types = FALSE)

all_docs <- bind_rows(
  fomc_stmt %>% select(text_clean) %>% mutate(corpus = "fomc_stmt"),
  fomc_pc   %>% select(text_clean) %>% mutate(corpus = "fomc_pc"),
  ecb_stmt  %>% select(text_clean) %>% mutate(corpus = "ecb_stmt"),
  ecb_dec   %>% select(text_clean) %>% mutate(corpus = "ecb_dec")
) %>%
  filter(!is.na(text_clean), nchar(text_clean) > 50) %>%
  mutate(doc_id = row_number(),
         text   = tolower(text_clean))

cat(sprintf("  Total documents: %d\n\n", nrow(all_docs)))

# -- Build stemmed bigram and trigram inventories ---------------------------
cat("Extracting and stemming bigrams...\n")
corpus_bigrams <- all_docs %>%
  select(doc_id, text) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram)) %>%
  mutate(stemmed = sapply(str_split(bigram, " "), function(p)
    paste(wordStem(p, language = "english"), collapse = " "))) %>%
  count(stemmed, name = "n_hits")

cat(sprintf("  Unique stemmed bigrams: %d\n", nrow(corpus_bigrams)))

cat("Extracting and stemming trigrams...\n")
corpus_trigrams <- all_docs %>%
  select(doc_id, text) %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  filter(!is.na(trigram)) %>%
  mutate(stemmed = sapply(str_split(trigram, " "), function(p)
    paste(wordStem(p, language = "english"), collapse = " "))) %>%
  count(stemmed, name = "n_hits")

cat(sprintf("  Unique stemmed trigrams: %d\n\n", nrow(corpus_trigrams)))

# -- Dictionary (v32 final) ------------------------------------------------
dict_list <- list(
  policy_hawkish = c(
    "raise its","decided to raise","raise interest","raise interest rates",
    "raising interest rates","raised interest rates","raise the target",
    "raising rates","decided to increase","increase rates","increasing rates",
    "point increase","ongoing increases","rate increase","rate hike",
    "raise rates","ongoing rate increases","further rate increases",
    "additional rate increases","pace of increases",
    "further increases","additional increases",
    "sufficiently restrictive","restrictive policy","restrictive stance",
    "restrictive territory","restrictive monetary policy","restrictive policy stance",
    "more restrictive",
    "policy tightening","monetary tightening","further tightening",
    "tighten monetary policy","tightening monetary policy",
    "additional policy firming","further policy firming",
    "cumulative tightening","quantitative tightening","balance sheet reduction",
    "tightening cycle","remove accommodation","faster pace",
    "determined to return","restore price stability",
    "bring inflation down","bring inflation back","returning inflation",
    "appropriate to raise","remain vigilant","vigilant about",
    "maintain price stability","strongly committed","firmly committed",
    "policy normalization","normalize policy","committed to returning"
  ),
  policy_dovish = c(
    "lower its","lower the target","lower the rate",
    "lower interest rates","lowering interest rates",
    "decided to lower","decided to reduce","reduce the key",
    "cut rates","point reduction","point decrease","be decreased",
    "rate cut","point cut","rate reduction",
    "appropriate to lower","appropriate to reduce",
    "pace of reductions","pace of cuts","slower pace",
    "ease monetary policy","policy easing","monetary easing","further easing",
    "policy accommodation","monetary accommodation",
    "accommodative policy","accommodative stance",
    "highly accommodative","remains accommodative","ample degree",
    "asset purchases","purchase programme","purchase program",
    "net asset purchases","net purchases","quantitative easing",
    "forward guidance","reinvesting principal","reinvestment policy",
    "targeted longer","pandemic emergency","favourable financing",
    "preserve favourable","ample liquidity","smooth transmission",
    "support the economy","support the recovery","support economic",
    "extended period","at least until","until substantial further",
    "lower bound","neutral rate",
    "monetary stimulus","policy stimulus","additional stimulus",
    "further stimulus","provide stimulus",
    "generate economic","stabilises sustainably","sustainably at"
  ),
  inflation_hawkish = c(
    "high inflation","higher inflation","rising inflation",
    "inflation remains elevated","inflation is high","inflation is elevated",
    "elevated inflation","inflation remains high","still too high",
    "inflation running","far too high","remain well above","significantly above",
    "above target","above our aim","above our target","considerably above",
    "higher prices","rising prices","high prices","prices rose","inflation rose",
    "price pressures","price pressures remain",
    "core inflation","underlying inflation","services inflation",
    "domestic price","producer prices","upward trend",
    "wage growth","wage pressures","nominal wage","labour costs",
    "unit labour cost","unit labor cost",
    "second round effects","wage price spiral","cost pressures","pass through",
    "tight labor","tight labour",
    "upward pressure","inflationary pressures","inflationary risks",
    "inflationary expectations","entrenched inflation",
    "inflation overshoot","pricing power"
  ),
  inflation_dovish = c(
    "low inflation","lower inflation","subdued inflation","muted inflation",
    "falling inflation","inflation declined","inflation fell","inflation dropped",
    "inflation has eased","inflation has declined","inflation has fallen",
    "continued to decline","moving down","moving sustainably toward",
    "ongoing disinflation","disinflation process","disinflationary pressures",
    "prices fell","projected to decline","lower prices",
    "transitory factors","temporary factors",
    "below target","below our aim","significantly below",
    "converge to","price stability objective",
    "closer to target","back to target",
    "deflation risk","deflationary pressure","deflationary risk",
    "well anchored","firmly anchored"
  ),
  employment_hawkish = c(
    "strong growth","robust growth","higher growth","high growth",
    "rapid growth","above potential","solid growth","solid pace",
    "strong pace","strong economic","strong demand",
    "economic expansion","continued to expand","expanding at",
    "strong labor market","strong labour market",
    "tight labor market","tight labour market",
    "very tight","very strong labor","low unemployment","job gains",
    "rising wages","higher wages","strong wages",
    "supply constraints","capacity constraints",
    "labour shortage","labor shortage","exceeds supply"
  ),
  employment_dovish = c(
    "economic weakness","weak growth","low growth","lower growth",
    "weaker growth","slower growth","modest growth",
    "weak demand","weaker demand","lower demand",
    "growth slowed","drag on growth",
    "economic slowdown","economic downturn",
    "weak economic","weaker economic",
    "sharp decline","negative growth","deep recession",
    "high unemployment","rising unemployment","higher unemployment",
    "labor market slack","job losses",
    "loss of momentum","rising layoffs"
  ),
  residual_hawkish = c(
    "upside risk","tail risks","strong vigilance","continued vigilance",
    "be vigilant","remain alert","closely monitor",
    "upward revision","revised upward","revised up",
    "stronger than expected","higher than expected",
    "supply shock","supply disruption","energy shock","second round"
  ),
  residual_dovish = c(
    "downside risk","risks to growth","path of inflation",
    "negative territory","economic headwinds",
    "credit tightening","financial tightening",
    "tighter credit","tighter financing","tightening of financial",
    "credit standards","market turmoil","systemic risk",
    "financial vulnerabilities",
    "liquidity trap","zero lower bound","effective lower bound",
    "downward revision","revised downward","revised down",
    "weaker than expected","lower than expected",
    "geopolitical risks","geopolitical tensions","trade tensions",
    "sovereign debt","debt crisis","fiscal consolidation",
    "high uncertainty","heightened uncertainty","elevated uncertainty",
    "financial fragmentation"
  )
)

# -- Stem each phrase and look it up in the corpus inventories --------------
cat("Matching dictionary phrases against corpus...\n\n")

results <- data.frame(
  section      = character(),
  phrase       = character(),
  stemmed      = character(),
  n_words      = integer(),
  n_hits       = integer(),
  status       = character(),
  stringsAsFactors = FALSE
)

for (sec_name in names(dict_list)) {
  phrases <- dict_list[[sec_name]]
  for (phrase in phrases) {
    words   <- unlist(str_split(phrase, "\\s+"))
    stemmed <- paste(wordStem(words, language = "english"), collapse = " ")
    n_words <- length(words)

    if (n_words == 1) {
      # Unigrams are stored but never matched by the scoring function
      hits   <- 0L
      status <- "UNIGRAM (not matched by scoring)"
    } else if (n_words == 2) {
      row <- corpus_bigrams %>% filter(stemmed == !!stemmed)
      hits <- if (nrow(row) > 0) row$n_hits[1] else 0L
      status <- if (hits == 0) "ZERO" else if (hits < 5) "LOW" else "OK"
    } else if (n_words == 3) {
      row <- corpus_trigrams %>% filter(stemmed == !!stemmed)
      hits <- if (nrow(row) > 0) row$n_hits[1] else 0L
      status <- if (hits == 0) "ZERO" else if (hits < 5) "LOW" else "OK"
    } else {
      # 4+ words: cannot match any trigram in the scoring function
      hits   <- 0L
      status <- "TOO LONG (4+ words, only up to 3 matched)"
    }

    results <- rbind(results, data.frame(
      section = sec_name, phrase = phrase,
      stemmed = stemmed, n_words = n_words,
      n_hits = hits, status = status,
      stringsAsFactors = FALSE
    ))
  }
}

# -- Report -----------------------------------------------------------------
total   <- nrow(results)
n_zero  <- sum(results$status == "ZERO")
n_low   <- sum(results$status == "LOW")
n_ok    <- sum(results$status == "OK")
n_uni   <- sum(results$status == "UNIGRAM (not matched by scoring)")
n_long  <- sum(results$status == "TOO LONG (4+ words, only up to 3 matched)")

cat(sprintf("  Total phrases checked: %d\n", total))
cat(sprintf("  OK (>= 5 hits):       %d\n", n_ok))
cat(sprintf("  LOW (1-4 hits):        %d\n", n_low))
cat(sprintf("  ZERO (0 hits):         %d\n", n_zero))
cat(sprintf("  UNIGRAM (ignored):     %d\n", n_uni))
cat(sprintf("  TOO LONG (>3 words):   %d\n", n_long))

# -- Print problems ---------------------------------------------------------
problems <- results %>% filter(status %in% c("ZERO", "UNIGRAM (not matched by scoring)",
                                              "TOO LONG (4+ words, only up to 3 matched)"))
if (nrow(problems) > 0) {
  cat("\n========================================\n")
  cat("  PHRASES TO REMOVE (zero / unmatchable)\n")
  cat("========================================\n")
  for (i in seq_len(nrow(problems))) {
    cat(sprintf("  [%s] \"%s\" -> \"%s\" (%s)\n",
                problems$section[i], problems$phrase[i],
                problems$stemmed[i], problems$status[i]))
  }
} else {
  cat("\n  >> ALL PHRASES FIRE AT LEAST ONCE. Nothing to remove. <<\n")
}

# -- Print low-frequency phrases (informational) ----------------------------
low <- results %>% filter(status == "LOW") %>% arrange(n_hits)
if (nrow(low) > 0) {
  cat("\n=========================================\n")
  cat("  LOW-FREQUENCY PHRASES (1-4 hits, kept)\n")
  cat("=========================================\n")
  for (i in seq_len(nrow(low))) {
    cat(sprintf("  [%s] \"%s\" -> \"%s\" : %d hits\n",
                low$section[i], low$phrase[i],
                low$stemmed[i], low$n_hits[i]))
  }
}

# -- Full table (sorted by hits, ascending) ---------------------------------
cat("\n==========================================\n")
cat("  FULL RESULTS (sorted by hits ascending)\n")
cat("==========================================\n")
results_sorted <- results %>% arrange(n_hits, section, phrase)
print(as.data.frame(results_sorted), row.names = FALSE, right = FALSE)

# -- Save to CSV ------------------------------------------------------------
output_path <- file.path(dirname(raw_dir), "output v30",
                         "dictionary_coverage_check.csv")
write_csv(results_sorted, output_path)
cat(sprintf("\n  Saved -> %s\n", output_path))

cat("\nDone.\n")
