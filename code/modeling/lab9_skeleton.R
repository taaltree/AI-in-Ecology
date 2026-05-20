# lab9_skeleton.R
# ---------------------------------------------------------------------------
# Single-season occupancy parameter recovery sim — intentionally minimal.
# You (and Claude Code) will fill this in for Lab 9.
#
# Goal: prove that unmarked::occu returns unbiased estimates of psi and p
# across 500 simulated datasets, in each of 4 (S, K) design scenarios.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(unmarked)
  library(tidyverse)
  # library(furrr); plan(multisession) # for parallel — Claude can set up
})

set.seed(42)

# 1. Simulator -----------------------------------------------------------------
simulate_occu <- function(S, K, psi, p, seed = NULL) {
  # TODO: draw z ~ Bernoulli(psi), draw y ~ Bernoulli(z * p)
  # TODO: return unmarkedFrameOccu(y)
  stop("Implement me.")
}

# 2. Fitter --------------------------------------------------------------------
fit_occu <- function(uf) {
  # TODO: fit occu(~1 ~1, uf), back-transform psi and p, return tibble:
  #   psi_hat, psi_lo, psi_hi, p_hat, p_lo, p_hi
  stop("Implement me.")
}

# 3. Single replicate ----------------------------------------------------------
one_rep <- function(S, K, psi, p, seed) {
  uf <- simulate_occu(S, K, psi, p, seed)
  fit_occu(uf)
}

# 4. Many reps -----------------------------------------------------------------
run_scenario <- function(scenario_name, S, K, psi, p, n_rep = 500) {
  message("Scenario ", scenario_name, " ...")
  # TODO: loop n_rep times (parallel ok), bind rows, return tibble:
  #   scenario, replicate, psi_true, p_true, psi_hat, ...
  stop("Implement me.")
}

# 5. Main ----------------------------------------------------------------------
scenarios <- tibble::tribble(
  ~name, ~S,  ~K,  ~psi, ~p,
   "A",  60,  6,   0.4,  0.30,
   "B",  60,  6,   0.4,  0.10,
   "C",  20,  6,   0.4,  0.30,
   "D",  60, 12,   0.4,  0.30,
)

# results <- purrr::pmap_dfr(scenarios, ~ run_scenario(..1, ..2, ..3, ..4, ..5))
# saveRDS(results, "recovery_results.rds")

# 6. Plotting (Task 4 of Lab 9) ------------------------------------------------
# TODO: 2x2 panel of histograms of psi_hat with true psi as a vline.
