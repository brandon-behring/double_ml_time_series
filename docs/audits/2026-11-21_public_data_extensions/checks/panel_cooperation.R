# Check the public estimator on all assignment paths of a synthetic rematching experiment.
args <- commandArgs(trailingOnly = TRUE)
check_null <- length(args) == 2L && args[2] == "--check-null"
stopifnot(length(args) == 1L || check_null)
zipfile <- normalizePath(args[1])
digest <- strsplit(system2("sha256sum", shQuote(zipfile), stdout = TRUE), " +")[[1]][1]
stopifnot(digest == "b790111a6bd5c8d77180ef83d152fb98418f2fbfd4083d4d2502a9cf055dcc47")
src <- unz(zipfile, "QE-replicationFiles/Code/panel_analysis_functions.R")
expressions <- parse(text = readLines(src, warn = FALSE))
close(src)
target_function <- ".est_plag_qstep_ht"
definition <- Filter(function(e) {
  is.call(e) && identical(e[[1]], as.name("<-")) && identical(e[[2]], as.name(target_function))
}, as.list(expressions))
stopifnot(length(definition) == 1L)
published <- new.env(parent = baseenv())
eval(definition[[1]], published)

prob <- 5 / 11
# Each column is a new round. Two participants share each pair-round assignment.
matching <- cbind(c(1, 1, 2, 2), c(3, 4, 3, 4), c(5, 6, 6, 5))
subjects <- nrow(matching)
rounds <- ncol(matching)
make_data <- function(bits) {
  W <- matrix(bits[matching], subjects, rounds)
  past <- cbind(0, W[, 1:2])
  id <- row(W)
  period <- col(W)
  Y <- 0.15 + 0.03 * id + 0.1 * period + 0.08 * W + 0.12 * past + 0.20 * W * past
  list(W = W, past = past, id = id, period = period, Y = Y)
}
features <- function(d, lag) {
  # Covariates available immediately before the first assignment in the effect window.
  history <- if (lag == 0L) d$past else cbind(0, d$past[, 1:2])
  cbind(intercept = 1, person = as.vector(d$id), period = as.vector(d$period), past = as.vector(history))
}
set.seed(20261121)
training <- replicate(20, make_data(rbinom(6, 1, prob)), simplify = FALSE)
fit <- function(lag) {
  x <- do.call(rbind, lapply(training, features, lag = lag))
  y <- unlist(lapply(training, function(d) as.vector(d$Y)))
  lm.fit(x, y)$coefficients
}
fitted <- lapply(0:1, fit)
score <- function(d, lag, adjustment) {
  w <- d$W
  received <- ifelse(w == 1, prob, 1 - prob)
  idx <- (1 + lag):rounds
  propensity <- received[, idx]
  if (lag == 1) propensity <- propensity * received[, idx - lag]
  if (adjustment == "trained") {
    nuisance <- matrix(features(d, lag) %*% fitted[[lag + 1L]], subjects, rounds)
  } else if (adjustment == "uses_lagged_treatment") {
    nuisance <- 0.12 * d$past
  } else {
    nuisance <- 0
  }
  mean((d$Y - nuisance)[, idx] * (2 * w[, idx - lag] - 1) / (2^lag * propensity))
}
paths <- as.matrix(expand.grid(rep(list(0:1), 6)))
weights <- apply(paths, 1, function(x) prod(ifelse(x == 1, prob, 1 - prob)))
stopifnot(abs(sum(weights) - 1) < 1e-12)
for (lag in 0:1) {
  values <- apply(paths, 1, function(bits) {
    d <- make_data(bits)
    original <- mean(vapply(seq_len(subjects), function(i) {
      published$.est_plag_qstep_ht(d$Y[i, ], d$W[i, ], rep(prob, rounds), lag,
                                  normalize = FALSE, pT.treatmentprob = TRUE)$est.mean
    }, numeric(1)))
    raw <- score(d, lag, "raw")
    stopifnot(abs(raw - original) < 1e-12)
    c(raw = raw, trained = score(d, lag, "trained"),
      uses_lagged_treatment = score(d, lag, "uses_lagged_treatment"))
  })
  truth <- if (lag == 0) 0.08 + 0.20 * prob * 2 / 3 else 0.12 + 0.20 / 2
  for (name in rownames(values)) {
    x <- values[name, ]
    expectation <- sum(weights * x)
    sd <- sqrt(sum(weights * (x - expectation)^2))
    expected <- if (lag == 1 && name == "uses_lagged_treatment") truth - 0.12 else truth
    stopifnot(abs(expectation - expected) < 1e-12)
    cat(sprintf("lag=%d score=%s target=%.8f mean=%.8f sd=%.8f\n",
                lag, name, truth, expectation, sd))
  }
}

if (check_null) {
  # Sharp-null outcomes are fixed over all assignments. Every path is weighted by its design probability.
  Y <- 0.15 + 0.03 * row(matching) + 0.10 * col(matching)
  distribution <- function(paired) {
    coins <- if (paired) max(matching) else length(matching)
    assignments <- as.matrix(expand.grid(rep(list(0:1), coins)))
    weighted <- prob^rowSums(assignments) * (1 - prob)^(coins - rowSums(assignments))
    scores <- t(apply(assignments, 1, function(bits) {
      W <- matrix(if (paired) bits[matching] else bits, subjects, rounds)
      vapply(0:1, function(lag) {
        mean(vapply(seq_len(subjects), function(i) {
          published$.est_plag_qstep_ht(Y[i, ], W[i, ], rep(prob, rounds), lag,
                                      normalize = FALSE, pT.treatmentprob = TRUE)$est.mean
        }, numeric(1)))
      }, numeric(1))
    }))
    stopifnot(abs(sum(weighted) - 1) < 1e-12)
    list(scores = scores, weights = weighted)
  }
  actual <- distribution(TRUE)
  references <- list(pair_round = actual, individual_round = distribution(FALSE))
  exact <- list(pair_round = c(81031/1771561, 625/14641),
                individual_round = c(313921/1771561, 28631/161051))
  for (name in names(references)) {
    ref <- references[[name]]
    for (lag in 0:1) {
      # Include numerically tied copies of the same rational statistic.
      tail_prob <- vapply(abs(actual$scores[, lag + 1L]), function(obs) {
        sum(ref$weights[abs(ref$scores[, lag + 1L]) >= obs - 1e-12])
      }, numeric(1))
      size <- sum(actual$weights[tail_prob <= 0.05])
      stopifnot(abs(size - exact[[name]][lag + 1L]) < 1e-12)
      cat(sprintf("sharp null reference=%s lag=%d size=%.12f\n", name, lag, size))
    }
  }
}
