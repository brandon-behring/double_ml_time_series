# Public HeartSteps V1. Base R, no additional packages.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% 1:2, tools::md5sum(args[1]) == "4b6a5b0bdb44550694acb570fbaadbdd")
raw <- read.csv(args[1], na.strings = c("", "NA"))
raw <- raw[order(raw$user.index, raw$decision.index), ]
stopifnot(!anyDuplicated(raw[c("user.index", "decision.index")]))
first_slot <- ave(raw$sugg.select.slot, raw$user.index, FUN = function(x) x[1])
raw$day <- (raw$decision.index.nogap + first_slot - 1) %/% 5
in_window <- subset(raw, !is.na(day) & day <= 41)
conflicting <- with(in_window, send %in% "True" & (avail == "False" | is.na(send.sedentary)))
retained <- in_window[!conflicting, ]
d <- subset(retained, avail == "True")
d$delivery <- as.numeric(d$send == "True")
d$draw <- as.numeric(d$is.randomized == "True")
d$pre <- log(d$jbsteps30pre.zero + 0.5)
d$y <- log(d$jbsteps30.zero + 0.5)
stopifnot(nrow(retained) == 7540, nrow(d) == 6061, !anyNA(d[c("delivery","draw","pre","y")]))
cat("rows, travel, after day 41, contradictions, retained, available, participants, sent, mismatches\n")
print(c(nrow(raw), sum(is.na(raw$day)), sum(raw$day > 41, na.rm = TRUE), sum(conflicting),
        nrow(retained), nrow(d), length(unique(d$user.index)), sum(d$delivery), sum(d$draw != d$delivery)))

# Person-clustered CR1 sandwich and t interval, distinct from the paper's correction.
summary_row <- function(theta, influence_sums, degrees) {
  n <- length(influence_sums)
  se <- sqrt(n / (n - 1) * sum(influence_sums^2))
  q <- qt(.975, degrees)
  c(theta = theta, se = se, lower = theta - q*se, upper = theta + q*se)
}
benchmark <- function(z, action) {
  design <- cbind(1, z$pre, action - .6)
  fit <- lm.fit(design, z$y)
  # Mancl-DeRouen correction for working independence: the adjusted
  # influence of each person equals the full-minus-delete-person fit.
  full_information <- crossprod(design)
  groups <- split(seq_len(nrow(z)), z$user.index)
  person_slope <- vapply(groups, function(rows) {
    x <- design[rows, , drop = FALSE]
    solve(full_information - crossprod(x), crossprod(x, fit$residuals[rows]))[3]
  }, numeric(1))
  se <- sqrt(sum(person_slope^2))
  theta <- unname(fit$coefficients[3])
  width <- qt(.975, length(groups) - ncol(design)) * se
  c(theta = theta, se = se, lower = theta - width, upper = theta + width)
}
# Known randomization probability, nonlinear outcome adjustment trained on held-out folds.
# Estimate the PLR slope using sum((A-p)*(Y-m)) / sum((A-p)^2).
orthogonal <- function(z, predictions) {
  subset <- !is.na(predictions)
  z <- z[subset, ]
  resid_y <- z$y - predictions[subset]
  resid_a <- z$draw - .6
  denominator <- sum(resid_a^2)
  theta <- sum(resid_a * resid_y) / denominator
  score <- resid_a * (resid_y - theta * resid_a) / denominator
  influence <- rowsum(score, z$user.index)[,1]
  summary_row(theta, influence, length(influence) - 1)
}
fit_nuisance <- function(tr, te) {
  fit <- lm(y ~ poly(pre, 2, raw = TRUE) + factor(sugg.select.slot) + day, data = tr)
  as.numeric(predict(fit, te))
}
set.seed(20261119)
people <- sample(unique(d$user.index))
fold <- (match(d$user.index, people) - 1) %% 5
cv_person <- rep(NA_real_, nrow(d))
for (k in 0:4) {
  held <- fold == k
  cv_person[held] <- fit_nuisance(d[!held, ], d[held, ])
}
# Days 0-6 warm up. Leave a whole study day between each training and test block.
cv_time <- rep(NA_real_, nrow(d))
for (start in seq(7, 35, 7)) {
  held <- d$day >= start & d$day < start + 7
  cv_time[held] <- fit_nuisance(d[d$day < start - 1, ], d[held, ])
}
scored <- !is.na(cv_time)
results <- rbind(
  published_delivery_definition = benchmark(d, d$delivery),
  recorded_assignment_draw = benchmark(d, d$draw),
  draw_person_folds = orthogonal(d, cv_person),
  draw_WCLS_after_warmup = benchmark(d[scored, ], d$draw[scored]),
  draw_forward_study_weeks = orthogonal(d, cv_time)
)
print(round(results, 6))
cat("forward-scored rows", sum(scored), "; zero-imputed post-window records", sum(is.na(d$jbsteps30)), "\n")
stopifnot(abs(results[1,"theta"] - .13120) < .00001, abs(results[1,"se"] - .06739) < .00001)

# Run the study authors' actual finite-sample covariance and testing routines.
# Their paper script loads a private R workspace. The public CSV supplies the
# same working-independent regression here, adapted by their glm2gee function.
if (length(args) == 2) {
  stopifnot(tools::md5sum(args[2]) == "83a2d166604b45083e7c2a61eeee16df")
  source(args[2])
  observed <- lm(y ~ pre + I(delivery - .6), data = d)
  reference <- estimate(glm2gee(observed, id = factor(d$user.index)))
  print(reference)
  stopifnot(max(abs(reference[3, c("Estimate", "SE", "95% LCL", "95% UCL")]
    - results[1, ])) < 1e-8)
}
