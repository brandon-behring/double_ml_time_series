# Run using the original authors' computational routines from the source tree.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
for (file in c("utils.R", "solver.R", "synthdid.R", "vcov.R")) {
  source(file.path(args[2], "R", file))
}
df <- read.csv(args[1], sep = ";")
s <- panel.matrices(df)
stopifnot(nrow(df) == 1209, identical(dim(s$Y), c(39L, 31L)),
          s$N0 == 38, s$T0 == 19, sum(df$treated) == 12,
          all(df$treated == (df$State == "California" & df$Year >= 1989)))

for (method in c("did_estimate", "sc_estimate", "synthdid_estimate")) {
  est <- get(method)(s$Y, s$N0, s$T0)
  set.seed(12345)
  se <- sqrt(vcov.synthdid_estimate(est, method = "placebo", replications = 200))[1]
  cat(sprintf("%s: %.4f SE %.4f CI [%.4f, %.4f]\n",
              method, as.numeric(est), se, est - 1.96 * se, est + 1.96 * se))
}
# Sensitivity to individual donors and a dated pretreatment falsification.
loo <- vapply(seq_len(s$N0), function(i) {
  as.numeric(synthdid_estimate(s$Y[-i, ], s$N0 - 1L, s$T0))
}, numeric(1))
cat(sprintf("Leave-one-donor-out range [%.4f, %.4f] excluding %s and %s\n",
            min(loo), max(loo), rownames(s$Y)[which.min(loo)], rownames(s$Y)[which.max(loo)]))
pretend <- synthdid_estimate(s$Y[, seq_len(s$T0)], s$N0, 13L)
cat(sprintf("Pretreatment pseudo-adoption in 1983 through 1988: %.4f\n", as.numeric(pretend)))
# Assign a known additive shift to each donor in turn. Exclude California always.
donors <- s$Y[seq_len(s$N0), , drop = FALSE]
post <- seq(s$T0 + 1L, ncol(donors))
errors <- t(vapply(seq_len(nrow(donors)), function(i) {
  y <- rbind(donors[-i, , drop = FALSE], donors[i, ])
  fit <- function(z) as.numeric(synthdid_estimate(z, nrow(z) - 1L, s$T0))
  null <- fit(y)
  y[nrow(y), post] <- y[nrow(y), post] - 15
  c(null, fit(y) + 15)
}, numeric(2)))
cat(sprintf("donor placebos n=%d mean error=%.4f RMSE=%.4f empirical 2.5%%, 97.5%% [%.4f, %.4f]\n",
            nrow(errors), mean(errors[,1]), sqrt(mean(errors[,1]^2)),
            quantile(errors[,1], 0.025), quantile(errors[,1], 0.975)))
cat(sprintf("max paired recovery error (null vs -15) %.10g\n", max(abs(errors[,1]-errors[,2]))))
