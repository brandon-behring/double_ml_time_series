# Compare the paper's split-adjusted SE to the arithmetic in 401K.R and
# 401K-LATE.R at VC2015/DMLonGitHub commit b91cbf96. Uses saved fits only.
# Run: Rscript check_401k_se_aggregation.R

this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- dirname(normalizePath(this_file))
draws <- read.csv(gzfile(file.path(here, "401k-draws.csv.gz")))
joint <- subset(draws, run == "joint" & origin == "bundle")
reference <- read.csv(file.path(here, "401k-summary.csv"))
methods <- c("RLasso", "Trees", "Forest", "Boosting", "Nnet", "Ensemble")
columns <- c(methods, "best")
stopifnot(nrow(joint) == 4200L, nrow(reference) == 42L,
          all(is.finite(joint$estimate)), all(is.finite(joint$se)), all(joint$se > 0))

# Each author driver returned a data frame with seven point estimates followed
# by seven SEs. For one probability, matrixStats 0.52.2 colQuantiles()
# returned a vector of seven quantiles. Use base R to obtain the same vector
# on the complete, finite columns. Modern matrixStats rejects a data frame.
colQuantiles <- function(x, probs) {
  vapply(x, quantile, numeric(1), probs = probs, names = FALSE, type = 7)
}

driver_se <- function(r) {
  # RHS of result[2,] in both original drivers. Subtracting a seven-element
  # vector directly from this data frame recycles values down its columns.
  colQuantiles(
    sqrt(r[, (length(methods) + 2):ncol(r)]^2 +
           (r[, 1:(length(methods) + 1)] -
              colQuantiles(r[, 1:(length(methods) + 1)], probs = 0.5))^2),
    probs = 0.5
  )
}

panels <- unique(reference[c("panel", "folds")])
tables <- lapply(seq_len(nrow(panels)), function(i) {
  panel <- panels$panel[i]
  k <- panels$folds[i]
  d <- joint[joint$panel == panel & joint$folds == k, ]
  d <- d[order(match(d$method, columns), d$partition), ]
  stopifnot(identical(d$partition, rep(1:100, times = 7)),
            identical(d$method, rep(columns, each = 100)))
  theta <- matrix(d$estimate, nrow = 100)
  se <- matrix(d$se, nrow = 100)
  centered <- sweep(theta, 2, apply(theta, 2, median), "-")
  paper_rule <- sqrt(apply(se^2 + centered^2, 2, median))
  r <- as.data.frame(cbind(theta, se))
  targets <- reference[reference$panel == panel & reference$folds == k, ]
  targets <- targets[match(columns, targets$method), ]
  stopifnot(isTRUE(all.equal(unname(paper_rule), targets$adjusted_se, tolerance = 1e-12)))
  data.frame(panel, folds = k, method = columns, published = targets$published_adjusted_se,
             paper_rule, driver_forward = unname(driver_se(r)),
             driver_reverse = unname(driver_se(r[100:1, ])))
})

results <- do.call(rbind, tables)
print(results, row.names = FALSE, digits = 8)
cat(sprintf("Across 42 cells, the median absolute SE gap was $%.2f under the paper rule and $%.2f under the driver in ascending partition order.\n",
            median(abs(results$paper_rule - results$published)),
            median(abs(results$driver_forward - results$published))))
cat("The forward and reverse orders use the same 100 draws per cell. The original row order is unknown.\n")
