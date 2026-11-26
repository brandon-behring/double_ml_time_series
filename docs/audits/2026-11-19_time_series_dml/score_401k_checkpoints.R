# Recover base-learner columns as soon as every outer fold has finished.
# The all-method driver iterates through whole learners before the ensemble.
# Source inputs: checkpoint directory, CSV in exactly the authors' Stata row order.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
output <- args[1]
data <- read.csv(args[2])
stopifnot(nrow(data) == 9915L, !any(data$p401[data$e401 == 0] != 0))
methods <- c("RLasso", "Trees", "Forest", "Boosting", "Nnet")
pooled_score <- function(panel, folds, partitions, fits) {
  fold_scores <- lapply(seq_len(folds), function(i) {
    f <- fits[[i]]
    w <- data[partitions == i, ]
    if (panel == "plinear") return(cbind(u = as.numeric(f$ry), v = as.numeric(f$rz)))
    propensity <- as.numeric(if (panel == "LATE") f$mz2_x else f$mz_x)
    use <- which(propensity > 0.01 & propensity < 0.99)
    augment <- function(response, pred1, pred0) {
      w$e401 * (response - pred1) / propensity - (1 - w$e401) *
        (response - pred0) / (1 - propensity) + pred1 - pred0
    }
    a <- augment(w$net_tfa, as.numeric(f$my_z1x), as.numeric(f$my_z0x))
    b <- if (panel == "LATE") augment(w$p401, as.numeric(f$md_z1x), 0) else rep(1, nrow(w))
    cbind(a = a[use], b = b[use])
  })
  scores <- do.call(rbind, fold_scores)
  if (panel == "plinear") {
    u <- scores[, "u"]
    v <- scores[, "v"]
    theta <- sum(u * v) / sum(v^2)
    return(c(theta, sqrt(sum((v * (u - theta * v))^2)) / sum(v^2)))
  }
  # These reproduce the author code's SE.ATE and SE.LATE, including the
  # latter's use of the denominator mean as a fixed scaling factor.
  scaled <- scores[, "a"] / mean(scores[, "b"])
  c(mean(scaled), sd(scaled) / sqrt(nrow(scores)))
}

for (path in Sys.glob(file.path(output, "partial-all-*-k*-r*"))) {
  match <- regmatches(basename(path), regexec("^partial-all-(interactive|plinear|LATE)-k([25])-r([0-9]{3})$", basename(path)))[[1]]
  if (!length(match)) next
  panel <- match[2]
  folds <- as.integer(match[3])
  partition <- as.integer(match[4])
  suffix <- sprintf("%s-k%d-r%03d.csv", panel, folds, partition)
  if (file.exists(file.path(output, paste0("all-", suffix)))) next
  set.seed(1210L + partition)
  split <- runif(nrow(data))
  partitions <- as.numeric(cut(split, quantile(split, seq(0, 1, 1 / folds)), include.lowest = TRUE))
  for (i in seq_along(methods)) {
    method <- methods[i]
    target <- file.path(output, sprintf("component-%s-%s", method, suffix))
    if (file.exists(target)) next
    names <- file.path(path, sprintf("fit-%03d.rds", (i - 1L) * folds + seq_len(folds)))
    if (!all(file.exists(names))) break
    fits <- tryCatch(lapply(names, function(f) readRDS(f)$fit), error = function(e) NULL)
    if (is.null(fits)) break # A completed joint row can retire its cache concurrently.
    values <- pooled_score(panel, folds, partitions, fits)
    result <- data.frame(panel, folds, partition, method, estimate = values[1],
                         se = values[2], seconds = NA_real_)
    tmp <- tempfile(tmpdir = output)
    write.csv(result, tmp, row.names = FALSE)
    stopifnot(file.rename(tmp, target))
  }
}
