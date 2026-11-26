# Fit a complete or selected 401(k) table row with the pinned authors' code.
# Rscript replicate_401k.R SOURCE_DIRECTORY CHECKPOINT_DIRECTORY PARTITIONS WORKERS METHOD
# METHOD is all (includes Best), Trees, Forest, RLasso, Boosting or Nnet.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("Expected source dir, output dir, partition count, workers and method")
source_dir <- normalizePath(args[1])
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
output <- normalizePath(args[2])
repetitions <- as.integer(args[3])
workers <- as.integer(args[4])
selected <- args[5]
stopifnot(repetitions >= 1L, repetitions <= 100L, workers >= 1L)
job_file <- Sys.getenv("DML_401K_JOBS")
if (nzchar(job_file)) job_file <- normalizePath(job_file, mustWork = TRUE)
setwd(source_dir)
expected <- c("401K.R" = "74dd7fefdbaf59ede59af0c6b60333c7",
              "401K-LATE.R" = "92e17124d0c5fa394f947d382c373665",
              "Moment_Functions.R" = "6cf10d7d16b5e38cafe797fd7ac3703b",
              "ML_Functions.R" = "79f12a0e6a27ad43cc1646b5696c18e5",
              "sipp1991.dta" = "537eba85ba6ec692740f82138c3f91b7")
stopifnot(identical(unname(tools::md5sum(names(expected))), unname(expected)))
suppressPackageStartupMessages({
  library(foreign)
  library(rpart)
  library(nnet)
  library(sandwich)
  library(randomForest)
  library(gbm)
  library(hdm)
})
source("Moment_Functions.R")
data <- read.dta("sipp1991.dta")
settings <- parse("401K.R")
wanted <- c("x", "xl", "Boosting", "Forest", "RLasso", "Nnet", "Trees",
            "arguments", "ensemble", "methods")
for (expr in settings) {
  if (identical(expr[[1]], as.name("<-")) && as.character(expr[[2]])[1] %in% wanted) eval(expr)
}
if (selected != "all" && !selected %in% setdiff(methods, "Ensemble")) stop("Unknown method")
active <- if (selected == "all") methods else selected

# The mirror repeats the >1 guard in the interactive Best branch. A single
# learner has no model selection and must use itself as its own Best result.
# This patch only affects single-learner runs; keep all multi-learner code as supplied.
if (selected != "all") {
  rewrite <- function(node) {
    if (!is.call(node)) return(node)
    if (identical(node[[1]], as.name("if")) &&
        identical(node[[2]], quote(length(methods) > 1)) &&
        grepl("which.min(mean(MSE1", paste(deparse(node[[3]]), collapse = ""), fixed = TRUE)) {
      node[[2]] <- quote(length(methods) == 1)
    }
    as.call(lapply(node, rewrite))
  }
  body(DoubleML) <- rewrite(body(DoubleML))
}

# A fitted forest/boosting model can be large. The outer score and ensemble
# use predictions, residuals and errors exclusively; release fit objects after
# cond_comp has computed those outputs.
# Enumerate the same 0.02 weight grid in the authors' order without creating
# the 51^4 array of mostly infeasible points. The loop evaluates fold RMSE
# for every grid point using the squared-residual cross-product identity.
simplex <- function(lst) {
  stopifnot(length(lst) == 4L, identical(lst[[1]], seq(0, 1, 0.02)))
  grid <- expand.grid(i1 = 0:50, i2 = 0:50, i3 = 0:50)
  grid$i4 <- 50L - rowSums(grid)
  grid <- grid[grid$i4 >= 0L, ]
  grid <- grid[order(grid$i4, grid$i3, grid$i2, grid$i1), ]
  vapply(grid, function(i) lst[[1]][i + 1L], numeric(nrow(grid)))
}
grid_rmse <- function(predictions, outcomes, weights, fold, target, memory) {
  memory[[paste(fold, target)]] <- list(x = predictions, y = as.numeric(outcomes))
  # All simplex weights sum to one. Directly accumulate forecast errors to
  # avoid cancellation when predictions and outcomes agree almost exactly.
  residuals <- sweep(predictions, 1L, as.numeric(outcomes), "-")
  sums <- rowSums((weights %*% crossprod(residuals)) * weights)
  sqrt(pmax(0, sums / nrow(predictions)))
}
best_weight <- function(errors, target, weights, memory) {
  scores <- rowSums(errors[, , target])
  first <- which.min(scores)
  folds <- lapply(1:2, function(fold) memory[[paste(fold, target)]])
  if (all(vapply(folds, function(fold) all(fold$x == 0), logical(1)))) return(1L)
  # The source compares literal RMSE values in original grid order. Near a
  # tie, rounding in the quadratic identity can select a different mixture.
  # Replay the original per-weight products on both validation folds.
  magnitude <- max(vapply(folds, function(fold) max(abs(fold$x), abs(fold$y)), numeric(1)))
  threshold <- 16 * sqrt(.Machine$double.eps) * max(1, magnitude)
  candidates <- which(scores <= scores[first] + threshold)
  if (length(candidates) == 1L) return(first)
  exact <- vapply(candidates, function(i) {
    sum(vapply(folds, function(fold) {
      error(fold$x %*% weights[i, ], fold$y)$err
    }, numeric(1)))
  }, numeric(1))
  candidates[which.min(exact)]
}
rewrite_grid <- function(node) {
  if (!is.call(node)) return(node)
  if (identical(node, quote(as.matrix(expand.grid(lst))))) return(quote(simplex(lst)))
  if (identical(node[[1]], as.name("which.min")) &&
      grepl("rowSums(errorM", paste(deparse(node), collapse = ""), fixed = TRUE)) {
    target <- node[[2]][[2]][[2]][[5]]
    return(call("best_weight", as.name("errorM"), target, as.name("weight"), as.name("memory")))
  }
  if (identical(node[[1]], as.name("for")) && identical(node[[2]], as.name("p"))) {
    replace_score <- function(expr) {
      if (!is.call(expr)) return(expr)
      if (identical(expr[[1]], as.name("<-"))) {
        lhs <- expr[[2]]
        rhs <- expr[[3]][[2]]
        stopifnot(identical(lhs[[2]], as.name("errorM")), identical(rhs[[1]], as.name("error")))
        lhs[[3]] <- quote(expr = )
        expr[[2]] <- lhs
        expr[[3]] <- call("grid_rmse", rhs[[2]][[2]], rhs[[3]], as.name("weight"),
                          as.name("j"), lhs[[5]], as.name("memory"))
        return(expr)
      }
      as.call(lapply(expr, replace_score))
    }
    return(replace_score(node[[4]]))
  }
  as.call(lapply(node, rewrite_grid))
}
body(ensembleF) <- call("{", quote(memory <- new.env(parent = emptyenv())),
                        rewrite_grid(body(ensembleF)))

# Checkpoint each costly nuisance fit inside the joint row. Preserve its final
# RNG state so a resumed run follows the same random stream as an uninterrupted one.
state <- new.env()
original_cond_comp <- cond_comp
# The pinned IRM and LATE score paths never read the unconditional outcome
# fit. LATE also never reads the unconditional participation fit. Only these
# rigorous-lasso calls are deterministic under the pinned 0.2.0 penalty.
# Leave their fields absent from new checkpoints; older full fits can replay.
score_lasso <- original_cond_comp
skip_unused <- function(expr) {
  if (!is.call(expr)) return(expr)
  if (identical(expr[[1]], as.name("<-")) && is.call(expr[[3]]) &&
      identical(expr[[3]][[1]], as.name("rlassoF"))) {
    if (identical(expr[[2]], as.name("fit.y"))) {
      expr[[3]] <- quote(list(yhatout = NA_real_))
    }
    if (identical(expr[[2]], as.name("fit.z"))) {
      expr[[3]] <- call("if", quote(plinear == 3), quote(list(yhatout = NA_real_)), expr[[3]])
    }
  }
  as.call(lapply(expr, skip_unused))
}
body(score_lasso) <- skip_unused(body(score_lasso))
pinned_hdm <- as.character(packageVersion("hdm")) == "0.2.0"
cond_comp <- function(...) {
  cache <- selected == "all"
  args <- list(...)
  compute <- function() {
    if (pinned_hdm && identical(args$method, "RLasso") && args$plinear %in% c(0, 3)) {
      fit <- score_lasso(...)
      fit[c("my_x", "ry", "err.y")] <- NULL
      if (args$plinear == 3) fit[c("mz_x", "rz", "err.z", "mis.z")] <- NULL
    } else {
      fit <- original_cond_comp(...)
    }
    fit[c("fit.dz0", "fit.dz1", "fit.y")] <- NULL
    fit
  }
  if (!cache) return(compute())
  state$call <- state$call + 1L
  path <- file.path(state$cache_dir, sprintf("fit-%03d.rds", state$call))
  before <- .Random.seed
  key <- NULL
  # The full-training ensemble lasso repeats the outer base-learner fit.
  # Restrict reuse to this pinned deterministic learner and identical ordered
  # row IDs, feature formula, score context, and all fitting arguments.
  if (identical(args$method, "RLasso")) {
    key <- list(train = rownames(args$datause), test = rownames(args$dataout),
                y = args$y, d = args$d, z = args$z, x = args$x, xL = args$xL,
                plinear = args$plinear, binary = args$binary, flag = if (is.null(args$flag)) 0 else args$flag,
                arguments = args$arguments)
  }
  if (file.exists(path)) {
    saved <- readRDS(path)
    assign(".Random.seed", saved$seed, .GlobalEnv)
    fit <- saved$fit
  } else {
    found <- if (is.null(key)) integer() else which(vapply(state$lasso,
                              function(entry) identical(entry$key, key), logical(1)))
    fit <- if (length(found)) state$lasso[[found[1L]]]$fit else compute()
    temp <- tempfile(tmpdir = state$cache_dir)
    saveRDS(list(fit = fit, seed = .Random.seed), temp, compress = FALSE)
    stopifnot(file.rename(temp, path))
  }
  if (!is.null(key) && state$call <= state$folds && identical(before, .Random.seed)) {
    state$lasso[[length(state$lasso) + 1L]] <- list(key = key, fit = fit)
  }
  fit
}

fit_row <- function(job) {
  with(job, {
    draw_name <- sprintf("%s-%s-k%d-r%03d", selected, panel, folds, partition)
    path <- file.path(output, paste0(draw_name, ".csv"))
    if (file.exists(path)) return(TRUE)
    claim <- file.path(output, paste0(".claim-", draw_name))
    if (!dir.create(claim, showWarnings = FALSE)) return(TRUE)
    on.exit(unlink(claim, recursive = TRUE), add = TRUE)
    writeLines(c(Sys.info()[["nodename"]], as.character(Sys.getpid()),
                 format(Sys.time(), tz = "UTC", usetz = TRUE)), file.path(claim, "owner"))
    if (file.exists(path)) return(TRUE)
    state$call <- 0L
    state$folds <- folds
    state$lasso <- list()
    state$cache_dir <- file.path(output, paste0("partial-", draw_name))
    if (selected == "all") dir.create(state$cache_dir, showWarnings = FALSE)
    start <- proc.time()[3]
    tryCatch({
      set.seed(1210L + partition)
      late <- panel == "LATE"
      values <- DoubleML(data = data, y = "net_tfa", d = if (late) "p401" else "e401",
                         z = if (late) "e401" else NULL, xx = x, xL = xl,
                         methods = active, DML = "DML2", nfold = folds, est = panel,
                         arguments = arguments, ensemble = ensemble, silent = TRUE,
                         trim = c(0.01, 0.99))
      columns <- if (selected == "all") colnames(values) else active
      results <- data.frame(panel, folds, partition, method = columns,
                            estimate = as.numeric(values[1, columns]),
                            se = as.numeric(values[2, columns]), seconds = unname(proc.time()[3] - start))
      temp <- tempfile(tmpdir = output)
      write.csv(results, temp, row.names = FALSE)
      stopifnot(file.rename(temp, path))
      if (selected == "all") unlink(state$cache_dir, recursive = TRUE)
      unlink(file.path(output, paste0(draw_name, ".error")))
      TRUE
    }, error = function(e) {
      writeLines(conditionMessage(e), file.path(output, paste0(draw_name, ".error")))
      FALSE
    })
  })
}
jobs <- expand.grid(panel = c("plinear", "interactive", "LATE"), folds = c(2L, 5L),
                    partition = seq.int(as.integer(Sys.getenv("FIRST_PARTITION", 1L)), repetitions), stringsAsFactors = FALSE)
if (nzchar(job_file)) {
  requested <- read.csv(job_file, stringsAsFactors = FALSE)
  stopifnot(identical(names(requested), c("panel", "folds", "partition")))
  allowed <- merge(requested, jobs, by = names(jobs))
  stopifnot(nrow(allowed) == nrow(requested), anyDuplicated(requested) == 0L)
  jobs <- requested
}
outcome <- parallel::mclapply(split(jobs, seq_len(nrow(jobs))), fit_row,
                              mc.cores = workers, mc.preschedule = FALSE, mc.set.seed = FALSE)
if (!all(vapply(outcome, identical, logical(1), TRUE))) quit(status = 1)
