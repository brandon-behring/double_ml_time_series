# Extract Meager's standardized published effects and a same-file household check.
args <- commandArgs(trailingOnly = TRUE)
workspace <- new.env(parent = baseenv())
suppressWarnings(load(args[1], envir = workspace))
prefixes <- c("angelucci", "attanasio", "augsberg", "banerjee", "crepon", "karlan", "tarozzi")
results <- lapply(seq_along(prefixes), function(i) {
  get_column <- function(suffix) workspace[[paste0(prefixes[i], "_", suffix)]]
  profit <- get_column("profit") * workspace$the_profit_standardiser_USD_PPP_per_fortnight[i]
  treatment <- get_column("treatment")
  keep <- is.finite(profit) & treatment %in% 0:1
  d <- treatment[keep]
  y <- profit[keep]
  if (!all(0:1 %in% d)) stop("Missing an assignment arm")
  data.frame(country = workspace$study_country[i],
             source_rows = length(treatment), assigned_profit_rows = length(y),
             reported = workspace$standardised_reported_coefficients_profit[i],
             se = workspace$standardised_reported_coefficients_profit_sds[i],
             raw_difference = mean(y[d == 1]) - mean(y[d == 0]))
})
write.csv(do.call(rbind, results), stdout(), row.names = FALSE)
