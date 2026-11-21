# Two-arm design benchmark on the original iFlex Oslo hourly demand experiment.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Pass the published iFlex dataset.RData file")
if (unname(tools::md5sum(args[1L])) != "7547c0931b0cda997609d9e7df70fdd2") stop("Different Zenodo file")
env <- new.env(parent = emptyenv())
load(args[1L], envir = env)
p <- env$participants
p <- p[p$Group_Phase2 %in% c("Os_1", "Os_Control"), c("ID", "Group_Phase2")]
stopifnot(nrow(p) == 512L, !anyDuplicated(p$ID))
p$offer <- as.integer(p$Group_Phase2 == "Os_1")
days <- sort(env$experiment_days$Date[env$experiment_days$Group == "Os_1"])
stopifnot(length(days) == 20L, !anyDuplicated(days))
hourly <- env$data_hourly
include <- hourly$Participation_Phase == "Phase_2" & hourly$ID %in% p$ID &
    (hourly$Date < as.Date("2020-12-15") | hourly$Date %in% days)
h <- hourly[which(include), c("ID", "Date", "Hour", "Demand_kWh")]
rm(env, hourly)
h$Hour <- as.integer(as.character(h$Hour))
stopifnot(!anyDuplicated(h[, c("ID","Date","Hour")]))
cat("Randomized arm membership, invited price group and control:", sum(p$offer),sum(1L-p$offer),"\n")
daily <- function(hours) {
    sel <- h[h$Hour %in% hours & is.finite(h$Demand_kWh), ]
    sums <- aggregate(Demand_kWh ~ ID + Date, sel, sum)
    counts <- aggregate(Demand_kWh ~ ID + Date, sel, length)
    stopifnot(identical(sums[,1:2], counts[,1:2]))
    sums$y <- sums$Demand_kWh / length(hours)
    sums[counts$Demand_kWh == length(hours), c("ID","Date","y")]
}
peak <- daily(8:20)
night <- daily(1:6)
late <- daily(22:24)
# Unit of analysis is the household. Give each eligible calendar day equal weight.
mean_days <- function(d,calendar,minimum) {
    sel <- d[d$Date %in% calendar,]
    means <- tapply(sel$y,sel$ID,mean)
    n <- table(sel$ID)
    result <- unname(means[p$ID])
    result[is.na(n[p$ID]) | n[p$ID] < minimum] <- NA_real_
    result
}
pre <- seq(as.Date("2020-12-01"),as.Date("2020-12-14"),by="day")
p$pre_peak <- mean_days(peak,pre,10)
p$pre_night <- mean_days(night,pre,10)
p$first <- mean_days(peak,days[1:10],10)
p$last <- mean_days(peak,days[11:20],10)
p$post_first <- mean_days(late,days[1:10],10)
p$post_last <- mean_days(late,days[11:20],10)
p$available <- mean_days(peak,days[11:20],1)
diff_means <- function(y,offer) mean(y[offer==1L],na.rm=TRUE)-mean(y[offer==0L],na.rm=TRUE)
cat("Availability by arm (0 control, 1 price)\n")
print(aggregate(cbind(pre_peak,first,last,available)~offer,
    transform(p,pre_peak=is.finite(pre_peak),first=is.finite(first),last=is.finite(last),
        available=is.finite(available)),sum))
cat("Raw pre-experiment peak difference:",diff_means(p$pre_peak,p$offer),"\n")
cat("Available-case peak difference on final ten event days:",diff_means(p$available,p$offer),"kWh/hour\n")
# Cross both dependence axes. Train on separate households on the first ten event
# days. Score only the held-out households on the last ten. Baseline features end
# before the first day-ahead notification. Membership is randomized at household level.
analyse <- function(training, evaluation, label) {
    x <- p
    x$earlier <- training; x$outcome <- evaluation
    x <- x[complete.cases(x[, c("pre_peak","pre_night","earlier","outcome")]),]
    x <- x[order(x$ID),]
    set.seed(20261120)
    x$fold <- 0L
    for (a in 0:1) {
        at <- which(x$offer==a)
        x$fold[at] <- sample(rep(1:5,length.out=length(at)))
    }
    forecasts <- matrix(NA_real_,nrow(x),2)
    for (k in 1:5) for (a in 0:1) {
        fit <- lm(earlier ~ pre_peak + pre_night + I(pre_peak^2),
            data=x[x$fold!=k & x$offer==a,])
        forecasts[x$fold==k,a+1L] <- predict(fit,newdata=x[x$fold==k,])
    }
    stopifnot(all(is.finite(forecasts)))
    g <- x$offer; prob <- mean(g)
    untreated <- forecasts[,1]; treated <- forecasts[,2]
    contrast <- treated-untreated
    r0 <- x$outcome-untreated; r1 <- x$outcome-treated
    ct <- mean(r1[g==1]); cc <- mean(r0[g==0])
    estimate <- mean(contrast)+ct-cc
    influence <- contrast-mean(contrast)+g*(r1-ct)/prob-(1-g)*(r0-cc)/(1-prob)
    se <- sd(influence)/sqrt(nrow(x))
    # Design-based comparator at the same last-ten-day target. Center each
    # baseline column so the adjusted offer coefficient averages over households.
    x$c_peak <- x$pre_peak - mean(x$pre_peak)
    x$c_night <- x$pre_night - mean(x$pre_night)
    x$c_peak2 <- x$pre_peak^2 - mean(x$pre_peak^2)
    adjustment <- lm(outcome ~ offer * (c_peak + c_night + c_peak2), data=x)
    design <- model.matrix(adjustment)
    inverse <- chol2inv(chol(crossprod(design)))
    residual_weights <- residuals(adjustment)^2 / (1 - hatvalues(adjustment))
    robust <- inverse %*% crossprod(design, sweep(design, 1L, as.numeric(residual_weights), "*")) %*% inverse
    arm <- match("offer", names(coef(adjustment)))
    lin <- coef(adjustment)[arm]
    lin_se <- sqrt(robust[arm, arm])
    cat(label,"interacted baseline OLS", lin, "HC2 se", lin_se, "95% interval",
        lin + c(-1, 1)*1.96*lin_se, "\n")
    cat(label,"n control / price",sum(g==0),sum(g==1),
        "control mean",mean(x$outcome[g==0]),"raw",diff_means(x$outcome,g),
        "adjusted",estimate,"se",se,"95% interval",estimate+c(-1,1)*1.96*se,"\n")
}
analyse(p$first,p$last,"Peak, hours 8-20")
analyse(p$post_first,p$post_last,"Late, hours 22-24")
