# iFlex v2 analysis and daily data preparation. Base R.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2, tools::md5sum(args[1]) == "7547c0931b0cda997609d9e7df70fdd2")
e <- new.env()
load(args[1], envir = e)
h <- e$data_hourly
print(tapply(h$ID, h$Participation_Phase, function(x) length(unique(x))))
roster <- subset(e$participants, Group_Phase2 %in% c("Trond_1","Trond_Control"))
print(with(roster, table(as.character(Group_Phase2), as.character(Participation_status_Phase2))))
stopifnot(!anyDuplicated(roster$ID), all(roster$Participation_Experiment == "Yes"))
x <- h[h$Participation_Phase == "Phase_2" & h$ID %in% roster$ID,
       c("ID", "Date", "Hour", "Participation_Phase", "Demand_kWh", "Temperature", "Price_signal")]
x$arm <- as.character(roster$Group_Phase2[match(x$ID, roster$ID)])
x$hour <- as.integer(as.character(x$Hour))
x$day <- as.Date(x$Date)
days <- subset(e$experiment_days, Group == "Trond_1" & Participation_Phase == "Phase_2")
stopifnot(!anyDuplicated(days$Date), !anyDuplicated(x[c("ID","Date","hour")]),
          !anyNA(x$Demand_kWh), all(table(x$Date,x$ID)==24),
          all(as.character(days$Price_signal) %in% c("B_2","B_5","B_10","B_15")),
          all(x$Price_signal[x$arm == "Trond_Control"] == ""))
event <- min(days$Date)
print(days[days$Date == event, ])
peak <- subset(x, hour >= 8 & hour <= 20)
daily <- aggregate(Demand_kWh ~ ID + arm + day, data = peak, FUN = sum)
profile_test <- function(v, label) {
  treated <- v$Demand_kWh[v$arm == "Trond_1"]
  control <- v$Demand_kWh[v$arm == "Trond_Control"]
  value <- t.test(treated, control)
  cat(label," arm means, difference, Welch SE, CI, p\n")
  print(round(c(mean(treated), mean(control), mean(treated) - mean(control),
                value$stderr, value$conf.int, value$p.value), 6))
}
first <- subset(daily, day == event)
profile_test(first, "First event")
baseline <- aggregate(Demand_kWh ~ ID + arm, subset(daily, day < event - 1), mean)
profile_test(baseline, "Pre-announcement baseline")
delta <- first
delta$Demand_kWh <- first$Demand_kWh - baseline$Demand_kWh[match(first$ID, baseline$ID)]
profile_test(delta, "First event minus baseline")
events_mean <- aggregate(Demand_kWh ~ ID + arm, subset(daily, day %in% days$Date), mean)
profile_test(events_mean, "All event days (one average per household)")
events_delta <- events_mean
events_delta$Demand_kWh <- events_mean$Demand_kWh - baseline$Demand_kWh[match(events_mean$ID, baseline$ID)]
profile_test(events_delta, "All events minus baseline")
# Daily fixed-cohort difference in kWh per household over the 13 price-plateau hours.
wide <- reshape(aggregate(Demand_kWh ~ day + arm, daily, mean), idvar = "day",
                timevar = "arm", direction = "wide")
wide <- wide[order(wide$day), ]
temp <- aggregate(Temperature ~ day, subset(x, ID == x$ID[1]), mean)
series <- data.frame(day = wide$day, difference = wide$Demand_kWh.Trond_1 - wide$Demand_kWh.Trond_Control,
                     event = as.integer(wide$day %in% days$Date),
                     temperature = temp$Temperature[match(wide$day, temp$day)])
stopifnot(all(diff(series$day) == 1), nrow(first)==405, sum(series$event)==21, !anyNA(series))
cat("hourly rows, days, participating homes, event dates\n")
print(c(nrow(x), nrow(series), nrow(first), sum(series$event)))
write.csv(series, args[2], row.names = FALSE)
