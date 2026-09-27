# Data Analytics for Business - Homework 2, Part 2
# Censored Models: Linear regression vs. Tobit on daily mobile data usage
#
# Run from this folder:  Rscript homework2_part2.R
# Requires the censReg package: install.packages("censReg")

library(censReg)

usage <- read.csv("Mobile_data_usage.csv")
str(usage)
summary(usage)

# ------------------------------------------------------------------
# Task 1: Scatter plot of DataUse (y) against Quota (x)
# ------------------------------------------------------------------
cat("Observations with DataUse == 0:", sum(usage$DataUse == 0),
    sprintf("(%.1f%% of %d)\n", 100 * mean(usage$DataUse == 0), nrow(usage)))
cat("Observations with DataUse < 0:", sum(usage$DataUse < 0), "\n")

# Plot shown in the R graphics window (as in RStudio) ...
plot(usage$Quota, usage$DataUse,
     xlab = "Remaining quota at start of day (MB)",
     ylab = "Daily data use (MB)",
     main = "Daily data use vs. remaining quota",
     pch = 16, cex = 0.4, col = rgb(0.16, 0.47, 0.84, 0.25))

# ... and saved to a file for the write-up
png("scatter_DataUse_vs_Quota.png", width = 1600, height = 1100, res = 200)
par(mar = c(4.5, 4.5, 3, 1), col.axis = "#52514e", col.lab = "#0b0b0b",
    fg = "#52514e", bg = "#fcfcfb")
plot(usage$Quota, usage$DataUse,
     xlab = "Remaining quota at start of day (MB)",
     ylab = "Daily data use (MB)",
     main = "Daily data use vs. remaining quota",
     type = "n", bty = "l", las = 1)
grid(col = "#e4e3df", lty = 1)
points(usage$Quota, usage$DataUse,
       pch = 16, cex = 0.4, col = adjustcolor("#2a78d6", alpha.f = 0.25))
invisible(dev.off())

# ------------------------------------------------------------------
# Task 2: Linear model (OLS)
# ------------------------------------------------------------------
ols <- lm(DataUse ~ Quota + Days, data = usage)
summary(ols)

# ------------------------------------------------------------------
# Task 3: Tobit model, left-censored at 0
# ------------------------------------------------------------------
tobit <- censReg(DataUse ~ Quota + Days, left = 0, right = Inf, data = usage)
summary(tobit)

# ------------------------------------------------------------------
# Task 4: Compare the estimates (intercept, Quota, Days)
# ------------------------------------------------------------------
compare <- data.frame(Linear = coef(ols), Tobit = coef(tobit)[1:3])
print(compare, digits = 6)
cat("Tobit / Linear ratio:\n")
print(round(coef(tobit)[1:3] / coef(ols), 3))
cat(sprintf("Tobit sigma = exp(logSigma) = %.4f\n", exp(coef(tobit)["logSigma"])))

# ------------------------------------------------------------------
# Task 5: Marginal effects of Quota from the Tobit model
# xValues order: (intercept = 1, Quota, Days)
# ------------------------------------------------------------------
mean_days <- mean(usage$Days)
cat("Mean of Days:", mean_days, "\n")

me_q10   <- margEff(tobit, xValues = c(1, 10, mean_days))
me_q2000 <- margEff(tobit, xValues = c(1, 2000, mean_days))
cat("\n(i) Quota = 10, Days = mean\n");   print(summary(me_q10))
cat("\n(ii) Quota = 2000, Days = mean\n"); print(summary(me_q2000))

# Check by hand: in a Tobit model, dE[y|x]/dx_j = beta_j * Phi(x'beta / sigma)
b <- coef(tobit)
sigma <- exp(b["logSigma"])
manual_me <- function(quota) {
  xb <- b["(Intercept)"] + b["Quota"] * quota + b["Days"] * mean_days
  c(xb = unname(xb), z = unname(xb / sigma), Phi = unname(pnorm(xb / sigma)),
    ME_Quota = unname(b["Quota"] * pnorm(xb / sigma)),
    ME_Days = unname(b["Days"] * pnorm(xb / sigma)))
}
cat("\nManual check, Quota = 10:\n");   print(manual_me(10))
cat("Manual check, Quota = 2000:\n");   print(manual_me(2000))

me_table <- data.frame(
  Model = c("Linear (constant)", "Tobit, Quota = 10", "Tobit, Quota = 2000"),
  ME_Quota = c(coef(ols)["Quota"], me_q10["Quota"], me_q2000["Quota"]),
  ME_Days = c(coef(ols)["Days"], me_q10["Days"], me_q2000["Days"])
)
print(me_table, digits = 6)
