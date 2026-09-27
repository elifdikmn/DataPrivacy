# Data Analytics for Business - Homework 2, Part 1
# Binary Response Models: Linear Probability Model vs. Logit
#
# Run from this folder:  Rscript homework2.R

# ------------------------------------------------------------------
# Task 1: Linear Probability Model
# ------------------------------------------------------------------

# Q1. Import the data and convert Education into a factor
loan <- read.csv("Loan.csv")
loan$Education <- as.factor(loan$Education)
str(loan)
summary(loan)

# Q2. Linear probability model: Loan on Income, Family, CCAvg, Education
lpm <- lm(Loan ~ Income + Family + CCAvg + Education, data = loan)
summary(lpm)

# Q3. Fitted values y_hat = X * beta_hat (predicted probability of Loan = 1)
loan$yhat_lpm <- fitted(lpm)
summary(loan$yhat_lpm)
cat("Fitted y_hat > 1:", sum(loan$yhat_lpm > 1), "\n")
cat("Fitted y_hat < 0:", sum(loan$yhat_lpm < 0), "\n")
cat("Outside [0, 1]  :", sum(loan$yhat_lpm > 1 | loan$yhat_lpm < 0),
    "of", nrow(loan), "\n")

cat("\nSome customers with y_hat > 1:\n")
above <- loan[loan$yhat_lpm > 1, ]
print(head(above[order(-above$yhat_lpm), ], 10))

cat("\nSome customers with y_hat < 0:\n")
below <- loan[loan$yhat_lpm < 0, ]
print(head(below[order(below$yhat_lpm), ], 10))

# ------------------------------------------------------------------
# Task 2: Logit Model
# ------------------------------------------------------------------

# Q4. Logit model with glm()
logit <- glm(Loan ~ Income + Family + CCAvg + Education,
             data = loan, family = binomial(link = "logit"))
summary(logit)

# Q5. Confusion matrix and Percent Correctly Predicted (PCP)
# Threshold = fraction of successes in the original data
threshold <- mean(loan$Loan)
cat("Threshold (share of Loan = 1):", threshold, "\n")

loan$p_logit <- predict(logit, type = "response")
loan$pred_logit <- ifelse(loan$p_logit > threshold, 1, 0)

conf_mat <- table(Actual = loan$Loan, Predicted = loan$pred_logit)
print(conf_mat)

pcp_overall <- mean(loan$pred_logit == loan$Loan)
pcp_y1 <- conf_mat["1", "1"] / sum(conf_mat["1", ])
pcp_y0 <- conf_mat["0", "0"] / sum(conf_mat["0", ])
cat(sprintf("PCP overall : %.4f (%.2f%%)\n", pcp_overall, 100 * pcp_overall))
cat(sprintf("PCP (y = 1) : %.4f (%.2f%%)\n", pcp_y1, 100 * pcp_y1))
cat(sprintf("PCP (y = 0) : %.4f (%.2f%%)\n", pcp_y0, 100 * pcp_y0))

# Q6. Predicted probability at mean Income/Family/CCAvg and Education = "2"
m_income <- mean(loan$Income)
m_family <- mean(loan$Family)
m_ccavg  <- mean(loan$CCAvg)
cat(sprintf("Means: Income = %.4f, Family = %.4f, CCAvg = %.4f\n",
            m_income, m_family, m_ccavg))

b <- coef(logit)
print(b)

# Linear index: z = b0 + b1*Income + b2*Family + b3*CCAvg + b4*Education2 + b5*Education3
# with Education2 = 1 and Education3 = 0
z <- b["(Intercept)"] + b["Income"] * m_income + b["Family"] * m_family +
     b["CCAvg"] * m_ccavg + b["Education2"] * 1 + b["Education3"] * 0
# Logistic CDF: P(y = 1 | x) = G(z) = exp(z) / (1 + exp(z))
p_manual <- exp(z) / (1 + exp(z))
cat(sprintf("z = %.6f\n", z))
cat(sprintf("P(Loan = 1) by formula = exp(z) / (1 + exp(z)) = %.6f\n", p_manual))

new_x <- data.frame(Income = m_income, Family = m_family, CCAvg = m_ccavg,
                    Education = factor("2", levels = levels(loan$Education)))
p_predict <- predict(logit, newdata = new_x, type = "response")
cat(sprintf("P(Loan = 1) by predict() = %.6f\n", p_predict))

# Q7. Co-list the coefficients of the two models
coef_table <- data.frame(LPM = coef(lpm), Logit = coef(logit))
print(round(coef_table, 6))
cat("Ratio Logit / LPM (slopes):\n")
print(round(coef(logit)[-1] / coef(lpm)[-1], 2))

# Q8. Partial effects at Income/Family/CCAvg means and Education = "2"
# Continuous variables: dP/dx_j = g(z) * b_j, with g(z) = exp(z) / (1 + exp(z))^2
g_z <- exp(z) / (1 + exp(z))^2
cat(sprintf("g(z) = exp(z) / (1 + exp(z))^2 = %.6f\n", g_z))
pe_income <- g_z * b["Income"]
pe_family <- g_z * b["Family"]
pe_ccavg  <- g_z * b["CCAvg"]

# Education (discrete): change in predicted probability relative to base level 1,
# holding the other variables at their means
p_at_edu <- function(level) {
  predict(logit, type = "response",
          newdata = data.frame(Income = m_income, Family = m_family, CCAvg = m_ccavg,
                               Education = factor(level, levels = levels(loan$Education))))
}
p_edu1 <- p_at_edu("1")
p_edu2 <- p_at_edu("2")
p_edu3 <- p_at_edu("3")
cat(sprintf("P at Education = 1, 2, 3: %.6f, %.6f, %.6f\n", p_edu1, p_edu2, p_edu3))
pe_edu2 <- p_edu2 - p_edu1
pe_edu3 <- p_edu3 - p_edu1

# For reference, the derivative-style approximation g(z) * b for the dummies
pe_edu2_approx <- g_z * b["Education2"]
pe_edu3_approx <- g_z * b["Education3"]

pe_table <- data.frame(
  Variable = c("Income", "Family", "CCAvg", "Education2", "Education3"),
  LPM = unname(coef(lpm)[c("Income", "Family", "CCAvg", "Education2", "Education3")]),
  Logit = unname(c(pe_income, pe_family, pe_ccavg, pe_edu2, pe_edu3)),
  Logit_gz_times_b = unname(c(pe_income, pe_family, pe_ccavg,
                              pe_edu2_approx, pe_edu3_approx))
)
print(pe_table, digits = 5)
