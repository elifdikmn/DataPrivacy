# Homework 2 – Part 1: Binary Response Models

Data: `Loan.csv` (5,000 customers; 480 accepted the loan, so the success rate is 9.6%).
Code: [`homework2.R`](homework2.R). Full R output: [`homework2_output.txt`](homework2_output.txt).

Education levels: 1 = Undergrad (the base level), 2 = Graduate, 3 = Advanced/Professional.

---

## Task 1: Linear Probability Model

### Q1. Import the data and convert Education into a factor

```r
loan <- read.csv("Loan.csv")
loan$Education <- as.factor(loan$Education)
```

| Variable | Min | Mean | Max |
|---|---|---|---|
| Loan | 0 | 0.096 | 1 |
| Income ($1000s) | 8 | 73.77 | 224 |
| Family | 1 | 2.396 | 4 |
| CCAvg ($1000s/month) | 0 | 1.938 | 10 |

Education counts: level 1 = 2,096, level 2 = 1,403, level 3 = 1,501.

### Q2. LPM: `lm(Loan ~ Income + Family + CCAvg + Education)`

```
              Estimate Std. Error t value Pr(>|t|)
(Intercept) -3.455e-01  1.127e-02 -30.653  < 2e-16 ***
Income       3.367e-03  9.799e-05  34.364  < 2e-16 ***
Family       3.160e-02  3.010e-03  10.499  < 2e-16 ***
CCAvg        1.373e-02  2.538e-03   5.412 6.52e-08 ***
Education2   1.517e-01  8.473e-03  17.908  < 2e-16 ***
Education3   1.605e-01  8.229e-03  19.511  < 2e-16 ***

Residual standard error: 0.2393 on 4994 degrees of freedom
Multiple R-squared:  0.3409,  Adjusted R-squared:  0.3402
F-statistic: 516.6 on 5 and 4994 DF,  p-value: < 2.2e-16
```

**Interpreting the two Education coefficients.** R turns the factor into two dummy variables and leaves out Education = 1 (Undergrad) as the base level. Each coefficient is therefore a difference from an undergraduate customer with the same Income, Family and CCAvg:

- **Education2 = 0.1517.** A customer with a graduate degree is about **15.2 percentage points** more likely to accept the loan offer than an otherwise identical undergraduate.
- **Education3 = 0.1605.** A customer with an advanced/professional degree is about **16.1 percentage points** more likely to accept than an otherwise identical undergraduate.

Both are highly significant (p < 2e-16). Most of the difference is between undergraduates and everyone else. Graduate and advanced/professional customers differ by less than 1 percentage point (0.1605 − 0.1517 ≈ 0.009).

### Q3. What the fitted ŷ means, and values outside [0, 1]

In a linear probability model, ŷ = Xβ̂ is the **estimated probability that the customer accepts the loan**, P(Loan = 1 | X) = E[Loan | X]. It comes from a linear function, so nothing keeps it between 0 and 1.

In this data set:

- **ŷ > 1: none.** The largest fitted value is 0.727.
- **ŷ < 0: 1,508 of 5,000 customers (30.2%).** The smallest is −0.286.

Examples of customers with ŷ < 0 (the ten lowest):

| Row | Loan | Income | Family | CCAvg | Education | ŷ |
|---|---|---|---|---|---|---|
| 1483 | 0 | 8 | 1 | 0.1 | 1 | −0.2856 |
| 1721 | 0 | 8 | 1 | 0.3 | 1 | −0.2829 |
| 4417 | 0 | 8 | 1 | 0.3 | 1 | −0.2829 |
| 1771 | 0 | 9 | 1 | 0.1 | 1 | −0.2823 |
| 2599 | 0 | 9 | 1 | 0.2 | 1 | −0.2809 |
| 1807 | 0 | 10 | 1 | 0.1 | 1 | −0.2789 |
| 180 | 0 | 11 | 1 | 0.1 | 1 | −0.2755 |
| 1224 | 0 | 11 | 1 | 0.2 | 1 | −0.2742 |
| 4267 | 0 | 11 | 1 | 0.2 | 1 | −0.2742 |
| 3917 | 0 | 12 | 1 | 0.2 | 1 | −0.2708 |

These are low-income, single-person, undergraduate customers with very little credit card spending. A negative "probability" has no meaning. This is the main weakness of the LPM, and it is why we fit a logit model next.

---

## Task 2: Logit Model

### Q4. Logit: `glm(Loan ~ Income + Family + CCAvg + Education, family = binomial)`

```
              Estimate Std. Error z value Pr(>|z|)
(Intercept) -13.177833   0.517777 -25.451  < 2e-16 ***
Income        0.059791   0.002687  22.255  < 2e-16 ***
Family        0.587079   0.071275   8.237  < 2e-16 ***
CCAvg         0.162679   0.040505   4.016 5.91e-05 ***
Education2    3.910609   0.251037  15.578  < 2e-16 ***
Education3    3.933173   0.244329  16.098  < 2e-16 ***

    Null deviance: 3162.0  on 4999  degrees of freedom
Residual deviance: 1334.8  on 4994  degrees of freedom
AIC: 1346.8
```

Every variable raises the log-odds of accepting the loan, and every coefficient is significant at the 0.1% level.

### Q5. Confusion matrix and Percent Correctly Predicted

Threshold = share of successes in the data = 480 / 5000 = **0.096**. A customer is predicted to accept (ŷ = 1) when the logit predicted probability is above 0.096.

| | Predicted 0 | Predicted 1 | Total |
|---|---|---|---|
| **Actual 0** | 4,001 | 519 | 4,520 |
| **Actual 1** | 61 | 419 | 480 |

- **PCP overall** = (4,001 + 419) / 5,000 = **88.40%**
- **PCP for y = 1** = 419 / 480 = **87.29%**
- **PCP for y = 0** = 4,001 / 4,520 = **88.52%**

Using the low threshold (0.096 instead of 0.5) means the model catches most of the customers who actually accepted, while still classifying 88.5% of the non-acceptors correctly.

### Q6. Predicted probability at the means with Education = "2"

Evaluation point: Income = 73.7742, Family = 2.3964, CCAvg = 1.937938 (sample means), Education2 = 1, Education3 = 0.

**By the formula.** The logit model says P(Loan = 1 | x) = G(z) = e^z / (1 + e^z), where z = xβ̂.

```
z = β0 + β_Income·Income + β_Family·Family + β_CCAvg·CCAvg + β_Edu2·1 + β_Edu3·0
  = −13.177833 + 0.059791×73.7742 + 0.587079×2.3964 + 0.162679×1.937938 + 3.910609×1 + 3.933173×0
  = −13.177833 + 4.411015 + 1.406876 + 0.315262 + 3.910609 + 0
  = −3.134071

P = e^(−3.134071) / (1 + e^(−3.134071)) = 0.043540 / 1.043540 = 0.041724
```

**With `predict()`:**

```r
new_x <- data.frame(Income = mean(loan$Income), Family = mean(loan$Family),
                    CCAvg = mean(loan$CCAvg),
                    Education = factor("2", levels = levels(loan$Education)))
predict(logit, newdata = new_x, type = "response")   # 0.041724
```

Both methods give **P(Loan = 1) = 0.0417**, about a 4.2% chance of accepting the offer.

### Q7. LPM and logit coefficients side by side

| Variable | LPM | Logit | Logit / LPM |
|---|---|---|---|
| (Intercept) | −0.345541 | −13.177833 | – |
| Income | 0.003367 | 0.059791 | 17.8 |
| Family | 0.031602 | 0.587079 | 18.6 |
| CCAvg | 0.013735 | 0.162679 | 11.8 |
| Education2 | 0.151743 | 3.910609 | 25.8 |
| Education3 | 0.160547 | 3.933173 | 24.5 |

**No, the coefficients are not directly comparable.** They measure different things:

- **LPM:** β_j is the change in the **probability** of acceptance for a one-unit increase in x_j. It is the same for every customer.
- **Logit:** β_j is the change in the **linear index z = xβ, the log-odds** ln[P / (1 − P)], for a one-unit increase in x_j. The effect on the probability is g(z)·β_j, where g(z) = e^z / (1 + e^z)². This depends on where x is evaluated, and g(z) ≤ 0.25, so logit coefficients are always on a larger scale.

What the two models do agree on:

- **Signs and significance.** All five slopes are positive and highly significant in both models.
- **Relative size.** Ratios of coefficients are similar across the two models. For example, Family / Income is 0.0316 / 0.00337 ≈ 9.4 in the LPM and 0.587 / 0.0598 ≈ 9.8 in the logit.

The usual rule of thumb (logit β ≈ 4 × LPM β, from the maximum g = 0.25) does not hold here: the ratios are 12–26. Only 9.6% of customers accept, so most of them sit in the flat lower tail of the logistic curve, where g(z) is much smaller than 0.25.

To compare the two models on the same scale, we have to convert the logit coefficients into partial effects on the probability (Q8).

### Q8. Partial effects

Evaluation point: the same as Q6 (Income, Family, CCAvg at their means; Education = "2"). Here z = −3.134071, G(z) = 0.041724, and

g(z) = e^z / (1 + e^z)² = 0.043540 / 1.043540² = **0.039983**

**Continuous variables (logit):** partial effect = g(z)·β_j

- Income: 0.039983 × 0.059791 = **0.002391**
- Family: 0.039983 × 0.587079 = **0.023473**
- CCAvg: 0.039983 × 0.162679 = **0.006504**

**Education dummies (logit):** these are discrete, so the partial effect is the change in the predicted probability relative to the base level (Undergrad), with the other variables held at their means.

- P(Edu = 1) = 0.000871, P(Edu = 2) = 0.041724, P(Edu = 3) = 0.042635
- Education2: 0.041724 − 0.000871 = **0.040852**
- Education3: 0.042635 − 0.000871 = **0.041764**

(The calculus approximation g(z)·β gives 0.1564 and 0.1573 for the two dummies. That overstates the effect a lot, because the jump in z from a dummy (≈ 3.9) is far too large for a derivative to approximate. The discrete difference is the correct partial effect for a dummy variable.)

**LPM:** the partial effects are just the β coefficients, and they are the same for every customer.

| Variable | LPM partial effect | Logit partial effect (at means, Edu = 2) |
|---|---|---|
| Income (+$1,000) | 0.003367 | 0.002391 |
| Family (+1 person) | 0.031602 | 0.023473 |
| CCAvg (+$1,000/month) | 0.013735 | 0.006504 |
| Education2 vs. 1 | 0.151743 | 0.040852 |
| Education3 vs. 1 | 0.160547 | 0.041764 |

**Are they comparable?** Yes. Unlike the raw coefficients in Q7, both columns are now in the same units: the change in the probability of accepting the loan. Examples:

- **Income:** an extra $1,000 of income raises the acceptance probability by 0.34 percentage points in the LPM and by 0.24 points in the logit.
- **Family:** one more family member raises it by 3.2 points in the LPM and by 2.3 points in the logit.

Signs agree and magnitudes are in the same range for Income and Family. The logit effects are smaller at this point, and much smaller for Education, for two reasons:

1. **The logit partial effect depends on x.** This customer's predicted probability is only 4.2%, which puts them on the flat lower part of the logistic curve (g(z) = 0.04, compared with the maximum of 0.25). For a customer with a higher predicted probability, the logit partial effects would be larger.
2. **The LPM uses one constant slope for everyone.** It is roughly an average effect across the whole sample. For low-probability customers it overstates the effect, and it is the same linearity that produces the negative fitted probabilities in Q3. The LPM's prediction at this same point is 0.157, compared with 0.042 from the logit.

The partial effects can be compared, but the logit ones only hold at the chosen evaluation point. The LPM's single number is roughly a sample-wide average.
