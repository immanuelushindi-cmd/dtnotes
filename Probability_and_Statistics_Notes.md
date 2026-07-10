# Probability & Statistics for Data Science — Worked Notes
### eMobilis · Data Science & AI

These notes cover the same material as the notebook, but as pure explanation and hand-worked math — no code. Every topic follows: **concept → formula → worked example**, using small numbers you can check by hand.

---

## 1. Probability Foundations

**Experiment** — a process with an uncertain outcome (rolling a die, sending out a matatu trip).
**Outcome** — one single result of the experiment.
**Sample space (S)** — the set of *all* possible outcomes.
**Event (A)** — any subset of the sample space you care about.
**Notation:** P(A) = probability of event A, always between 0 and 1.

**Worked example:** Rolling a fair die.
S = {1, 2, 3, 4, 5, 6}
Event A = "rolling an even number" = {2, 4, 6}
$$P(A) = \frac{\text{outcomes in A}}{\text{outcomes in S}} = \frac{3}{6} = 0.5$$

---

## 2. Types of Probability

| Type | How it's found | Worked example |
|---|---|---|
| **Classical** | Reasoned from equally-likely outcomes, no data needed | Fair die: P(rolling a 4) = 1/6 = 0.167 |
| **Empirical** | Relative frequency from observed data | Out of 500 past trips, 343 were profitable → P(profitable) = 343/500 = 0.686 |
| **Subjective** | A person's degree of belief | "I'm 80% sure this new route succeeds" — a manager's judgement, not derived from counting |

---

## 3. Types of Events

| Type | Meaning | Example |
|---|---|---|
| Mutually exclusive | Cannot both happen at once | A trip is "Profitable" or "Not profitable" — never both |
| Independent | One doesn't affect the other's probability | Two separate die rolls |
| Dependent | One changes the other's probability | Rain today changes the chance of a late trip |
| Complementary | "A" and "not A" — together sum to 1 | P(late) + P(not late) = 1 |
| Collectively exhaustive | Together, the events cover the whole sample space | {Profitable, Not profitable} covers every trip outcome |

### The mutually exclusive vs. independent mix-up

These sound similar but are opposites. Here's the proof, worked both ways.

**Mutually exclusive example.** Rolling one die. Event A = {1, 2}, Event B = {3, 4}.
$$P(A) = \frac{2}{6} = 0.333 \qquad P(B) = \frac{2}{6} = 0.333$$
Since A and B share no outcomes, $P(A \text{ and } B) = 0$, so:
$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)} = \frac{0}{0.333} = 0$$
This is **not** equal to P(A) = 0.333 — knowing B happened completely changed our belief about A (crashed it to zero). **Mutually exclusive events are always dependent.**

**Independent example.** Two separate die rolls. Event A = "first roll is 6", Event B = "second roll is 6".
$$P(A) = \frac{1}{6} \qquad P(B) = \frac{1}{6} \qquad P(A \text{ and } B) = \frac{1}{6}\times\frac{1}{6} = \frac{1}{36}$$
$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)} = \frac{1/36}{1/6} = \frac{1}{6} = P(A)$$
Here, knowing B happened changed *nothing* about A. **The key rule:** if two events are mutually exclusive (and both have a nonzero chance), they can never be independent — mutual exclusivity is the most extreme possible dependence.

---

## 4. Probability Rules

**Complement rule:**
$$P(\text{not } A) = 1 - P(A)$$

**Addition rule** (P(A or B)):
$$P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$$

**Multiplication rule** (P(A and B)):
$$P(A \text{ and } B) = P(A) \times P(B \mid A)$$
(simplifies to $P(A) \times P(B)$ if independent)

**Worked example** — using 300 matatu trips, split by weather and lateness:

| | Late | On-time | Row total |
|---|---|---|---|
| Rainy | 60 | 40 | 100 |
| Dry | 20 | 180 | 200 |
| **Total** | **80** | **220** | **300** |

$$P(\text{Rain}) = \frac{100}{300} = 0.333 \qquad P(\text{Late}) = \frac{80}{300} = 0.267 \qquad P(\text{Rain and Late}) = \frac{60}{300} = 0.20$$

**Addition rule:**
$$P(\text{Rain or Late}) = 0.333 + 0.267 - 0.20 = 0.40$$

**Complement rule:**
$$P(\text{not Late}) = 1 - 0.267 = 0.733$$

**Multiplication rule (dependent case), verified from the same table:**
$$P(\text{Rain and Late}) = P(\text{Rain}) \times P(\text{Late} \mid \text{Rain}) = 0.333 \times 0.60 = 0.20 \checkmark$$

**Multiplication rule (independent case):** two separate die rolls, P(both show 6):
$$P(6 \text{ and } 6) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36} = 0.0278$$

---

## 5. Conditional Probability

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

**Worked example**, using the same 300-trip table above:

$$P(\text{Late} \mid \text{Rainy}) = \frac{P(\text{Rainy and Late})}{P(\text{Rainy})} = \frac{60/300}{100/300} = \frac{60}{100} = 0.60$$

$$P(\text{Late} \mid \text{Dry}) = \frac{20/300}{200/300} = \frac{20}{200} = 0.10$$

Since 0.60 ≠ 0.10, rain clearly raises the chance of lateness — weather and lateness are dependent.

---

## 6. Bayes' Theorem & Bayesian Thinking

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}, \qquad P(B) = P(B\mid A)P(A) + P(B\mid \text{not }A)P(\text{not }A)$$

**Bayesian thinking:** treat P(A) as a **prior** belief, update it with new evidence B to get a **posterior** belief P(A|B) — a degree-of-belief view of probability, contrasted with the classical/frequentist "long-run frequency" view.

**Worked example — matatu breakdowns:**

Given:
- P(breakdown | old) = 0.25
- P(breakdown | new) = 0.05
- P(old) = 0.40 (prior), P(new) = 0.60

**Step 1 — total probability of a breakdown:**
$$P(\text{breakdown}) = (0.25)(0.40) + (0.05)(0.60) = 0.10 + 0.03 = 0.13$$

**Step 2 — Bayes' theorem, the reverse question:** given a breakdown happened, what's the chance it was an old matatu?
$$P(\text{old} \mid \text{breakdown}) = \frac{(0.25)(0.40)}{0.13} = \frac{0.10}{0.13} = 0.769$$

The prior belief P(old) = 0.40 jumps to a posterior of 0.769 once we see the breakdown — that jump *is* Bayesian updating.

---

## 7. Random Variables & the Two Families of Distributions

**Random variable** — a function assigning a number to every outcome of a random process. Example: X = number of passengers on a trip.

| Family | Type of values | Example |
|---|---|---|
| **Discrete** | Countable, separate values (0,1,2,...) | Number of full trips out of 10 |
| **Continuous** | Any value in a range | Fare amount, waiting time |

Every distribution below belongs to one of these two families, which determines whether it's described with a PMF or a PDF.

---

## 8. PMF, PDF, CDF, PPF

| Function | Full name | Used for | Answers |
|---|---|---|---|
| PMF | Probability Mass Function | Discrete | "P(exactly this value)?" |
| PDF | Probability Density Function | Continuous | Relative likelihood; area under it over a range = probability |
| CDF | Cumulative Distribution Function | Both | "P(this value or less)?" |
| PPF | Percent Point Function (inverse CDF / quantile) | Both | "What value is at this percentile?" |

**Worked example** — Binomial with n = 5 trials, p = 0.5 (e.g. 5 coin flips). PMF formula:
$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

| k | PMF | Cumulative (CDF) |
|---|---|---|
| 0 | 1/32 = 0.03125 | 0.03125 |
| 1 | 5/32 = 0.15625 | 0.1875 |
| 2 | 10/32 = 0.3125 | 0.5000 |
| 3 | 10/32 = 0.3125 | 0.8125 |
| 4 | 5/32 = 0.15625 | 0.96875 |
| 5 | 1/32 = 0.03125 | 1.00000 |

- **PMF** at k=2: P(X=2) = 0.3125
- **CDF** at k=2: P(X≤2) = 0.03125+0.15625+0.3125 = 0.500
- **PPF** at 0.9: the smallest k whose CDF ≥ 0.9 → CDF(3)=0.8125, CDF(4)=0.96875, so **PPF(0.9) = 4**

---

## 9. Discrete Distributions

**Bernoulli** — one yes/no trial with success probability p. Example: does *one* trip break down (p=0.3)?
$$P(X=1)=0.3 \qquad P(X=0)=0.7$$

**Binomial** — number of successes in n independent Bernoulli trials.
$$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
**Worked example:** n=5 trips, p=0.4 chance of breakdown each. P(exactly 2 breakdowns)?
$$P(X=2) = \binom{5}{2}(0.4)^2(0.6)^3 = 10 \times 0.16 \times 0.216 = 0.3456$$

**Poisson** — number of events in a fixed interval, given an average rate λ.
$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$$
**Worked example:** matatus arriving at a stage, λ=3 per hour. P(exactly 2 arrivals)?
$$P(X=2) = \frac{e^{-3}(3)^2}{2!} = \frac{0.0498 \times 9}{2} = 0.224$$

---

## 10. Continuous Distributions

**Uniform** — every value in [a,b] equally likely.
$$P(c < X < d) = \frac{d-c}{b-a}$$
**Worked example:** matatus depart every 10 minutes (a=0, b=10). P(you arrive between minute 2 and 5 of the cycle)?
$$P(2<X<5) = \frac{5-2}{10-0} = \frac{3}{10} = 0.30$$

**Exponential** — time until the next random event, given rate λ (mean = 1/λ).
$$P(X>x) = e^{-\lambda x}$$
**Worked example:** average wait for the next matatu is 8 minutes (λ = 1/8). P(waiting more than 10 minutes)?
$$P(X>10) = e^{-10/8} = e^{-1.25} = 0.287$$

**Normal** — the bell curve, defined by mean μ and standard deviation σ. Covered in depth via standardisation next.

---

## 11. Standardisation & the Z-score

$$z = \frac{x-\mu}{\sigma}$$

**Worked example:** fares are Normal(mean=150, std=20). A fare of 180 KES:
$$z = \frac{180-150}{20} = \frac{30}{20} = 1.5$$
This fare is 1.5 standard deviations above average. A z beyond ±2 is already unusual (roughly the top/bottom 2.5% under a normal distribution); beyond ±3 is very rare (roughly 0.1%).

---

## 12. The t-distribution and Chi-square Distribution

Both are **sampling distributions** — they describe a statistic computed from a sample, not raw data.

**t-distribution** — like Normal but with fatter tails; used when the population standard deviation is unknown and estimated from a small sample. As sample size grows, it converges to Normal.

**Chi-square distribution** — always positive, right-skewed; built from sums of squared standardised values. Powers the chi-square test and variance-related tests.

**Worked example — critical values from standard tables:**
- t-distribution, df = 9, two-tailed α = 0.05 → critical value = **±2.262**
- Chi-square distribution, df = 4, α = 0.05 → critical value = **9.488**

These are the cutoffs a test statistic must cross to be judged "significant" at that sample size / degrees of freedom.

---

## 13. Why Probability Distributions Matter — Two Worked Examples

**Example 1 — Inventory planning with Poisson.** Breakdowns average λ=3/week. What's P(needing 6 or more spare parts this week)?
$$P(X \ge 6) = 1 - P(X \le 5) = 1 - 0.916 = 0.084$$
About an 8.4% chance — a number you can actually plan a spares budget around.

**Example 2 — Fraud/anomaly detection with z-scores.** Fares ~ Normal(150, 20). A fare of 230 KES:
$$z = \frac{230-150}{20} = 4.0$$
$$P(Z>4) \approx 0.0000317$$
Essentially impossible under normal operation — this fare is worth flagging for review.

**The general point:** a distribution turns a vague "that seems unusual" into an exact, defensible probability.

---

## 14. The Central Limit Theorem (CLT)

**Statement:** no matter the shape of the population, the distribution of *sample means* (for reasonably large samples) becomes approximately Normal.

**Where it's used:** justifying normal/t-based tests on non-normal data; building confidence intervals for a mean; any reasoning about a sample mean or sum.

**Standard error formula (how spread out the sample mean is):**
$$SE = \frac{\sigma}{\sqrt{n}}$$

**Worked example:** waiting times have population std σ=8 minutes (skewed, not normal). Take samples of n=40.
$$SE = \frac{8}{\sqrt{40}} = \frac{8}{6.325} = 1.265$$
Even though individual waiting times are skewed, the *mean* of 40 of them will be approximately normally distributed around the true population mean, with this SE.

---

## 15. Standard Error vs Standard Deviation

| | Standard Deviation (SD) | Standard Error (SE) |
|---|---|---|
| Measures | Spread of individual data points | Spread of the sample **mean** across repeated samples |
| Formula | $s = \sqrt{\dfrac{\sum(x_i-\bar x)^2}{n-1}}$ | $SE = \dfrac{s}{\sqrt n}$ |
| Shrinks with bigger n? | No | Yes |

**Worked example:** sample of 5 fares: 130, 150, 160, 140, 170.

$$\bar x = \frac{130+150+160+140+170}{5} = \frac{750}{5} = 150$$

Deviations: −20, 0, 10, −10, 20. Squared: 400, 0, 100, 100, 400. Sum = 1000.

$$s = \sqrt{\frac{1000}{5-1}} = \sqrt{250} = 15.81$$

$$SE = \frac{15.81}{\sqrt{5}} = \frac{15.81}{2.236} = 7.07$$

The SD (15.81) describes how spread out individual fares are; the SE (7.07) — much smaller — describes how precisely we've pinned down the *mean* fare.

---

## 16. Confidence Intervals

**What it is:** a range of plausible values for a population parameter, built around a sample estimate at a stated confidence level (usually 95%).

**Correct interpretation:** if you repeated the sampling process many times, about 95% of the intervals built this way would contain the true population mean.

$$CI = \bar x \pm t_{crit} \times SE$$

**Worked example**, continuing the 5-fare sample above (mean=150, SE=7.07, df=4):

t-critical for df=4, 95% confidence (two-tailed) = **2.776**

$$\text{margin} = 2.776 \times 7.07 = 19.63$$

$$CI = 150 \pm 19.63 = (130.4,\ 169.6)$$

We're 95% confident the true mean fare lies between 130.4 and 169.6 KES.

---

## 17. Hypothesis Testing — Framework

**What it is:** a structured procedure for deciding whether a pattern in a sample reflects something real in the population, or could be random noise.

**The logic:** assume the boring explanation (H₀) is true, then ask "if H₀ really were true, how unlikely would this data be?" A very unlikely result casts doubt on H₀ — this is reasoning by contradiction, not direct proof.

**Which distribution, and why:**

| Situation | Distribution | Why |
|---|---|---|
| Comparing a mean, population SD unknown | t-distribution | Extra uncertainty from estimating SD → fatter tails |
| Comparing a mean, SD known / n very large | Normal (z) | CLT guarantees the sample mean is normal |
| Categorical relationships / variance | Chi-square | Built from squared standardised values |

**The 5 steps:**
1. State H₀ (nothing interesting) and H₁ (what you suspect).
2. Choose significance level α (usually 0.05).
3. Choose the test, compute the test statistic.
4. Compute the p-value.
5. Decide: if p-value < α, reject H₀; otherwise, fail to reject H₀.

**P-values:** the p-value is the probability of seeing data this extreme *if H₀ were true* — **not** the probability that H₀ is true.

---

## 18. Type I and Type II Errors

| | H₀ actually True | H₀ actually False |
|---|---|---|
| **Reject H₀** | Type I error (false positive), probability = α | Correct |
| **Fail to reject H₀** | Correct | Type II error (false negative), probability = β |

- **Type I:** concluding there's an effect when there isn't (false alarm).
- **Type II:** missing a real effect that exists (a miss).

**Critical value / critical region:** the cutoff on the test statistic's distribution beyond which you reject H₀ — sized so the area past it equals α. E.g. for a t-test with df=9, two-tailed α=0.05, the critical value is ±2.262 (from Module 12) — any observed |t| beyond that triggers rejection.

---

## 19. One-Sample t-test

**When to use it:** comparing one sample's mean to a known/claimed value.

**Formula:**
$$t = \frac{\bar x - \mu_0}{s/\sqrt n}$$

**Worked example:** manufacturer claims average fuel use is 4.0L/trip. Six measured trips: 4.5, 4.0, 4.8, 4.2, 4.6, 4.1.

$$\bar x = \frac{4.5+4.0+4.8+4.2+4.6+4.1}{6} = \frac{26.2}{6} = 4.367$$

Deviations from mean: 0.133, −0.367, 0.433, −0.167, 0.233, −0.267. Squared and summed ≈ 0.494.

$$s = \sqrt{\frac{0.494}{6-1}} = \sqrt{0.0988} = 0.314$$

$$t = \frac{4.367 - 4.0}{0.314/\sqrt 6} = \frac{0.367}{0.128} = 2.86$$

df = 6−1 = 5. Critical value (two-tailed, α=0.05, df=5) = 2.571. Since |2.86| > 2.571, and the corresponding p-value ≈ **0.035** (< 0.05):

**Reject H₀** — the fleet's actual fuel use is significantly different from the manufacturer's claim of 4.0L.

---

## 20. Two-Sample t-test

**When to use it:** comparing means of two independent groups.

**Formula (equal-variance/pooled version):**
$$t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_p^2\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}, \qquad s_p^2 = \frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$$

**Worked example:** morning fares (140,150,145,155,148) vs afternoon fares (160,170,165,175,168).

$$\bar x_1 = 147.6, \quad \bar x_2 = 167.6$$
$$s_1 = s_2 = 5.59 \quad (\text{same spread in this example})$$
$$s_p^2 = \frac{(4)(5.59^2)+(4)(5.59^2)}{5+5-2} = \frac{4(31.3)+4(31.3)}{8} = 31.3$$
$$t = \frac{147.6-167.6}{\sqrt{31.3(1/5+1/5)}} = \frac{-20}{3.538} = -5.65$$

df = 5+5−2 = 8. Critical value (two-tailed, α=0.05, df=8) = 2.306. Since |−5.65| > 2.306, p-value ≈ **0.0005**:

**Reject H₀** — afternoon fares are genuinely higher than morning fares.

---

## 21. Chi-Square Test of Independence

**When to use it:** testing whether two categorical variables are related.

**Formula:**
$$\chi^2 = \sum \frac{(O-E)^2}{E}, \qquad E = \frac{\text{row total}\times\text{column total}}{N}$$

**Worked example:** does age group relate to breakdowns? Observed counts (100 matatus):

| | Broke down | Didn't | Row total |
|---|---|---|---|
| Old | 12 | 28 | 40 |
| New | 8 | 52 | 60 |
| **Total** | **20** | **80** | **100** |

Expected counts (if independent):
$$E_{\text{Old,Broke}} = \frac{40\times20}{100}=8 \qquad E_{\text{Old,Didn't}}=\frac{40\times80}{100}=32$$
$$E_{\text{New,Broke}} = \frac{60\times20}{100}=12 \qquad E_{\text{New,Didn't}}=\frac{60\times80}{100}=48$$

$$\chi^2 = \frac{(12-8)^2}{8}+\frac{(28-32)^2}{32}+\frac{(8-12)^2}{12}+\frac{(52-48)^2}{48}$$
$$= 2.0 + 0.5 + 1.333 + 0.333 = 4.167$$

df = (rows−1)(cols−1) = 1. Critical value (α=0.05, df=1) = 3.841. Since 4.167 > 3.841, p-value ≈ **0.041** (< 0.05):

**Reject H₀** — age group and breakdowns are related.

---

## 22. A/B Testing

**What it is:** hypothesis testing applied to comparing two versions (control A vs variant B), usually on a proportion or mean. Key vocabulary: **test statistic** (z or t computed from the data), **critical value** (the significance cutoff), **significance level α**, **confidence level** ($1-\alpha$).

**Two-proportion z-test formula:**
$$z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}, \qquad \hat p = \frac{x_A+x_B}{n_A+n_B}$$

**Worked example:** Pricing Policy A: 50 of 100 trips profitable ($\hat p_A$=0.50). Policy B: 65 of 100 trips profitable ($\hat p_B$=0.65).

$$\hat p = \frac{50+65}{100+100} = \frac{115}{200} = 0.575$$

$$SE = \sqrt{0.575(0.425)\left(\frac{1}{100}+\frac{1}{100}\right)} = \sqrt{0.2444 \times 0.02} = 0.0699$$

$$z = \frac{0.65-0.50}{0.0699} = 2.15$$

Critical value (two-tailed, α=0.05) = ±1.96. Since |2.15| > 1.96, p-value ≈ **0.032** (< 0.05):

**Reject H₀** — Policy B has a significantly higher profitable-trip rate.

---

## 23. Confusion Matrix — Accuracy, Precision, Recall

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | TN | FP |
| **Actual: Yes** | FN | TP |

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} \quad \text{Precision} = \frac{TP}{TP+FP} \quad \text{Recall} = \frac{TP}{TP+FN}$$

**Worked example:** a breakdown-prediction model tested on 200 matatus: TP=40, FP=10, FN=15, TN=135.

$$\text{Accuracy} = \frac{40+135}{200} = 0.875$$
$$\text{Precision} = \frac{40}{40+10} = 0.80$$
$$\text{Recall} = \frac{40}{40+15} = 0.727$$

Interpretation: the model is right 87.5% of the time overall; when it predicts "will break down," it's correct 80% of the time (precision); of all matatus that actually broke down, it caught 72.7% of them (recall).

---

## 24. Variance, Covariance, Correlation

$$\text{Var}(X) = \frac{\sum(x_i-\bar x)^2}{n-1} \qquad \text{Cov}(X,Y) = \frac{\sum(x_i-\bar x)(y_i-\bar y)}{n-1} \qquad r = \frac{\text{Cov}(X,Y)}{s_X s_Y}$$

**Worked example:** trips per day (X) and fare revenue in hundreds (Y) for 5 days:

| x | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| y | 3 | 7 | 5 | 10 | 12 |

$$\bar x = 6, \qquad \bar y = 7.4$$

| $x_i-\bar x$ | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| $y_i-\bar y$ | −4.4 | −0.4 | −2.4 | 2.6 | 4.6 |
| product | 17.6 | 0.8 | 0.0 | 5.2 | 18.4 |

Sum of products = 42.0

$$\text{Cov}(X,Y) = \frac{42.0}{5-1} = 10.5$$

$$\text{Var}(X) = \frac{(-4)^2+(-2)^2+0^2+2^2+4^2}{4} = \frac{40}{4} = 10.0 \implies s_X = 3.16$$
$$\text{Var}(Y) = 13.3 \implies s_Y = 3.65$$

$$r = \frac{10.5}{3.16 \times 3.65} = \frac{10.5}{11.53} = 0.91$$

A strong positive correlation — as trips increase, revenue tends to increase too.

---

## 25. Skewness & Kurtosis

**Skewness** — asymmetry. Zero = symmetric. Positive = long tail to the right. Negative = long tail to the left.

**Kurtosis (excess)** — "tailedness" relative to Normal. 0 matches Normal; positive means heavier tails/more extreme outliers; negative means lighter tails.

**Worked example:** waiting times (minutes) for 6 trips: 2, 3, 3, 4, 5, 20 (one long outlier).

$$\bar x = \frac{2+3+3+4+5+20}{6} = \frac{37}{6} = 6.17$$

This single value of 20 pulls the mean well above where "most" of the data sits (2–5) — a visual sign of right skew. Computing the full skewness statistic (a formula using cubed deviations) confirms it:

**Skewness ≈ 1.70** (strongly positive/right-skewed)
**Excess kurtosis ≈ 1.04** (heavier tails than Normal — driven by that one outlier)

Compare to a symmetric dataset like fares clustered tightly around a mean, which would have skewness near 0.

---

## 26. The Cardinal Sin: Correlation ≠ Causation

A strong correlation only tells you two variables move together. Three explanations are always possible:
1. A causes B
2. B causes A
3. A hidden confounder C causes both

**Worked example:** fuel used and revenue earned per matatu, across 5 days, are strongly correlated (r ≈ 0.99). It's tempting to conclude "burning more fuel causes more revenue." But look at the confounder — **trips per day**:

- corr(trips, fuel) ≈ 0.99
- corr(trips, revenue) ≈ 1.00

Both fuel and revenue are independently driven by how many trips were made that day. Trips is the real cause; fuel and revenue are just two side-effects that happen to move together. Treating their correlation as causation would lead to a wrong conclusion (e.g. "just burn more fuel to earn more").

**The rule:** correlation can motivate further investigation (like a proper A/B test), but on its own it is never proof of cause and effect.

---

# Quick Reference

| Concept | Formula |
|---|---|
| Complement rule | $P(\text{not }A) = 1-P(A)$ |
| Addition rule | $P(A\text{ or }B) = P(A)+P(B)-P(A\text{ and }B)$ |
| Multiplication rule | $P(A\text{ and }B) = P(A)\times P(B\mid A)$ |
| Conditional probability | $P(A\mid B) = \dfrac{P(A\text{ and }B)}{P(B)}$ |
| Bayes' theorem | $P(A\mid B) = \dfrac{P(B\mid A)P(A)}{P(B)}$ |
| Binomial PMF | $\binom{n}{k}p^k(1-p)^{n-k}$ |
| Poisson PMF | $\dfrac{e^{-\lambda}\lambda^k}{k!}$ |
| Uniform probability | $\dfrac{d-c}{b-a}$ |
| Exponential tail | $e^{-\lambda x}$ |
| Z-score | $z = \dfrac{x-\mu}{\sigma}$ |
| Standard error | $SE = \dfrac{s}{\sqrt n}$ |
| Confidence interval | $\bar x \pm t_{crit}\times SE$ |
| One-sample t | $t = \dfrac{\bar x - \mu_0}{s/\sqrt n}$ |
| Two-sample t (pooled) | $t = \dfrac{\bar x_1-\bar x_2}{\sqrt{s_p^2(1/n_1+1/n_2)}}$ |
| Chi-square statistic | $\sum \dfrac{(O-E)^2}{E}$ |
| Two-proportion z | $z = \dfrac{\hat p_B - \hat p_A}{\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}}$ |
| Accuracy / Precision / Recall | $\frac{TP+TN}{\text{all}}$ / $\frac{TP}{TP+FP}$ / $\frac{TP}{TP+FN}$ |
| Variance | $\frac{\sum(x_i-\bar x)^2}{n-1}$ |
| Covariance | $\frac{\sum(x_i-\bar x)(y_i-\bar y)}{n-1}$ |
| Correlation | $\dfrac{\text{Cov}(X,Y)}{s_X s_Y}$ |
