---
layout: mathjax
title:  Hypothesis Testing
date:   2023-02-26
---

# 1. Definition of hypothesis testing

> [!NOTE] Hypothesis Testing
>-  To test the statistical hypothesis whether this is the actual discovery or just happen by chance.
>1. Determine a model
>2. Determine a **null** and **alternative** hypothesis
>3. Determine a **test statistic**
>4. Determine a **significance level $$\alpha$$**


The criteria to reject the null hypothesis
We can **reject** the Null Hypothesis when
the $$\text{p-value}< \alpha \Rightarrow  \quad t_{q_{\alpha/2}} < t_{p-value/2}$$
- ![Alt text](/images/R-Null_hypothesis_alpha_and_p-value.png)

> A Test
>- **A test is a statistic** $$\psi \in \{0,1\}, \psi \text{ takes value 0 or 1 only}$$, that *does not* depend on unknown quantities and such that: $$\psi = \mathbb 1\{\mathbb R\} \text{, where R is an event called rejection region}$$
>	- If $$\psi = 0, H_0$$ is not rejected;
>	- If $$\psi = 1, H_0$$ is rejected
>- **Example**
>	- Waiting time in the ER: $$\psi - \mathbb 1\{ \bar X_n > c\}$$
>	- Kiss example: $$\psi - \mathbb 1\{  \mid \bar X_n -0.5 \mid > c\}$$
>	- Clinical trials: $$\psi - \mathbb 1\{\bar X_n - \bar Y_m > c\}$$


