# Credit Offer Decision Regret: Exact Multiple-Choice Oracle

A small, dependency-light research benchmark demonstrating why minimizing
forecast mean squared error (MSE) is not the same as minimizing **downstream
decision regret**.

The synthetic bank has an integer risk/resource budget and a menu of candidate
offers for each synthetic customer. It may select **at most one** offer per
customer (or no offer). The objective is a synthetic expected-contribution
score, not a real customer lending decision or calibrated profit estimate.

The downstream optimization oracle uses **pure Python dynamic programming**;
there is no Gurobi, FICO Xpress, Hexaly, or other licensed optimizer. With
budget B, n customers, and k offers each, the algorithm is O(n·k·B) for
nonnegative integer resource costs. This pseudo-polynomial complexity is
intended for small teaching cases only.

## Run

From the repository root:

    python -m dfl_spo.credit_offer_dp --instances 100 --customers 12 --budget 55
    pytest -q tests/test_credit_offer_dp.py

Two synthetic forecasting methods are evaluated on identical independently
generated portfolio instances: independent Gaussian coefficient noise and a
lower-noise but systematically biased alternative. For each instance, the exact
oracle is solved with hidden true coefficients, then separately with the
forecast coefficients. The selected forecast decision is evaluated on the
hidden truth to measure regret:

    regret = true_profit(oracle_true) - true_profit(policy_from_forecast)
    normalized_regret = regret / max(1, abs(true_profit(oracle_true)))

The report includes mean and sample standard deviation of prediction MSE,
regret, normalized regret and zero-regret fraction. These are descriptive
statistics across seeded synthetic instances, **not** confidence intervals.

The unit tests include a constructed example in which a forecast with lower
MSE causes a worse decision than one with substantially higher MSE. This is a
counterexample to using coefficient accuracy as a substitute for policy quality.

## Scope

This is an **oracle and evaluation** benchmark, not an implementation of SPO+
training or differentiable MILP learning. It complements the existing SPO+
examples in the repository and provides a possible downstream oracle for future
training experiments. Integer costs and small budgets are required for tractable
DP. No capital adequacy, real credit underwriting, regulatory rules, customer
fairness, or selection-bias estimates are represented.

**Research/education only; entirely synthetic.**
