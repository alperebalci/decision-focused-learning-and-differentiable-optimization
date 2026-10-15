"""Exactness and decision-quality tests for the multiple-choice credit offer oracle."""

import math
import random
from itertools import product

from dfl_spo.credit_offer_dp import (
    Offer, make_instance, metrics, realized_profit, run_experiment, solve_exact,
)


def test_dynamic_programming_matches_brute_force():
    offers = [Offer(i, j, j + 1, (i + 1) * 2 + j) for i in range(3) for j in range(2)]
    values = [o.true_profit for o in offers]
    solution = solve_exact(offers, values, 6)
    brute = 0.0
    for choices in product((-1, 0, 1), repeat=3):
        selected = [i * 2 + ch for i, ch in enumerate(choices) if ch >= 0]
        if sum(offers[i].cost for i in selected) <= 6:
            brute = max(brute, sum(values[i] for i in selected))
    assert math.isclose(realized_profit(offers, solution), brute)
    assert len({offers[i].customer for i in solution}) == len(solution)


def test_lower_mse_does_not_guarantee_lower_decision_regret():
    offers = [Offer(0, 0, 2, 9.0), Offer(0, 1, 2, 8.0)]
    low_mse_wrong_action = metrics(offers, [8.0, 9.0], budget=2)
    high_mse_right_action = metrics(offers, [20.0, 3.0], budget=2)
    assert low_mse_wrong_action["prediction_mse"] < high_mse_right_action["prediction_mse"]
    assert low_mse_wrong_action["regret"] == 1.0
    assert high_mse_right_action["regret"] == 0.0


def test_generated_instance_and_summary_are_reproducible():
    offers, a, b = make_instance(random.Random(8), customers=6, choices=3)
    for forecast in (a, b):
        m = metrics(offers, forecast, budget=25)
        assert m["regret"] >= -1e-9
        assert m["achieved_profit"] <= m["oracle_profit"] + 1e-9
    r = run_experiment(instances=10, customers=6, budget=25, seed=8)
    assert r == run_experiment(instances=10, customers=6, budget=25, seed=8)
    assert r["methods"]["independent_noise"]["prediction_mse"]["mean"] >= 0
