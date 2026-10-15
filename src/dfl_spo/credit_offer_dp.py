"""Exact small-budget credit-offer oracle for decision-regret research.

Pure Python dynamic programming; no commercial or third-party solver required.
The financial parameters are synthetic and not underwriting recommendations.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass(frozen=True)
class Offer:
    customer: int
    offer: int
    cost: int
    true_profit: float


def solve_exact(offers: list[Offer], predicted_profits: list[float], budget: int) -> tuple[int, ...]:
    """Maximize supplied coefficients with at most one option per customer.

    Exact for integer nonnegative resource costs; O(customers * options * budget).
    Zero offers per customer is feasible, so negative coefficients are skipped.
    """
    if budget < 0 or len(offers) != len(predicted_profits) or not offers:
        raise ValueError("nonnegative budget and aligned nonempty offers are required")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(offers):
        if item.cost < 0 or item.offer < 0:
            raise ValueError("cost/offer identifiers must be nonnegative")
        groups[item.customer].append(index)
    # State: resource consumed -> (best objective, selected original row indices).
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for group in groups.values():
        updated = states.copy()  # Include the choice of no offer for this customer.
        for used, (value, chosen) in states.items():
            for index in group:
                new_cost = used + offers[index].cost
                if new_cost > budget:
                    continue
                objective = value + float(predicted_profits[index])
                old = updated.get(new_cost)
                if old is None or objective > old[0] + 1e-12:
                    updated[new_cost] = objective, chosen + (index,)
        states = updated
    return max(states.values(), key=lambda state: state[0])[1]


def realized_profit(offers: list[Offer], decisions: tuple[int, ...]) -> float:
    return sum(offers[i].true_profit for i in decisions)


def make_instance(rng: random.Random, customers: int = 12,
                  choices: int = 3) -> tuple[list[Offer], list[float], list[float]]:
    """Paired noisy profit forecasts for the same synthetic decision instance."""
    if customers < 1 or choices < 1:
        raise ValueError("customers/choices must be positive")
    offers = []
    forecast_a = []
    forecast_b = []
    for customer in range(customers):
        base_profit = rng.uniform(3, 15)
        for choice in range(choices):
            resource = rng.randint(2, 10)
            true_profit = base_profit + 0.40 * resource + rng.uniform(-2.0, 2.0)
            offers.append(Offer(customer, choice, resource, true_profit))
            forecast_a.append(true_profit + rng.gauss(0, 2.6))
            # Smaller stochastic noise, but systematic offer-specific bias.
            forecast_b.append(true_profit + rng.gauss(0, 1.8) + (2.2 if choice == 0 else -0.8))
    return offers, forecast_a, forecast_b


def metrics(offers: list[Offer], forecast: list[float], budget: int) -> dict[str, float]:
    true_values = [o.true_profit for o in offers]
    reference = solve_exact(offers, true_values, budget)
    selected = solve_exact(offers, forecast, budget)
    optimal_profit = realized_profit(offers, reference)
    achieved_profit = realized_profit(offers, selected)
    regret = max(0.0, optimal_profit - achieved_profit)
    return {
        "prediction_mse": mean((a - b) ** 2 for a, b in zip(forecast, true_values)),
        "oracle_profit": optimal_profit,
        "achieved_profit": achieved_profit,
        "regret": regret,
        "normalized_regret": regret / max(1.0, abs(optimal_profit)),
        "zero_regret": float(regret < 1e-9),
    }


def run_experiment(instances: int = 100, customers: int = 12, choices: int = 3,
                   budget: int = 55, seed: int = 42) -> dict:
    if instances < 2:
        raise ValueError("need at least two instances")
    rng = random.Random(seed)
    results = {"independent_noise": [], "biased_noise": []}
    for _ in range(instances):
        offers, a, b = make_instance(rng, customers, choices)
        results["independent_noise"].append(metrics(offers, a, budget))
        results["biased_noise"].append(metrics(offers, b, budget))
    summary = {}
    for label, outcomes in results.items():
        summary[label] = {}
        for field in ("prediction_mse", "regret", "normalized_regret", "zero_regret"):
            vals = [row[field] for row in outcomes]
            summary[label][field] = {
                "mean": mean(vals),
                "sample_std": stdev(vals),
            }
    return {
        "instances": instances,
        "customers_per_instance": customers,
        "offers_per_customer": choices,
        "integer_resource_budget": budget,
        "methods": summary,
        "oracle": "exact multiple-choice dynamic programming",
        "disclaimer": "Synthetic decision-regret benchmark, not bank credit policy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--customers", type=int, default=12)
    parser.add_argument("--budget", type=int, default=55)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_experiment(instances=args.instances, customers=args.customers,
                                    budget=args.budget, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
