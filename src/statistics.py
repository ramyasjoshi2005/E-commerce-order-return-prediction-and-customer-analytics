"""One primary hypothesis test: two-proportion z-test on subsequent returns."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import ARTIFACTS, FEATURES_PATH


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> dict:
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    zcrit = stats.norm.ppf(0.975)
    ci = (diff - zcrit * se_diff, diff + zcrit * se_diff)
    risk_ratio = p1 / p2 if p2 > 0 else float("inf")

    return {
        "z_statistic": float(z),
        "p_value": float(p_value),
        "diff_proportions": float(diff),
        "ci_95": [float(ci[0]), float(ci[1])],
        "risk_ratio": float(risk_ratio),
        "p1": float(p1),
        "p2": float(p2),
        "n1": int(n1),
        "n2": int(n2),
        "x1": int(x1),
        "x2": int(x2),
    }


def run_previous_return_test(df: pd.DataFrame) -> dict:
    """Group A: at least one previous return; Group B: none before this order."""
    a = df[df["previous_return_count"] >= 1]
    b = df[df["previous_return_count"] == 0]
    result = two_proportion_ztest(
        int(a["returned"].sum()),
        int(len(a)),
        int(b["returned"].sum()),
        int(len(b)),
    )
    alpha = 0.05
    significant = result["p_value"] < alpha
    if significant:
        interp = (
            f"At alpha = 0.05 we reject the null hypothesis (p = {result['p_value']:.4g}). "
            f"Orders placed after at least one previous return had an observed return rate of "
            f"{result['p1']:.1%}, versus {result['p2']:.1%} for orders with no previous returns "
            f"(difference {result['diff_proportions']:.1%} points; risk ratio {result['risk_ratio']:.2f}). "
            "This is an association in observational data, not evidence that previous returns cause future returns."
        )
    else:
        interp = (
            f"At alpha = 0.05 we do not reject the null hypothesis (p = {result['p_value']:.4g}). "
            "The observed difference in subsequent return rates is compatible with chance variation "
            "under equal population proportions. This does not prove the rates are equal, and it does not "
            "establish causation."
        )

    return {
        "research_question": (
            "Do customers with previous returns have a different subsequent return rate "
            "compared with customers without previous returns?"
        ),
        "null_hypothesis": "H0: p_A = p_B (equal subsequent return probabilities)",
        "alternative_hypothesis": "H1: p_A ≠ p_B (two-sided)",
        "test_name": "Two-proportion z-test (two-sided)",
        "group_a_definition": "Orders with at least one previous return by the same customer (point-in-time)",
        "group_b_definition": "Orders with zero previous returns by the same customer (includes first orders)",
        "outcome": "Whether the current order was returned",
        "alpha": alpha,
        **result,
        "significant_at_0.05": bool(significant),
        "interpretation": interp,
        "causation_disclaimer": (
            "Statistical significance describes evidence against equal proportions. "
            "It does not imply that prior returns cause later returns."
        ),
    }


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    payload = run_previous_return_test(df)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "hypothesis_test.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(payload["test_name"])
    print(f"z = {payload['z_statistic']:.4f}, p = {payload['p_value']:.6g}")
    print(f"p_A = {payload['p1']:.4f} (n={payload['n1']}), p_B = {payload['p2']:.4f} (n={payload['n2']})")
    print(f"95% CI for difference: {payload['ci_95']}")
    print(f"Risk ratio: {payload['risk_ratio']:.4f}")
    print(payload["interpretation"])


if __name__ == "__main__":
    main()
