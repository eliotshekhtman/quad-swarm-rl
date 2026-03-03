#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Dict


def _load_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(metric_block: Dict[str, float], level: str) -> float:
    if level == "max":
        return float(metric_block["max"])
    if level == "95":
        return float(metric_block["top95_max"])
    if level == "90":
        return float(metric_block["top90_max"])
    raise ValueError(f"Unknown level: {level}")


def _compute(values: Dict[str, float], T: int) -> Dict[str, float]:
    L_x = values["L_x"]
    L_u = values["L_u"]
    L_pi = values["L_pi"]
    L_eu = values["L_eu"]
    L_ex = values["L_ex"]
    L_U = values["L_U"]

    Lambda_x = L_x + L_u * L_pi
    geom_sum = sum((Lambda_x ** t) for t in range(max(T - 1, 0)))
    beta_T = L_eu + L_ex * L_u * geom_sum
    kappa = beta_T * L_U
    return {
        "Lambda_x": Lambda_x,
        "beta_T": beta_T,
        "kappa": kappa,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Lambda_x, beta_T, and kappa from lipschitz_metrics.json."
    )
    parser.add_argument("metrics_json", type=str, help="Path to lipschitz_metrics.json")
    parser.add_argument("T", type=int, help="Integer horizon T")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.T < 0:
        raise ValueError("T must be >= 0")

    metrics = _load_metrics(args.metrics_json)

    required = ("L_x", "L_u", "L_pi", "L_eu", "L_ex", "L_U")
    missing = [k for k in required if k not in metrics]
    if missing:
        raise KeyError(f"Missing keys in metrics JSON: {missing}")

    levels = [("max", "max"), ("95", "95th percentile"), ("90", "90th percentile")]
    print(f"T = {args.T}")
    for level_key, label in levels:
        print(f"\n[{label}]")
        try:
            vals = {
                "L_x": _pick(metrics["L_x"], level_key),
                "L_u": _pick(metrics["L_u"], level_key),
                "L_pi": _pick(metrics["L_pi"], level_key),
                "L_eu": _pick(metrics["L_eu"], level_key),
                "L_ex": _pick(metrics["L_ex"], level_key),
                "L_U": _pick(metrics["L_U"], level_key),
            }
            out = _compute(vals, args.T)
            print(f"Lambda_x = {out['Lambda_x']:.10g}")
            print(f"beta_T   = {out['beta_T']:.10g}")
            print(f"kappa    = {out['kappa']:.10g}")
        except OverflowError as exc:
            print(f"OverflowError while computing {label}: {exc}")


if __name__ == "__main__":
    main()
