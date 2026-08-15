"""Turn saved candidate-verifier probabilities into an end-to-end risk-coverage curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_candidate_verifier import safe_gate_summary


def sweep(rows: list[dict], points: int = 101) -> list[dict]:
    if points < 2:
        raise ValueError("points must be at least two")
    curve = []
    for index in range(points):
        threshold = index / (points - 1)
        gated = [{**row, "candidate_accepted": row["candidate_probability"] >= threshold} for row in rows]
        curve.append({"threshold": threshold, **safe_gate_summary(gated)})
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep a saved joint verifier's end-to-end risk and coverage.")
    parser.add_argument("--input", type=Path, required=True, help="Scored JSON from evaluate_candidate_verifier.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=101)
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    curve = sweep(report["examples"], args.points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"input": str(args.input), "curve": curve}, indent=2))
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
