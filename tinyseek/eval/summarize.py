"""Results table over training runs: mean and spread across seeds per condition and test set.

    python -m tinyseek.eval.summarize [--prompts cup backpack] [--out runs/summary.json]

Reads runs/<prompt>/<condition>/seed*/eval.json and prints, per prompt and test set, the
mean AP@0.5 (and its large-object part) and ECE with the seed-to-seed standard deviation
and the number of seeds. Only runs that exist are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("ap50", "ap50_large", "ece")


def collect(run_root: Path, prompts: list[str], conditions: list[str]) -> dict:
    table: dict = {}
    for prompt in prompts:
        for condition in conditions:
            evals = [json.loads(p.read_text()) for p in sorted((run_root / prompt / condition).glob("seed*/eval.json"))]
            if not evals:
                continue
            sets = sorted(set.intersection(*(set(e) for e in evals)))
            for test_set in sets:
                row = {"n_seeds": len(evals)}
                for m in METRICS:
                    vals = np.array([e[test_set][m] for e in evals], dtype=float)
                    row[m] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std(ddof=1)), 4)
                              if len(vals) > 1 else None, "values": [round(float(v), 4) for v in vals]}
                table.setdefault(prompt, {}).setdefault(test_set, {})[condition] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--prompts", nargs="+", default=["cup", "backpack"])
    parser.add_argument("--conditions", nargs="+", default=["real_only", "teacher_real", "synth_only", "synth_real"])
    parser.add_argument("--out", type=Path, default=Path("runs/summary.json"))
    args = parser.parse_args()

    table = collect(args.run_root, args.prompts, args.conditions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1))

    def fmt(cell: dict) -> str:
        return f"{cell['mean']:.3f}" + (f" ±{cell['std']:.3f}" if cell["std"] is not None else "")

    for prompt, sets in table.items():
        for test_set, rows in sets.items():
            print(f"\n{prompt} · {test_set}")
            print(f"  {'condition':14s} {'seeds':>5s}  {'AP@0.5':>14s}  {'AP@0.5 large':>14s}  {'ECE':>14s}")
            for condition, row in rows.items():
                print(f"  {condition:14s} {row['n_seeds']:5d}  {fmt(row['ap50']):>14s}  "
                      f"{fmt(row['ap50_large']):>14s}  {fmt(row['ece']):>14s}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
