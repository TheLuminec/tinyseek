"""Run the training-condition sweep, one avm-tracked job per run, then summarize.

    python -m tinyseek.train.sweep --conditions real_only teacher_real synth_only synth_real \
        --prompts cup backpack --seeds 0 1 2 [--dry-run] [extra overrides, e.g. train.epochs=60]

Runs whose eval.json already exists are skipped, so an interrupted sweep resumes. Each run
goes through ``avm run ... --gpu --exclusive --wait`` (when avm is installed) so it shows
on the dashboard and queues behind other heavy GPU jobs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from tinyseek.config import load_config
from tinyseek.train.train import run_dir_for


def run_one(condition: str, prompt: str, seed: int, overrides: list[str], dry_run: bool) -> int:
    args = [f"prompt={prompt}", f"seed={seed}", *overrides]
    cfg = load_config(f"configs/{condition}.yaml", args)
    run_dir = run_dir_for(cfg)
    if (run_dir / "eval.json").exists():
        print(f"skip {run_dir} (done)")
        return 0
    cmd = [sys.executable, "-u", "-m", "tinyseek.train.train", "--config", f"configs/{condition}.yaml", *args]
    if shutil.which("avm"):
        name = run_dir.relative_to(cfg["run_root"]).as_posix().replace("/", " ")
        cmd = ["avm", "run", f"tinyseek train {name}", "--gpu", "--exclusive", "--wait",
               "--", *cmd]
    print(" ".join(cmd), flush=True)
    return 0 if dry_run else subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", default=["real_only", "teacher_real", "synth_only", "synth_real"])
    parser.add_argument("--prompts", nargs="+", default=["cup", "backpack"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    failed = []
    for seed in args.seeds:  # seed-major: a partial sweep still covers every condition
        for prompt in args.prompts:
            for condition in args.conditions:
                if run_one(condition, prompt, seed, args.overrides, args.dry_run) != 0:
                    failed.append(f"{prompt}/{condition}/seed{seed}")
    if failed:
        print("failed runs: " + ", ".join(failed))
    if not args.dry_run:
        subprocess.run([sys.executable, "-m", "tinyseek.eval.summarize", "--prompts", *args.prompts,
                        "--conditions", *args.conditions], check=False)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
