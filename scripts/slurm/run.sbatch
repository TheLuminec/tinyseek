#!/bin/bash
# Run any tinyseek module as a single-GPU SLURM job (e.g. on OSC).
#
#   sbatch -A <PROJECT_ACCOUNT> scripts/slurm/run.sbatch tinyseek.train.train --config configs/synth_only.yaml prompt=cup seed=1
#   sbatch -A <PROJECT_ACCOUNT> --time=06:00:00 scripts/slurm/run.sbatch tinyseek.generate.sample prompt=cup num_images=2000
#
# Environment: set TINYSEEK_VENV to the virtualenv (default ~/.venvs/tinyseek) and
# TINYSEEK_DATA to a scratch directory holding data/ if it is not in the repo.
#SBATCH --job-name=tinyseek
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

VENV="${TINYSEEK_VENV:-$HOME/.venvs/tinyseek}"
source "$VENV/bin/activate"
if [[ -n "${TINYSEEK_DATA:-}" && ! -e data ]]; then
    ln -s "$TINYSEEK_DATA" data
fi

echo "job $SLURM_JOB_ID on $(hostname), git $(git rev-parse --short HEAD), $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module="$1"; shift
python -u -m "$module" "$@"
