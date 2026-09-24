#!/usr/bin/env bash
#SBATCH --job-name=mlip-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=mlip-scan-%j.out
set -euo pipefail

# TEMPLATE: the instructor must verify memory/time and add the site's required
# account/partition options and module/conda/container activation before class.
# Submit from this ConformerSampling folder with an AIMNet2 Python environment:
# sbatch submit_scan.slurm results/RUN/phi_1D_scan_structures.xyz results/RUN/Relaxed_2D_scan.xyz
# An activated environment may be inherited, but verify this on the target cluster.
if [[ $# -ne 2 ]]; then
    echo 'Usage: sbatch submit_scan.slurm INPUT_XYZ OUTPUT_XYZ' >&2
    exit 2
fi
python -c 'import sys, aimnet, ase; print(sys.executable)'
python -u parallel_relaxed_scan.py --input "$1" --output "$2" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" --step 30 --max-steps 150
