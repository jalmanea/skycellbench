#!/bin/bash
# One job of the held-out run on Ibex (KAUST). Submit from the repository root:
#   sbatch --export=ALL,SET=heldout,JOB=answers experiments/ibex/run.sbatch
# JOB=answers     every controller's answers, both scorers (experiments/heldout.py)
# JOB=reference   the 200,000-call reference search (experiments/headroom.py)
# JOB=check       ns-3 on this machine against the stored scores of results/compare-$SET.jsonl
# JOB=tests       the test suite
# Expects the environment at $ENV (default /ibex/user/$USER/skycellbench-env) and the ns-3
# scorer built with ns3/build.sh. Slurm logs go to /ibex/user/$USER/skycellbench-logs.
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J skycellbench
#SBATCH -o /ibex/user/%u/skycellbench-logs/%x-%j.out
set -euo pipefail
module load gcc/12.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=${ENV:-/ibex/user/$USER/skycellbench-env}/bin/python
N=${SLURM_CPUS_PER_TASK:-32}
echo "job $JOB set ${SET:-} on $(hostname), commit $(git rev-parse HEAD), $N cpus"
case $JOB in
    answers)   $PY experiments/heldout.py "$SET" "$N" && $PY experiments/analyse_heldout.py "$SET" ;;
    reference) WORKERS=$N $PY experiments/headroom.py "$SET" 200000 ;;
    check)     $PY experiments/ns3_machine_check.py "$SET" "$N" ;;
    tests)     $PY -m pytest -q tests ;;
    *)         echo "unknown JOB=$JOB"; exit 1 ;;
esac
