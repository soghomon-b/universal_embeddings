#!/usr/bin/env bash
set -euo pipefail

# Workaround for Intel OpenMP duplicate runtime issue on some conda/pip mixes
export KMP_DUPLICATE_LIB_OK=TRUE

# Keep threading sane (optional but recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run from repo root (universal_embeddings)
# Usage:
#   bash run_experiments.sh
#
# Assumes you added argparse main() to experiments/example_exp.py
# and your venv is already activated (or edit below to source it).
BATCH="ffmain2"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="experiments.exp_runner"

run_one () {
  local exp="$1" seed="$2" data_size="$3" epochs="$4" n_min="$5" n_max="$6" K="$7" r="$8" n_sent_ret="$9"

  echo "============================================================"
  echo "Experiment ${exp} | seed=${seed} data_size=${data_size} epochs=${epochs} n_min=${n_min} n_max=${n_max} K=${K} r=${r} n_sent_ret=${n_sent_ret}"
  echo "============================================================"

  "${PYTHON_BIN}" -m "${SCRIPT}" \
    --exp "${exp}" \
    --seed "${seed}" \
    --data_size "${data_size}" \
    --epochs "${epochs}" \
    --n_min "${n_min}" \
    --n_max "${n_max}" \
    --K "${K}" \
    --r "${r}" \
    --n_sent_ret "${n_sent_ret}"
}

# Experiment #  Seed  Data Size  Epochs  n_min  n_max   K   r    # sentence ret
run_one ff21       74     500      20      50    51   50  128  2000
run_one ff22       87    1000      20      100   101   50  128  2000
run_one ff23       32    2000      20      200    201   50  128  2000
run_one ff24       14    5000      20      500   501   50  128  2000
run_one ff25       73   10000      20      1000  1001   50  128  2000
run_one ff26       85   20000      20      2000  2001   50  128  2000
run_one ff27       97   40000      20      4000  4001   50  128  2000
run_one ff28       12   60000      60      2000  2001   50  128  2000
run_one ff29       36   80000      60      2666  2667   50  128  2000
run_one ff210      68  100000      60      3333  3334   50  128  2000
run_one ff211      01  150000      60      5000  5001   50  128  2000
echo "Finished batch #${BATCH}"
