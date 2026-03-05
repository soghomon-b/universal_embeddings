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
BATCH="ffmain"
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
run_one ff1       56     500      20      500    501   50  128  2000
run_one ff2       23    1000      20      1000   1001   50  128  2000
run_one ff3       43    2000      20      400    2000   50  128  2000
run_one ff4       76    5000      20      5000   5001   50  128  2000
run_one ff5       85   10000      20      10000  10001   50  128  2000
run_one ff6       16   20000      20      20000   20001   50  128  2000
run_one ff7       98   40000      20      40000  40001   50  128  2000
run_one ff8       16   60000      24      60000  60001   50  128  2000
run_one ff9        8   80000      32      80000  80001   50  128  2000
run_one ff10      75  100000      40      100000  100001   50  128  2000
run_one ff11      91  150000      60      150000  150001   50  128  2000
echo "Finished batch #${BATCH}"
