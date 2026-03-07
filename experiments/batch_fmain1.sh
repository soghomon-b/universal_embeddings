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
BATCH="fmain"
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
run_one lf11       56     500      10      50    51   50  128  2000
run_one lf12       23    1000      10      100   101   50  128  2000
run_one lf13       43    2000      10      200    201   50  128  2000
run_one lf14       76    5000      10      500   501   50  128  2000
run_one lf15       85   10000      10      1000  1001   50  128  2000
run_one lf16       16   20000      10      2000  2001   50  128  2000
run_one lf17       98   40000      10      4000  4001   50  128  2000
run_one lf18       16   60000      30      2000  2001   50  128  2000
run_one lf19        8   80000      30      2666  2667   50  128  2000
run_one lf110      75  100000      30      3333  3334   50  128  2000
run_one lf111      91  150000      30      5000  5000   50  128  2000
echo "Finished batch #${BATCH}"
