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
BATCH="fffmain1"
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
run_one lfff11       56     500      30      50    51   50  128  2000
run_one lfff12       23    1000      30      100   101   50  128  2000
run_one lfff13       43    2000      30      200    201   50  128  2000
run_one lfff14       76    5000      30      500   501   50  128  2000
run_one lfff15       85   10000      30      1000  1001   50  128  2000
run_one lfff16       16   20000      30      2000  2001   50  128  2000
run_one lfff17       98   40000      30      4000  4001   50  128  2000
run_one lfff18       16   60000      90      2000  2001   50  128  2000
run_one lfff19        8   80000      90      2666  2667   50  128  2000
run_one lfff110      75  100000      90      3333  3334   50  128  2000
echo "Finished batch #${BATCH}"
