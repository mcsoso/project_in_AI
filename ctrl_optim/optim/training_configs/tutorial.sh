#!/bin/bash

set -euo pipefail

PY="${PYTHON_CMD:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "Python not found: $PY" >&2; exit 127; }

exec "$PY" -m ctrl_optim.optim.train \
    --musc_model 22 \
    --model humotech \
    --sim_time 20 \
    --pose_key walk_left \
    --num_strides 5 \
    --delayed 0 \
    --optim_mode single \
    --reflex_mode uni \
    --tgt_vel 1.25 \
    --tgt_slope 0 \
    --trunk_err_type ref_diff \
    --tgt_sym_th 0.1 \
    --tgt_grf_th 1.5 \
    -kine \
    --ExoOn 1 \
    --n_points 6 \
    --max_torque 10.0 \
    --popsize 32 \
    --maxiter 1000 \
    --threads 32 \
    --sigma_gain 10 \
    --save_path humotech