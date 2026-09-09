#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/scannet.yaml}
GPUS=${GPUS:-4}

torchrun --standalone --nproc_per_node="${GPUS}" train.py --config "${CONFIG}" --variant vanilla

