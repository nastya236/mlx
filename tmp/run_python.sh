#!/usr/bin/env bash
set -euxo pipefail

# Require START_RANK to be provided: e.g., START_RANK=0 ./run.sh
: "${START_RANK:?Set START_RANK, e.g., START_RANK=0 ./run.sh}"

export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=lo
export NCCL_HOST_IP=240.62.162.72
export NCCL_PORT=12345

export MLX_WORLD_SIZE=16   # not used below; keep if your code reads it
END_RANK=$(( START_RANK + 8 ))   # run ranks START_RANK .. END_RANK-1

# launch one process per rank
for RANK in $(seq "$START_RANK" $(( END_RANK - 1 ))); do
  MLX_RANK="$RANK" \
    python3.10 tmp/test_nccl.py &
done
wait
