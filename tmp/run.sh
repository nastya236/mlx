#!/bin/bash
set -e

export NCCL_HOST_IP="127.0.0.1"
export NCCL_PORT=12345
export MLX_WORLD_SIZE=8
export NCCL_DEBUG=WARN

LOG_DIR="logs"
mkdir -p $LOG_DIR

for ((r=0; r<MLX_WORLD_SIZE; r++)); do
  (
    export MLX_RANK=$r
    # echo "=== Rank $r starting, logging to $LOG_DIR/rank_${r}.log ==="
    # Redirect both stdout and stderr to a log file for each rank
    ./test 
    # echo "=== Rank $r done ==="
  ) &
done

wait
echo "All ranks complete."

# Check for errors after the fact
echo "Checking for errors in logs..."
grep -i "error\|fault\|fail" $LOG_DIR/*.log || echo "No errors found."
