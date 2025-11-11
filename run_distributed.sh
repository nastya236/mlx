#!/bin/bash

export NCCL_HOST_IP="240.6.70.61"  # rendezvous address (rank 0’s host)
export NCCL_PORT=12345
export NCCL_DEBUG=DEBUG
export MLX_WORLD_SIZE=4
export MLX_LOCAL_WORLD_SIZE=2

for ((local_rank=0; local_rank<MLX_LOCAL_WORLD_SIZE; local_rank++)); do
  (
    export MLX_LOCAL_RANK=$local_rank
    export MLX_RANK=$(( NODE_RANK*MLX_LOCAL_WORLD_SIZE + local_rank ))   # unique global rank
    export CUDA_VISIBLE_DEVICES=$local_rank           # bind to a single GPU

    echo "=== Node $NODE_RANK | Global rank $MLX_RANK | GPU $CUDA_VISIBLE_DEVICES ==="
    build/examples/cpp/all_gather
  ) &
done

wait