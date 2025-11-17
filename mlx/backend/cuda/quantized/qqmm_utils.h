// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

// Compute padded dimensions for tiled layout
// Tiles are 128 rows × 4 columns, must allocate full tiles
inline std::pair<size_t, size_t>
get_padded_scale_dims(int M, int K, int group_size) {
  constexpr size_t rows_per_tile = 128;
  constexpr size_t cols_per_tile = 4;

  size_t num_scale_cols = K / group_size;
  size_t padded_rows =
      ((M + rows_per_tile - 1) / rows_per_tile) * rows_per_tile;
  size_t padded_cols =
      ((num_scale_cols + cols_per_tile - 1) / cols_per_tile) * cols_per_tile;

  return {padded_rows, padded_cols};
}

void repack_scales(
    const array& scales,
    array& scales_tiled,
    int M,
    int K,
    int group_size,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
