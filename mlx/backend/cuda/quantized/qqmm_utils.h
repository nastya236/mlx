// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

// Repack scales from linear to tiled layout for tensor cores
void repack_scales(
    const array& scales,
    array& scales_tiled,
    int M,
    int K,
    int group_size,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
