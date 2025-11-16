// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace mlx::core {
namespace cu {

template <int bits>
struct Quantize {
  __device__ uint8_t operator()(float x) {
    if constexpr (bits == 8) {
      return __nv_fp8_e4m3(x).__x;
    } else {
      return __nv_fp4_e2m1(x).__x;
    }
  }
};

template <int bits>
struct Dequantize {
  __device__ float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(__nv_fp8_e4m3*)(&x));
    } else {
      return float(*(__nv_fp4_e2m1*)(&x));
    }
  }
};

namespace cg = cooperative_groups;

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void fp_quantize(
    const T* w,
    uint8_t* out,
    uint8_t* scales,
    size_t size,
    bool pack_for_tensor_cores = false) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;
  size_t index = tidx + grid_dim_x * size_t(tidy);
  if (index >= size) {
    return;
  }

  float w_thread = w[index];

  cg::greater<float> max_op;
  auto warp = cg::tiled_partition<group_size>(cg::this_thread_block());

  float scale = cg::reduce(warp, abs(w_thread), max_op);
  scale /= bits == 4 ? 6.0f : 448.0f;
  // Convert to mx scale or nv scale
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto s = ScaleType(scale);
  uint8_t q_scale = s.__x;
  scale = float(s);

  // Write out the scales
  size_t gindex = index / group_size;

  if (pack_for_tensor_cores) {
    // For tensor cores, we need to repack the scales into a tiled layout
    size_t row = index size_t col =
        gindex % (size / group_size / (group_size * 4));
    size_t tiled_offset = scale_tiled_offset(row, col, size / group_size);
    if (index % group_size == 0) {
      scales[tiled_offset] = q_scale;
    }
  } else if (index % group_size == 0) {
    scales[gindex] = q_scale;
  }

  uint8_t output = Quantize<bits>{}(scale == 0 ? 0.0f : w_thread / scale);
  if (bits == 4) {
    uint8_t sval = warp.shfl_down(output, 1);
    output |= sval << bits;
  }
  constexpr int pack_factor = bits == 8 ? 1 : 2;
  if (index % pack_factor == 0) {
    out[index / pack_factor] = output;
  }
}

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void
fp_dequantize(const uint8_t* w, const uint8_t* scales, T* out, size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;

  constexpr int pack_factor = bits == 8 ? 1 : 2;
  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scale = float(((ScaleType*)(scales))[gindex]);

  out += oindex;

  uint val = w[offset];
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    uint8_t d;
    if (bits == 4) {
      d = (val >> (bits * i)) & 0x0f;
    } else if (bits == 8) {
      d = val;
    }
    out[i] = static_cast<T>(scale * Dequantize<bits>{}(d));
  }
}

} // namespace cu

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "fp_quantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_quantize<T, 32, 4, true>;
      if (bits == 8) {
        kernel = cu::fp_quantize<T, 32, 8, true>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize<T, 16, 4, false>;
      }
      bool large = w.size() > UINT_MAX;
      auto [num_blocks, block_dims] =
          get_launch_args(w.size(), w.shape(), w.strides(), large);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<T>(w),
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  constexpr int uint8_per_uint32 = 4;
  int packs_per_int = 8 / bits;

  size_t size = w.size() / packs_per_int;
  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;

  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  dispatch_float_types(w.dtype(), "fp_dequantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_dequantize<T, 32, 4, true>;
      if (bits == 8) {
        kernel = cu::fp_dequantize<T, 32, 8, true>;
      } else if (group_size == 16) {
        kernel = cu::fp_dequantize<T, 16, 4, false>;
      }
      auto [num_blocks, block_dims] =
          get_launch_args(size, grid_shape, w.strides(), large);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          gpu_ptr<T>(w),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not dequantize to output with type float64.");
    }
  });
}

// To pass scales to tensor cores, they need to be repacked into a tiled layout
// https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
// Tiled layout for scale factors is very well described in CUTLASS
// documentation:
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
// Conceptually, it should be like this:
// q_w = mx.ones(shape=(M, N))
// s.shape = (M, N // 16) -- packed in row contigous order
// cbg_cnt = N // 16 // 4
// rb_cnt = M // 128
// tmp = x.reshape(rb_cnt, 4, 32, cbg_cnt, 4)
// repacked_scales = tmp.transpose(0, 3, 2, 1, 4)
// example: indecis of intial tile 128 x 4 of scales (packed in row major tensor
// (M, K // 16), where M = 128, K = 64): array([[0, 1, 2, 3],
//       [4, 5, 6, 7],
//       [8, 9, 10, 11],
//       ...,
//       [500, 501, 502, 503],
//       [504, 505, 506, 507],
//       [508, 509, 510, 511]]
// packed scales within tile 128 x 4:
// array([[[[[0, 1, 2, 3], <-- s_0,0..s_0,3 scales
//          [128, 129, 130, 131], <-- s_32,0..s_32,3 scales
//          [256, 257, 258, 259], <-- s_64,0..s_64,3 scales
//          [384, 385, 386, 387]], <-- s_96,0..s_96,3 scales
//         [[4, 5, 6, 7], <-- s_1,0..s_1,3 scales
//          [132, 133, 134, 135], ...
//          [260, 261, 262, 263],
//          [388, 389, 390, 391]],
//         [[124, 125, 126, 127],
//          [252, 253, 254, 255],
//          [380, 381, 382, 383],
//          [508, 509, 510, 511]]]]],
// Compute the tiled layout offset for scale factors used in tensor cores
// This function maps from a linear scale index to the tiled layout expected
// by cuBLAS tensor cores.
//
// Input: linear scale index (e.g., for a matrix M x K with group_size,
//        scale_index ranges from 0 to (M * K/group_size - 1))
//
// The tiled layout organizes scales into tiles of 128 rows x 4 columns,
// where each tile is subdivided into 4 sub-blocks of 32 rows x 4 columns.
//
// Layout hierarchy:
// - Tile: 128 rows x 4 cols (512 scales)
// - Sub-block: 32 rows x 4 cols (128 scales)
// - Tile contains 4 sub-blocks (stacked vertically)
__device__ size_t
scale_tiled_offset(size_t scale_index, size_t num_rows, size_t num_scale_cols) {
  // Convert linear scale index to 2D coordinates
  size_t row = scale_index / num_scale_cols;
  size_t col = scale_index % num_scale_cols;

  constexpr size_t rows_per_tile = 128;
  constexpr size_t rows_per_sub_block = 32;
  constexpr size_t cols_per_sub_block = 4;
  constexpr size_t sub_blocks_per_tile = 4; // Vertically stacked

  // Decompose row position
  size_t tile_row = row / rows_per_tile; // Which tile row
  size_t row_in_tile = row % rows_per_tile; // Row within tile
  size_t sub_block_row =
      row_in_tile / rows_per_sub_block; // Sub-block within tile (0-3)
  size_t row_in_sub_block =
      row_in_tile % rows_per_sub_block; // Row in sub-block (0-31)

  // Decompose column position
  size_t col_tile = col / cols_per_sub_block; // Which column tile
  size_t col_in_sub_block =
      col % cols_per_sub_block; // Column within sub-block (0-3)

  // Compute tile index and offset within tile
  size_t num_col_tiles = cuda::ceil_div(num_scale_cols, cols_per_sub_block);
  size_t tile_idx = tile_row * num_col_tiles + col_tile;

  // Offset within tile
  // Memory layout after swizzling:
  // [tile][row_in_sub_block][sub_block_row][col_in_sub_block] With dimensions:
  // [tile][32][4][4], giving strides: [512][16][4][1]
  size_t offset_in_tile =
      (row_in_sub_block * sub_blocks_per_tile * cols_per_sub_block) +
      (sub_block_row * cols_per_sub_block) + col_in_sub_block;

  constexpr size_t tile_size = rows_per_tile * cols_per_sub_block; // 512
  return tile_idx * tile_size + offset_in_tile;
}

namespace cu {

__global__ void repack_scales(
    const uint8_t* scales,
    uint8_t* scales_tiled,
    size_t num_scales,
    size_t M,
    size_t num_scale_cols) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;
  size_t scale_index = tidx + grid_dim_x * size_t(tidy);

  if (scale_index >= num_scales) {
    return;
  }

  // Compute tiled offset for this scale
  size_t tiled_offset = scale_tiled_offset(scale_index, M, num_scale_cols);

  // Copy scale from linear to tiled layout
  scales_tiled[tiled_offset] = scales[scale_index];
}

} // namespace cu

void repack_scales(
    const array& scales,
    array& scales_tiled,
    int M,
    int K,
    int group_size,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(scales);
  enc.set_output_array(scales_tiled);

  size_t num_scales = scales.size();
  bool large = num_scales > UINT_MAX;
  auto [num_blocks, block_dims] =
      get_launch_args(num_scales, scales.shape(), scales.strides(), large);

  enc.add_kernel_node(
      cu::repack_scales,
      num_blocks,
      block_dims,
      0,
      gpu_ptr<uint8_t>(scales),
      gpu_ptr<uint8_t>(scales_tiled),
      num_scales,
      static_cast<size_t>(M),
      static_cast<size_t>(K / group_size));
}

} // namespace mlx::core
