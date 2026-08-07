// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

template <typename F>
void dispatch_vec(int v, F&& f) {
  switch (v) {
    case 8:
      f(std::integral_constant<int, 8>{});
      return;
    case 4:
      f(std::integral_constant<int, 4>{});
      return;
    default:
      f(std::integral_constant<int, 2>{});
      return;
  }
}

template <typename T, bool traditional, bool forward>
__device__ void rope_single_impl(
    const T* in,
    T* out,
    int32_t offset,
    float inv_freq,
    float scale,
    int64_t stride,
    uint2 pos,
    uint2 dims) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = cos(theta);
  float sintheta = sin(theta);

  // Compute the input and output indices
  uint32_t index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + dims.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_single(
    const T* in,
    T* out,
    const int32_t* offset,
    float scale,
    float base,
    int64_t stride,
    uint2 dims) {
  uint2 pos = make_uint2(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y);
  if (pos.x >= dims.x || pos.y >= dims.y) {
    return;
  }

  float d = static_cast<float>(pos.x) / static_cast<float>(dims.x);
  float inv_freq = exp2(-d * base);
  rope_single_impl<T, traditional, forward>(
      in, out, *offset, inv_freq, scale, stride, pos, dims);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_single_freqs(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    int64_t stride,
    uint2 dims,
    int64_t freq_stride) {
  uint2 pos = make_uint2(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y);
  if (pos.x >= dims.x || pos.y >= dims.y) {
    return;
  }

  float inv_freq = 1.0 / freqs[freq_stride * pos.x];
  rope_single_impl<T, traditional, forward>(
      in, out, *offset, inv_freq, scale, stride, pos, dims);
}

template <typename T, bool traditional, bool forward, int N = 4>
__device__ void rope_impl(
    const T* in,
    T* out,
    const int* offset,
    float inv_freq,
    float scale,
    const cuda::std::array<int64_t, 4> strides,
    const cuda::std::array<int64_t, 4> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 pos,
    uint3 dims) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = cos(theta);
  float sintheta = sin(theta);

  size_t in_batch_head = batch_idx * strides[0] + head_idx * strides[1];
  size_t out_batch_head =
      batch_idx * out_strides[0] + head_idx * out_strides[1];

  // Compute the input and output indices
  size_t in_index_1, in_index_2;
  size_t out_index_1, out_index_2;
  if (traditional) {
    out_index_1 =
        2 * pos.x * out_strides[3] + pos.y * out_strides[2] + out_batch_head;
    out_index_2 = out_index_1 + out_strides[3];
    in_index_1 = 2 * pos.x * strides[3] + pos.y * strides[2] + in_batch_head;
    in_index_2 = in_index_1 + strides[3];
  } else {
    out_index_1 =
        pos.x * out_strides[3] + pos.y * out_strides[2] + out_batch_head;
    out_index_2 = out_index_1 + dims.x * out_strides[3];
    in_index_1 = pos.x * strides[3] + pos.y * strides[2] + in_batch_head;
    in_index_2 = in_index_1 + dims.x * strides[3];
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += strides[1];
    in_index_2 += strides[1];
    out_index_1 += out_strides[1];
    out_index_2 += out_strides[1];
  }
}

template <typename T, bool traditional, bool forward>
__global__ void rope(
    const T* in,
    T* out,
    const int32_t* offset,
    float scale,
    float base,
    const __grid_constant__ cuda::std::array<int64_t, 4> strides,
    const __grid_constant__ cuda::std::array<int64_t, 4> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 dims) {
  uint3 pos = make_uint3(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z);
  if (pos.x >= dims.x || pos.y >= dims.y || pos.z >= dims.z) {
    return;
  }

  float d = static_cast<float>(pos.x) / static_cast<float>(dims.x);
  float inv_freq = exp2(-d * base);
  rope_impl<T, traditional, forward>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      dims);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_freqs(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    float base,
    const __grid_constant__ cuda::std::array<int64_t, 4> strides,
    const __grid_constant__ cuda::std::array<int64_t, 4> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 dims,
    int64_t freq_stride) {
  uint3 pos = make_uint3(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z);
  if (pos.x >= dims.x || pos.y >= dims.y || pos.z >= dims.z) {
    return;
  }

  float inv_freq = 1.0 / freqs[freq_stride * pos.x];
  rope_impl<T, traditional, forward>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      dims);
}

template <typename T, bool forward, int VEC>
__global__ void rope_vec(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    float log2_base,
    const __grid_constant__ cuda::std::array<int64_t, 4> strides,
    const __grid_constant__ cuda::std::array<int64_t, 4> out_strides,
    int64_t offset_stride,
    int64_t freq_stride,
    int n_head,
    int n_seq,
    int half,
    int heads_per_block) {
  extern __shared__ float smem[];
  float* s_cos = smem;
  float* s_sin = smem + half;

  const int n_chunk = half / VEC;
  const int chunk = threadIdx.x % n_chunk;
  const int head_in_block = threadIdx.x / n_chunk;

  const int groups = (n_head + heads_per_block - 1) / heads_per_block;
  const int batch_idx = blockIdx.x / groups;
  const int head_idx = (blockIdx.x % groups) * heads_per_block + head_in_block;
  const int32_t batch_offset = offset[batch_idx * offset_stride];

  const int64_t in_bh = batch_idx * strides[0] + head_idx * strides[1];
  const int64_t out_bh = batch_idx * out_strides[0] + head_idx * out_strides[1];

  // Grid-stride over the sequence so gridDim.y stays within its 65535 limit.
  for (int s = blockIdx.y; s < n_seq; s += gridDim.y) {
    __syncthreads();
    for (int d = threadIdx.x; d < half; d += blockDim.x) {
      float inv_freq = freqs != nullptr
          ? 1.0f / freqs[freq_stride * d]
          : exp2f(
                -(static_cast<float>(d) / static_cast<float>(half)) *
                log2_base);
      float theta = scale * static_cast<float>(s + batch_offset) * inv_freq;
      s_cos[d] = cosf(theta);
      s_sin[d] = sinf(theta);
    }
    __syncthreads();

    // Trailing heads in the last group have no work, but must still reach the
    // __syncthreads() above on every iteration.
    if (head_idx >= n_head) {
      continue;
    }

    const int64_t i1 =
        in_bh + static_cast<int64_t>(s) * strides[2] + chunk * VEC;
    const int64_t o1 =
        out_bh + static_cast<int64_t>(s) * out_strides[2] + chunk * VEC;

    auto x1 = unsafe_load_vector<VEC>(in + i1, 0);
    auto x2 = unsafe_load_vector<VEC>(in + i1 + half, 0);
    AlignedVector<T, VEC> r1;
    AlignedVector<T, VEC> r2;

#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      float costheta = s_cos[chunk * VEC + i];
      float sintheta = s_sin[chunk * VEC + i];
      float f1 = static_cast<float>(x1[i]);
      float f2 = static_cast<float>(x2[i]);
      if (forward) {
        r1[i] = static_cast<T>(f1 * costheta - f2 * sintheta);
        r2[i] = static_cast<T>(f1 * sintheta + f2 * costheta);
      } else {
        r1[i] = static_cast<T>(f2 * sintheta + f1 * costheta);
        r2[i] = static_cast<T>(f2 * costheta - f1 * sintheta);
      }
    }

    unsafe_store_vector<VEC>(out + o1, 0, r1);
    unsafe_store_vector<VEC>(out + o1 + half, 0, r2);
  }
}

} // namespace cu

namespace fast {

bool RoPE::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RoPE::eval_gpu");

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  auto& in = inputs[0];
  auto& offset = inputs[1];
  auto& out = outputs[0];

  cuda::std::array<int64_t, 4> strides;
  cuda::std::array<int64_t, 4> out_strides;
  bool donated = false;
  int ndim = in.ndim();

  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  size_t mat_size = T * D;

  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }

  // if input has < 4 dims or row_contiguous: if we can donate the input, reuse
  // the buffer if in is not donatable, we allocate a new buffer for the output
  // with the same strides as input if input has > 4 dims and is not
  // row_contigous, we copy the input to a new buffer (we should not be here
  // often) in case it is partial rotation: we reuse if input is donatable and
  // copy if it is not donatable
  bool partial = dims_ < D;
  bool layout_ok = ndim <= 4 || in.flags().row_contiguous;

  if (layout_ok) {
    if (in.is_donatable()) {
      donated = true;
      out.copy_shared_buffer(in);
    } else if (!partial) {
      out.set_data(
          cu::malloc_async(in.data_size() * in.itemsize(), encoder),
          in.data_size(),
          in.strides(),
          in.flags());
    } else {
      donated = true;
      auto ctype =
          in.flags().row_contiguous ? CopyType::Vector : CopyType::General;
      copy_gpu(in, out, ctype, s);
    }
  } else {
    donated = true;
    copy_gpu(in, out, CopyType::General, s);
  }
  const auto& src_strides = donated ? out.strides() : in.strides();
  strides[0] = src_strides[0];
  strides[1] = src_strides[ndim - 3];
  strides[2] = src_strides[ndim - 2];
  strides[3] = src_strides[ndim - 1];

  const auto& dst_strides = out.strides();
  out_strides[0] = dst_strides[0];
  out_strides[1] = dst_strides[ndim - 3];
  out_strides[2] = dst_strides[ndim - 2];
  out_strides[3] = dst_strides[ndim - 1];

  // Some flags to help us dispatch below
  bool single = in.flags().row_contiguous && B == 1 && T == 1;
  bool with_freqs = inputs.size() == 3;

  int half = dims_ / 2;
  auto vec_ok = [&](int v) {
    if (traditional_ || single || half % v != 0) {
      return false;
    }
    if (strides[3] != 1 || out_strides[3] != 1) {
      return false;
    }
    for (int i = 0; i < 3; ++i) {
      if (strides[i] % v != 0 || out_strides[i] % v != 0) {
        return false;
      }
    }
    return true;
  };

  encoder.set_input_array(donated ? out : in);
  encoder.set_input_array(offset);
  if (with_freqs) {
    encoder.set_input_array(inputs[2]);
  }
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "rope", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    constexpr int max_vec = 16 / sizeof(DataType);
    auto* in_ptr = gpu_ptr<DataType>(donated ? out : in);
    auto* out_ptr = gpu_ptr<DataType>(out);
    auto ptrs_aligned = [&](int v) {
      size_t width = v * sizeof(DataType);
      return (reinterpret_cast<uintptr_t>(in_ptr) % width) == 0 &&
          (reinterpret_cast<uintptr_t>(out_ptr) % width) == 0;
    };
    int vec = 0;
    for (int v = max_vec; v >= 2; v /= 2) {
      if (vec_ok(v) && ptrs_aligned(v)) {
        vec = v;
        break;
      }
    }

    if (vec > 0) {
      dispatch_bool(forward_, [&](auto forward) {
        cu::dispatch_vec(vec, [&](auto vec_tag) {
          constexpr int VEC = MLX_GET_VALUE(vec_tag);
          // Widths wider than 16 bytes are never selected for this type.
          if constexpr (VEC * sizeof(DataType) <= 16) {
            auto kernel = cu::rope_vec<DataType, forward.value, VEC>;

            int n_chunk = half / VEC;
            // ~256 threads per block, but never more heads than we have.
            int heads_per_block = std::max(1, std::min(N, 256 / n_chunk));
            int groups = (N + heads_per_block - 1) / heads_per_block;
            dim3 block(n_chunk * heads_per_block, 1, 1);
            dim3 grid(B * groups, std::min<uint32_t>(T, 65535), 1);
            uint32_t smem = 2 * half * sizeof(float);

            int64_t offset_stride =
                inputs[1].ndim() > 0 ? inputs[1].strides()[0] : 0;
            const float* freqs_ptr =
                with_freqs ? gpu_ptr<float>(inputs[2]) : nullptr;
            int64_t fstride = with_freqs ? inputs[2].strides(0) : 0;
            encoder.add_kernel_node_ex(
                kernel,
                grid,
                block,
                dim3{},
                smem,
                in_ptr,
                out_ptr,
                gpu_ptr<int32_t>(offset),
                freqs_ptr,
                scale_,
                std::log2(base_),
                strides,
                out_strides,
                offset_stride,
                fstride,
                N,
                T,
                half,
                heads_per_block);
          }
        });
      });
      return;
    }

    dispatch_bool(traditional_, [&](auto traditional) {
      dispatch_bool(forward_, [&](auto forward) {
        if (single && !with_freqs) {
          auto kernel =
              cu::rope_single<DataType, traditional.value, forward.value>;
          uint2 dims = make_uint2(dims_ / 2, N);
          auto [grid, block] = get_grid_and_block(dims.x, dims.y, 1);
          encoder.add_kernel_node(
              kernel,
              grid,
              block,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              mat_size,
              dims);
        } else if (single) {
          auto kernel =
              cu::rope_single_freqs<DataType, traditional.value, forward.value>;
          uint2 dims = make_uint2(dims_ / 2, N);
          auto [grid, block] = get_grid_and_block(dims.x, dims.y, 1);
          encoder.add_kernel_node(
              kernel,
              grid,
              block,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              mat_size,
              dims,
              inputs[2].strides(0));
        } else if (with_freqs) {
          auto kernel =
              cu::rope_freqs<DataType, traditional.value, forward.value>;
          int n_per_thread = 4;
          uint32_t dimz = B * ((N + n_per_thread - 1) / n_per_thread);
          uint3 dims = make_uint3(dims_ / 2, T, dimz);
          auto [grid, block] = get_grid_and_block(dims.x, dims.y, dims.z);
          int64_t offset_stride = 0;
          if (inputs[1].ndim() > 0) {
            offset_stride = inputs[1].strides()[0];
          }
          encoder.add_kernel_node(
              kernel,
              grid,
              block,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims,
              inputs[2].strides(0));
        } else {
          auto kernel = cu::rope<DataType, traditional.value, forward.value>;
          int n_per_thread = 4;
          uint32_t dimz = B * ((N + n_per_thread - 1) / n_per_thread);
          uint3 dims = make_uint3(dims_ / 2, T, dimz);
          auto [grid, block] = get_grid_and_block(dims.x, dims.y, dims.z);
          int64_t offset_stride = 0;
          if (inputs[1].ndim() > 0) {
            offset_stride = inputs[1].strides()[0];
          }
          encoder.add_kernel_node(
              kernel,
              grid,
              block,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims);
        }
      });
    });
  });
}

} // namespace fast

} // namespace mlx::core
