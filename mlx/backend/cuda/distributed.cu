// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"

#include <cassert>

#include <cstdio>
#include <cuda_runtime_api.h>

enum class SIOBranch : unsigned char { CopyMade=0, Donated=1, FreshAlloc=2 };

static inline const char* branch_name(SIOBranch b) {
  switch (b) {
    case SIOBranch::CopyMade:   return "copy_gpu";
    case SIOBranch::Donated:    return "donated";
    case SIOBranch::FreshAlloc: return "allocated";
    default:                    return "unknown";
  }
}

struct PtrStamp { const void* ptr=nullptr; size_t nbytes=0; };
static PtrStamp g_in_stamp, g_out_stamp;

static inline void log_ptr_if_changed(const char* tag,
                                      const void* cur_ptr,
                                      size_t nbytes) {
  PtrStamp& s = (tag[0]=='i') ? g_in_stamp : g_out_stamp;
  const bool changed = (s.ptr != cur_ptr) || (s.nbytes != nbytes);
  if (changed) {
    std::fprintf(stderr, "[%s] ptr=%p bytes=%zu <-- CHANGED\n",
                 tag, cur_ptr, nbytes);
#if CUDART_VERSION >= 11000
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, cur_ptr) == cudaSuccess) {
      // 0=Host, 1=Device, 2=Managed (for new CUDA); older has memoryType
#if defined(cudaMemoryTypeDevice) // handle older SDKs gracefully
      std::fprintf(stderr, "        type=%d (0=Host,1=Device,2=Managed)\n",
                   int(attr.type));
#else
      std::fprintf(stderr, "        memoryType=%d (1=Host,2=Device)\n",
                   int(attr.memoryType));
#endif
    }
#endif
    s.ptr   = cur_ptr;
    s.nbytes = nbytes;
  }
}

namespace mlx::core::distributed {
void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  // auto set_input_output =
  //     [s = stream()](const array& in, array& out) -> std::pair<array, array>
  //     {
  //   if (!in.flags().row_contiguous) {
  //     copy_gpu(in, out, CopyType::General, s);
  //     return {out, out};
  //   } else if (in.is_donatable()) {
  //     out.copy_shared_buffer(in);
  //     return {in, out};
  //   } else {
  //     out.set_data(allocator::malloc(out.nbytes()));
  //     return {in, out};
  //   }
  // };
  SIOBranch used_branch = SIOBranch::FreshAlloc;

  auto set_input_output =
  [this, &used_branch, s = stream()](const array& in, array& out)
  -> std::pair<array, array> {
    if (!in.flags().row_contiguous) {
      copy_gpu(in, out, CopyType::General, s);
      used_branch = SIOBranch::CopyMade;
      return {out, out};
    } else if (in.is_donatable()) {
      out.copy_shared_buffer(in);
      used_branch = SIOBranch::Donated;
      return {in, out};
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
      used_branch = SIOBranch::FreshAlloc;
      return {in, out};
    }
  };

  auto [input, output] = set_input_output(inputs[0], outputs[0]);

  auto in_ptr = input.data<void>(),
  auto out_ptr = output.data<void>(),
  
  std::fprintf(stderr, "set_input_output branch: %s\n", branch_name(used_branch));

  log_ptr_if_changed("in",  in_ptr,  input.nbytes());
  log_ptr_if_changed("out", out_ptr, output.nbytes());

  auto& encoder = cu::get_command_encoder(stream());
  encoder.set_input_array(input);
  encoder.set_output_array(output);

  auto capture = encoder.capture_context();
  auto& s = stream();

  switch (reduce_type_) {
    case Sum:
      distributed::detail::all_sum(group(), input, output, s);
      break;
    case Max:
      distributed::detail::all_max(group(), input, output, s);
      break;
    case Min:
      distributed::detail::all_min(group(), input, output, s);
      break;
    default:
      throw std::runtime_error(
          "Only all reduce sum, max, and min are supported.");
  }
}
} // namespace mlx::core::distributed
