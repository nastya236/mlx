// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"

#include <cassert>

#include <cuda_runtime_api.h>
#include <cstdio>

enum class SIOBranch : uint8_t { CopyMade, Donated, FreshAlloc };

static const char* branch_name(SIOBranch b) {
  switch (b) {
    case SIOBranch::CopyMade:
      return "copy_gpu";
    case SIOBranch::Donated:
      return "donated";
    case SIOBranch::FreshAlloc:
      return "allocated";
    default:
      return "unknown";
  }
}

static void print_ptr_change(
    const char* tag,
    int rank,
    int step,
    const void* cur_ptr,
    size_t nbytes) {
  static const void* prev_in = nullptr;
  static size_t prev_in_sz = 0;
  static const void* prev_out = nullptr;
  static size_t prev_out_sz = 0;

  const bool is_in = (tag[0] == 'i'); // "in" or "out"
  const void*& prev_ptr = is_in ? prev_in : prev_out;
  size_t& prev_sz = is_in ? prev_in_sz : prev_out_sz;

  const bool changed = (cur_ptr != prev_ptr) || (nbytes != prev_sz);
  if (changed) {
    std::fprintf(
        stderr,
        "[rank %d] step %d %-3s ptr=%p bytes=%zu <-- CHANGED\n",
        rank,
        step,
        tag,
        cur_ptr,
        nbytes);
    prev_ptr = cur_ptr;
    prev_sz = nbytes;

    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, cur_ptr) == cudaSuccess) {
#if CUDART_VERSION >= 11000
      std::fprintf(
          stderr,
          "            type=%d (0=Host,1=Device,2=Managed)\n",
          int(attr.type));
#else
      std::fprintf(
          stderr,
          "            memoryType=%d (1=Host,2=Device)\n",
          int(attr.memoryType));
#endif
    }
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
  SIOBranch used_branch = SIOBranch::FreshAlloc; // will be set below

  auto set_input_output = [this, &used_branch, s = stream()](
                              const array& in,
                              array& out) -> std::pair<array, array> {
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

  std::fprintf(
      stderr,
      "[rank %d] step %d set_input_output: %s\n",
      rank_,
      global_step_,
      branch_name(used_branch));

  const void *in_ptr = input.data<void>(),
             const void *out_ptr = output.data<void>(),

             print_ptr_change(
                 "in", rank_, global_step_, in_ptr, input.nbytes());
  print_ptr_change("out", rank_, global_step_, out_ptr, output.nbytes());

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
