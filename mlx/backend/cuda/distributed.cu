// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/distributed/primitives.h"
#include "mlx/primitives.h"

#include <cassert>
#include <cstdio>
#include <iostream>

struct PtrStamp {
  const void* ptr = nullptr;
  size_t nbytes = 0;
};
static PtrStamp g_in_stamp, g_out_stamp;

static inline void
log_ptr_if_changed(const char* tag, const void* cur_ptr, size_t nbytes) {
  PtrStamp& s = (tag[0] == 'i') ? g_in_stamp : g_out_stamp;
  const bool changed = (s.ptr != cur_ptr) || (s.nbytes != nbytes);
  if (changed) {
    std::fprintf(
        stderr, "[%s] ptr=%p bytes=%zu <-- CHANGED\n", tag, cur_ptr, nbytes);
    s.ptr = cur_ptr;
    s.nbytes = nbytes;
  }
}

namespace mlx::core::distributed {
void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);

  auto set_input_output = [this, s = stream()](
                              const array& in,
                              array& out) -> std::pair<array, array> {
    if (!in.flags().row_contiguous) {
      std::cout << "set_input_output branch: copy_gpu" << std::endl;
      copy_gpu(in, out, CopyType::General, s);
      return {out, out};
    } else if (in.is_donatable()) {
      std::cout << "set_input_output branch: donated" << std::endl;
      out.copy_shared_buffer(in);
      return {in, out};
    } else {
      std::cout << "set_input_output branch: allocated" << std::endl;
      out.set_data(allocator::malloc(out.nbytes()));
      return {in, out};
    }
  };

  auto [input, output] = set_input_output(inputs[0], outputs[0]);
  const void* in_ptr = input.data<void>() const void* out_ptr =
      output.data<void>();

  log_ptr_if_changed("in", in_ptr, input.nbytes());
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
