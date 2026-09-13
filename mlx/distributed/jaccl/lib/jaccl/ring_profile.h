// Copyright © 2026 Apple Inc.

#pragma once

#include <pthread.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <vector>

namespace jaccl {

inline double ring_profile_wall_time() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct RingProfileStamp {
  double wall = 0;
  double cpu = 0;

  static RingProfileStamp now() {
    double wall = ring_profile_wall_time();
    timespec ts;
    double cpu = std::numeric_limits<double>::quiet_NaN();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
      cpu = ts.tv_sec + ts.tv_nsec * 1e-9;
    }
    return {wall, cpu};
  }
};

struct alignas(64) RingWireProfile {
  uint64_t thread_id = 0;
  RingProfileStamp start;
  RingProfileStamp reduced;
  RingProfileStamp finished;
};

class RingProfile {
 public:
  explicit RingProfile(int n_wires) : wires(n_wires) {}

  static bool enabled() {
    static const bool enabled = [] {
      const char* value = std::getenv("JACCL_PROFILE");
      return value && std::string_view(value) == "1";
    }();
    return enabled;
  }

  void report(
      int rank,
      int nranks,
      int directions,
      size_t input_bytes,
      size_t element_bytes,
      bool inplace) const {
    double total_us = (ring_profile_wall_time() - start_) * 1e6;
    for (size_t wire = 0; wire < wires.size(); ++wire) {
      const auto& p = wires[wire];
      auto phase = [&](const char* name,
                       RingProfileStamp begin,
                       RingProfileStamp end) {
        std::fprintf(
            stderr,
            "[jaccl-profile] rank=%d call=%llu ranks=%d wires=%zu dirs=%d "
            "bytes=%zu element_bytes=%zu inplace=%d wire=%zu thread=%llu "
            "phase=%s start_us=%.3f wall_us=%.3f cpu_us=%.3f total_us=%.3f\n",
            rank,
            call_,
            nranks,
            wires.size(),
            directions,
            input_bytes,
            element_bytes,
            int(inplace),
            wire,
            static_cast<unsigned long long>(p.thread_id),
            name,
            (begin.wall - start_) * 1e6,
            (end.wall - begin.wall) * 1e6,
            (end.cpu - begin.cpu) * 1e6,
            total_us);
      };
      phase("reduce_scatter", p.start, p.reduced);
      phase("all_gather", p.reduced, p.finished);
    }
  }

  std::vector<RingWireProfile> wires;

 private:
  inline static std::atomic<unsigned long long> next_call_{0};
  unsigned long long call_ = next_call_.fetch_add(1, std::memory_order_relaxed);
  double start_ = ring_profile_wall_time();
};

} // namespace jaccl
