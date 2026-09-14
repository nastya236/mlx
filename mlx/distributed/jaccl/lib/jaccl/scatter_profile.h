// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace jaccl {

struct alignas(64) ScatterProfile {
  enum Category { Copy, Reduce, Poll, PostSend, PostRecv, Count };
  inline static std::atomic<unsigned long long> next_call{0};
  double seconds[Count] = {};
  double start = 0;
  double elapsed = 0;
  double cpu_start = 0;
  double cpu = 0;
  double blocked_start[2] = {};
  double blocked_seconds[2] = {};
  unsigned long long blocked_events[2] = {};
  unsigned long long polls = 0;
  unsigned long long empty_polls = 0;
  unsigned long long completions = 0;
  unsigned long long buffer_bytes = 0;

  static double wall_time() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static double cpu_time() {
    timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
      return 0;
    }
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }

  // These intervals overlap the operation timers and each other.
  void set_blocked(int lr, bool blocked) {
    if (blocked && blocked_start[lr] == 0) {
      blocked_start[lr] = wall_time();
      blocked_events[lr]++;
    } else if (!blocked && blocked_start[lr] != 0) {
      blocked_seconds[lr] += wall_time() - blocked_start[lr];
      blocked_start[lr] = 0;
    }
  }

  void report(
      unsigned long long call,
      int rank,
      int ranks,
      int wires,
      int wire,
      unsigned long long output_bytes,
      int element_bytes,
      double dispatch_start,
      double dispatch_seconds) const {
    double measured = 0;
    for (double seconds_in_category : seconds) {
      measured += seconds_in_category;
    }
    std::fprintf(
        stderr,
        "JACCL_SCATTER_PROFILE {\"call\":%llu,\"rank\":%d,\"ranks\":%d,"
        "\"wires\":%d,\"wire\":%d,\"output_bytes\":%llu,\"element_bytes\":%d,"
        "\"buffer_bytes\":%llu,\"dispatch_ms\":%.6f,\"start_delay_ms\":%.6f,"
        "\"worker_ms\":%.6f,\"cpu_ms\":%.6f,\"copy_ms\":%.6f,"
        "\"reduce_ms\":%.6f,\"poll_ms\":%.6f,\"post_send_ms\":%.6f,"
        "\"post_recv_ms\":%.6f,\"other_ms\":%.6f,\"polls\":%llu,"
        "\"empty_polls\":%llu,\"completions\":%llu,"
        "\"blocked_ms\":[%.6f,%.6f],\"blocked_events\":[%llu,%llu]}\n",
        call,
        rank,
        ranks,
        wires,
        wire,
        output_bytes,
        element_bytes,
        buffer_bytes,
        dispatch_seconds * 1e3,
        (start - dispatch_start) * 1e3,
        elapsed * 1e3,
        cpu * 1e3,
        seconds[Copy] * 1e3,
        seconds[Reduce] * 1e3,
        seconds[Poll] * 1e3,
        seconds[PostSend] * 1e3,
        seconds[PostRecv] * 1e3,
        (elapsed - measured) * 1e3,
        polls,
        empty_polls,
        completions,
        blocked_seconds[0] * 1e3,
        blocked_seconds[1] * 1e3,
        blocked_events[0],
        blocked_events[1]);
  }
};

} // namespace jaccl
