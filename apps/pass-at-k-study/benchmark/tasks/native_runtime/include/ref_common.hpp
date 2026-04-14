#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace ref_common {

inline std::uint32_t &rng_seed() {
  static thread_local std::uint32_t s = 1u;
  return s;
}

inline void set_seed(std::uint32_t s) { rng_seed() = s; }

inline float randf01() {
  std::uint32_t &g = rng_seed();
  g = g * 1103515245u + 12345u;
  return float(g & 0xffffffu) / float(0x1000000u);
}

inline bool feq(float a, float b) {
  float s = std::fmax(std::fabs(a), std::fabs(b));
  if (s < 1.f) s = 1.f;
  return std::fabs(a - b) < 1e-3f * s;
}

inline bool feq_loose(float a, float b) {
  float s = std::fmax(std::fabs(a), std::fabs(b));
  if (s < 1.f) s = 1.f;
  return std::fabs(a - b) < 1e-2f * s;
}

}  // namespace ref_common
