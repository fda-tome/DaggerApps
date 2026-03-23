#ifndef NATIVE_RUNTIME_REF_COMMON_H
#define NATIVE_RUNTIME_REF_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>

static inline void ref_set_seed(uint32_t *g, uint32_t s) { *g = s; }

static inline float ref_randf01(uint32_t *g) {
  *g = *g * 1103515245u + 12345u;
  return (float)(*g & 0xffffffu) / (float)0x1000000u;
}

static inline int ref_feq(float a, float b) {
  float s = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
  if (s < 1.f) s = 1.f;
  return fabsf(a - b) < 1e-3f * s;
}

static inline int ref_feq_loose(float a, float b) {
  float s = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
  if (s < 1.f) s = 1.f;
  return fabsf(a - b) < 1e-2f * s;
}

#endif
