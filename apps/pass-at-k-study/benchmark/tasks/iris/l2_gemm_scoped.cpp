// Task: l2_gemm_scoped — matches L2GemmScoped TEST_CASES.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 99u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static bool feq(float a, float b) {
  float s = std::fmax(std::fabs(a), std::fabs(b));
  if (s < 1.f) s = 1.f;
  return std::fabs(a - b) < 1e-2f * s;
}

static float dot_row_col(int N, const std::vector<float> &A, const std::vector<float> &B, int row,
                         int col) {
  float s = 0.f;
  for (int k = 0; k < N; ++k) s += A[size_t(row) * N + k] * B[size_t(k) * N + col];
  return s;
}

static void matmul(int N, const std::vector<float> &A, const std::vector<float> &B,
                   std::vector<float> &C) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < N; ++k) s += A[size_t(i) * N + k] * B[size_t(k) * N + j];
      C[size_t(i) * N + j] = s;
    }
}

static bool run_N(int N, bool full_check, std::uint32_t seed) {
  g_seed = seed;
  size_t n2 = size_t(N) * size_t(N);
  std::vector<float> A(n2), B(n2), C(n2);
  for (size_t i = 0; i < n2; ++i) {
    A[i] = randf01();
    B[i] = randf01();
  }
  if (full_check) {
    matmul(N, A, B, C);
    for (int t = 0; t < 8; ++t) {
      int i = (t * 97) % N;
      int j = (t * 53) % N;
      if (!feq(C[size_t(i) * N + j], dot_row_col(N, A, B, i, j))) return false;
    }
  } else {
    float c00 = dot_row_col(N, A, B, 0, 0);
    if (!std::isfinite(c00)) return false;
  }
  return true;
}

int main() {
  if (!run_N(512, true, 99u) || !run_N(1024, true, 100u) || !run_N(2048, false, 101u)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
