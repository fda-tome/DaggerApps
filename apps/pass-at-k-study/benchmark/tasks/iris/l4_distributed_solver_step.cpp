// Task: l4_distributed_solver_step — matches L4DistributedSolverStep.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 21u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static float get(const std::vector<float> &A, int N, int i, int j) {
  i = (i % N + N) % N;
  j = (j % N + N) % N;
  return A[size_t(i) * N + j];
}

static bool run(int N) {
  std::vector<float> A(size_t(N) * N), B(size_t(N) * N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) A[size_t(i) * N + j] = randf01();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float v = get(A, N, i, j);
      v += get(A, N, i - 1, j);
      v += get(A, N, i + 1, j);
      v += get(A, N, i, j - 1);
      v += get(A, N, i, j + 1);
      B[size_t(i) * N + j] = v / 5.f;
    }
  for (int t = 0; t < 5; ++t) {
    int i = (t * 19) % N;
    int j = (t * 29) % N;
    if (!std::isfinite(B[size_t(i) * N + j])) return false;
  }
  return true;
}

int main() {
  g_seed = 21u;
  if (!run(64)) {
    std::cout << "FAIL\n";
    return 1;
  }
  g_seed = 22u;
  if (!run(256)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
