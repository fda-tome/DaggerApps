// Task: l3_partitioned_exchange_validate — matches L3PartitionedExchangeValidate.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 11u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static bool run(int N) {
  std::vector<float> A(size_t(N) * N), R(size_t(N) * N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) A[size_t(i) * N + j] = randf01();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      R[size_t(i) * N + j] = A[size_t(i) * N + j] + A[size_t(j) * N + i];
  for (int t = 0; t < 5; ++t) {
    int i = (t * 17) % N;
    int j = (t * 31) % N;
    float got = R[size_t(i) * N + j];
    float exp = A[size_t(i) * N + j] + A[size_t(j) * N + i];
    if (std::fabs(got - exp) > 1e-4f) return false;
  }
  return true;
}

int main() {
  g_seed = 11u;
  if (!run(128)) {
    std::cout << "FAIL\n";
    return 1;
  }
  g_seed = 12u;
  if (!run(512)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
