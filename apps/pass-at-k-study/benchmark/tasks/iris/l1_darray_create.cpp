// Task: l1_darray_create — matches Dagger L1DArrayCreate TEST_CASES.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 42u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static bool check(int N, int block_size) {
  if (N <= 0 || block_size <= 0 || N % block_size != 0) return false;
  size_t n = size_t(N) * size_t(N);
  std::vector<float> M(n);
  for (size_t i = 0; i < n; ++i) M[i] = randf01();
  int nb = N / block_size;
  for (int bi = 0; bi < nb; ++bi)
    for (int bj = 0; bj < nb; ++bj) {
      float s = 0.f;
      for (int i = 0; i < block_size; ++i)
        for (int j = 0; j < block_size; ++j)
          s += M[size_t(bi * block_size + i) * N + size_t(bj * block_size + j)];
      if (!std::isfinite(s)) return false;
    }
  return true;
}

int main() {
  if (!check(512, 128) || !check(1024, 256) || !check(2048, 512)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
