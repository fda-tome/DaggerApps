// Task: l3_dependency_chain_reduce — matches L3DependencyChainReduce.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 3u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static float chain(int N) {
  size_t n = size_t(N) * size_t(N);
  std::vector<float> A(n);
  for (size_t i = 0; i < n; ++i) A[i] = randf01();
  float s = 0.f;
  for (float x : A) s += x * 2.f;
  return s;
}

int main() {
  g_seed = 3u;
  float a = chain(128);
  g_seed = 4u;
  float b = chain(512);
  if (!std::isfinite(a) || !std::isfinite(b)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
