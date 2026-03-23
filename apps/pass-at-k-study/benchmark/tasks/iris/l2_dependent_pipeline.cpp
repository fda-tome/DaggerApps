// Task: l2_dependent_pipeline — matches L2DependentPipeline TEST_CASES.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 7u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static float pipeline(int N) {
  size_t n = size_t(N) * size_t(N);
  std::vector<float> A(n);
  for (size_t i = 0; i < n; ++i) A[i] = randf01();
  float mx = 0.f;
  for (float x : A)
    if (x > mx) mx = x;
  if (mx < 1e-6f) mx = 1e-6f;
  float s = 0.f;
  for (float x : A) s += x / mx;
  return s;
}

int main() {
  int Ns[] = {256, 1024, 4096};
  g_seed = 7u;
  for (int N : Ns) {
    float v = pipeline(N);
    if (!std::isfinite(v)) {
      std::cout << "FAIL\n";
      return 1;
    }
  }
  std::cout << "PASS\n";
  return 0;
}
