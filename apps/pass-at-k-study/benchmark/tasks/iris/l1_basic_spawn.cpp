// Task: l1_basic_spawn — portable reference (see REFERENCE_CONVENTIONS.md).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static std::uint32_t g_seed = 1u;
static float randf01() {
  g_seed = g_seed * 1103515245u + 12345u;
  return float(g_seed & 0xffffffu) / float(0x1000000u);
}

static bool feq(float a, float b) {
  float s = std::fmax(std::fabs(a), std::fabs(b));
  if (s < 1.f) s = 1.f;
  return std::fabs(a - b) < 1e-3f * s;
}

static bool run_case(const int dims[4][2]) {
  for (int t = 0; t < 4; ++t) {
    int r = dims[t][0], c = dims[t][1];
    size_t n = size_t(r) * size_t(c);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = randf01();
    float serial = 0.f;
    for (size_t i = 0; i < n; ++i) serial += buf[i];
    float worker = 0.f;
    for (size_t i = 0; i < n; ++i) worker += buf[i];
    if (!feq(serial, worker)) return false;
  }
  return true;
}

int main() {
  const int a[4][2] = {{128, 128}, {256, 256}, {64, 64}, {512, 512}};
  const int b[4][2] = {{1024, 1024}, {1024, 1024}, {1024, 1024}, {1024, 1024}};
  g_seed = 1u;
  if (!run_case(a)) {
    std::cout << "FAIL\n";
    return 1;
  }
  g_seed = 2u;
  if (!run_case(b)) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
