// Task: l4_fault_tolerant_aggregation — matches L4FaultTolerantAggregation.
#include <cmath>
#include <iostream>

static float aggregate(int N) {
  float s = 0.f;
  for (int i = 1; i <= 4; ++i) {
    float v = (i == 3) ? NAN : float(i) * float(N);
    if (std::isfinite(v)) s += v;
  }
  return s;
}

int main() {
  float a = aggregate(10);
  float b = aggregate(100);
  if (std::fabs(a - 70.f) > 1e-3f || std::fabs(b - 700.f) > 1e-3f) {
    std::cout << "FAIL\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
