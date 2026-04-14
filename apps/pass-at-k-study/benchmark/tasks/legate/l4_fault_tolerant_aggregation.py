# Task: l4_fault_tolerant_aggregation — matches L4FaultTolerantAggregation.
import math
import sys


def aggregate(n: int) -> float:
    s = 0.0
    for i in range(1, 5):
        v = float("nan") if i == 3 else float(i * n)
        if math.isfinite(v):
            s += v
    return s


def main() -> int:
    a = aggregate(10)
    b = aggregate(100)
    if abs(a - 70.0) > 1e-3 or abs(b - 700.0) > 1e-3:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
