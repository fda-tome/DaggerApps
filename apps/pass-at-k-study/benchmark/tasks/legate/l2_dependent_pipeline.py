# Task: l2_dependent_pipeline — matches L2DependentPipeline TEST_CASES.
import math
import sys

_s = 7


def randf01() -> float:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return float(_s & 0xFFFFFF) / float(0x1000000)


def pipeline(n: int) -> float:
    nn = n * n
    a = [randf01() for _ in range(nn)]
    mx = max(a) if a else 0.0
    if mx < 1e-6:
        mx = 1e-6
    return sum(x / mx for x in a)


def main() -> int:
    global _s
    _s = 7
    for n in (256, 1024, 4096):
        v = pipeline(n)
        if not math.isfinite(v):
            print("FAIL")
            return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
