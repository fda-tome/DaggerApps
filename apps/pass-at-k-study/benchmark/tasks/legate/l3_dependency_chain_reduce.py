# Task: l3_dependency_chain_reduce — matches L3DependencyChainReduce.
import math
import sys

_s = 3


def randf01() -> float:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return float(_s & 0xFFFFFF) / float(0x1000000)


def chain(n: int) -> float:
    nn = n * n
    a = [randf01() for _ in range(nn)]
    return sum(x * 2.0 for x in a)


def main() -> int:
    global _s
    _s = 3
    a = chain(128)
    _s = 4
    b = chain(512)
    if not math.isfinite(a) or not math.isfinite(b):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
