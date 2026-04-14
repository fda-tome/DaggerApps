# Task: l4_distributed_solver_step — matches L4DistributedSolverStep.
import math
import sys

_s = 21


def randf01() -> float:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return float(_s & 0xFFFFFF) / float(0x1000000)


def get(a, n, i, j):
    return a[i % n][j % n]


def run(n):
    a = [[randf01() for _ in range(n)] for _ in range(n)]
    b = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = get(a, n, i, j)
            v += get(a, n, i - 1, j)
            v += get(a, n, i + 1, j)
            v += get(a, n, i, j - 1)
            v += get(a, n, i, j + 1)
            b[i][j] = v / 5.0
    for t in range(5):
        i = (t * 19) % n
        j = (t * 29) % n
        if not math.isfinite(b[i][j]):
            return False
    return True


def main() -> int:
    global _s
    _s = 21
    if not run(64):
        print("FAIL")
        return 1
    _s = 22
    if not run(256):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
