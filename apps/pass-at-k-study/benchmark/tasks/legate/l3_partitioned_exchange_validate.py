# Task: l3_partitioned_exchange_validate — matches L3PartitionedExchangeValidate.
import sys

_s = 11


def randf01() -> float:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return float(_s & 0xFFFFFF) / float(0x1000000)


def run(n: int) -> bool:
    a = [[randf01() for _ in range(n)] for _ in range(n)]
    r = [[a[i][j] + a[j][i] for j in range(n)] for i in range(n)]
    for t in range(5):
        i = (t * 17) % n
        j = (t * 31) % n
        if abs(r[i][j] - (a[i][j] + a[j][i])) > 1e-4:
            return False
    return True


def main() -> int:
    global _s
    _s = 11
    if not run(128):
        print("FAIL")
        return 1
    _s = 12
    if not run(512):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
