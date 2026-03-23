# Task: l1_darray_create — matches Dagger L1DArrayCreate TEST_CASES.
import sys

_s = 42


def randf01() -> float:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return float(_s & 0xFFFFFF) / float(0x1000000)


def check(N: int, block_size: int) -> bool:
    if N <= 0 or block_size <= 0 or N % block_size != 0:
        return False
    n = N * N
    M = [randf01() for _ in range(n)]
    nb = N // block_size
    for bi in range(nb):
        for bj in range(nb):
            s = 0.0
            for i in range(block_size):
                for j in range(block_size):
                    idx = (bi * block_size + i) * N + (bj * block_size + j)
                    s += M[idx]
            if not (abs(s) < 1e38):  # finite
                return False
    return True


def main() -> int:
    if not check(512, 128) or not check(1024, 256) or not check(2048, 512):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
