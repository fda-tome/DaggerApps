# Task: l1_basic_spawn — portable reference (see REFERENCE_CONVENTIONS.md).
import sys
from concurrent.futures import ThreadPoolExecutor


def _rng() -> int:
    global _s
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    return _s


_s = 1


def randf01() -> float:
    return float(_rng() & 0xFFFFFF) / float(0x1000000)


def feq(a: float, b: float) -> bool:
    s = max(abs(a), abs(b), 1.0)
    return abs(a - b) < 1e-3 * s


def run_case(dims_list, seed):
    global _s
    _s = seed
    bufs = []
    for r, c in dims_list:
        n = r * c
        bufs.append([randf01() for _ in range(n)])
    serial = [sum(b) for b in bufs]

    def sum_idx(i):
        return sum(bufs[i])

    with ThreadPoolExecutor(max_workers=4) as ex:
        worker = list(ex.map(sum_idx, range(4)))
    return all(feq(a, b) for a, b in zip(serial, worker))


def main() -> int:
    a = [(128, 128), (256, 256), (64, 64), (512, 512)]
    b = [(1024, 1024)] * 4
    if not run_case(a, 1) or not run_case(b, 2):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
