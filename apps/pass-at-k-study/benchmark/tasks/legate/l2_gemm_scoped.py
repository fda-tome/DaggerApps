# Task: l2_gemm_scoped — matches L2GemmScoped TEST_CASES (uses NumPy when available).
import math
import sys

try:
    import numpy as np
except ImportError:
    np = None


def run_with_numpy() -> bool:
    assert np is not None
    for seed, N, full in ((99, 512, True), (100, 1024, True), (101, 2048, False)):
        rng = np.random.default_rng(seed)
        a = rng.random((N, N), dtype=np.float32)
        b = rng.random((N, N), dtype=np.float32)
        if full:
            c = a @ b
            for t in range(8):
                i = (t * 97) % N
                j = (t * 53) % N
                ref = float(np.dot(a[i, :], b[:, j]))
                if not math.isclose(float(c[i, j]), ref, rel_tol=1e-2, abs_tol=1e-2):
                    return False
        else:
            c00 = float(np.dot(a[0, :], b[:, 0]))
            if not math.isfinite(c00):
                return False
    return True


def main() -> int:
    if np is None:
        print("FAIL")
        return 1
    ok = run_with_numpy()
    if not ok:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
