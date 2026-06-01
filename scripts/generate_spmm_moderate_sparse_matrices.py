#!/usr/bin/env python3

import os
import random
import re
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRICES_ROOT = os.path.join(REPO_ROOT, "matrices")
OUT_DIR = os.path.join(MATRICES_ROOT, "synthetic_moderate")

_FILL_RE = re.compile(r"fill \[(\d+)%\s*,\s*(\d+)%\]", re.I)

# (filename, M, N, density_lo, density_hi) — per-row nnz fraction in [lo, hi]
MATRIX_SPECS = [
    ("spmm_mod_M50000_N1000.mtx", 50_000, 1_000, 0.30, 0.32),
    ("spmm_mod_M100000_N1000.mtx", 100_000, 1_000, 0.33, 0.35),
    ("spmm_mod_M200000_N1000.mtx", 200_000, 1_000, 0.46, 0.48),
    ("spmm_mod_M5000_N5000.mtx", 5_000, 5_000, 0.38, 0.40),
    ("spmm_mod_M8000_N8000.mtx", 8_000, 8_000, 0.40, 0.42),
    ("spmm_mod_M10000_N10000.mtx", 10_000, 10_000, 0.43, 0.45),
    ("spmm_mod_M1000_N50000.mtx", 1_000, 50_000, 0.48, 0.50),
    ("spmm_mod_M1000_N100000.mtx", 1_000, 100_000, 0.42, 0.44),
    ("spmm_mod_M1000_N200000.mtx", 1_000, 200_000, 0.34, 0.36),
]


def _fill_tag(lo, hi):
    return round(lo * 100), round(hi * 100)


def _mtx_ok(path, M, N, lo, hi):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    want = _fill_tag(lo, hi)
    got_m = got_n = None
    got_fill = None
    with open(path) as f:
        for line in f:
            if line.startswith("%"):
                m = _FILL_RE.search(line)
                if m:
                    got_fill = int(m.group(1)), int(m.group(2))
                continue
            parts = line.split()
            got_m, got_n = int(parts[0]), int(parts[1])
            break
    return got_m == M and got_n == N and got_fill == want


def _write_mtx(path, M, N, seed, lo, hi):
    rng = random.Random(seed)
    lo_pct, hi_pct = _fill_tag(lo, hi)
    tmp = path + ".tmp"
    nnz = 0
    with open(tmp, "w") as body:
        for r in range(1, M + 1):
            k_lo = max(1, int(lo * N))
            k_hi = min(N, max(k_lo, int(hi * N)))
            k = rng.randint(k_lo, k_hi)
            for c in sorted(rng.sample(range(1, N + 1), k)):
                body.write(f"{r} {c} {rng.uniform(-1, 1):.7f}\n")
                nnz += 1
    with open(path, "w") as out, open(tmp) as body:
        out.write("%%MatrixMarket matrix coordinate real general\n")
        out.write(f"% M={M} N={N} fill [{lo_pct}%,{hi_pct}%] (per-row band)\n")
        out.write(f"{M} {N} {nnz}\n")
        shutil.copyfileobj(body, out)
    os.remove(tmp)
    return nnz


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (name, M, N, lo, hi) in enumerate(MATRIX_SPECS):
        if M % 2 or N % 2:
            raise ValueError(f"{name}: M,N must be even for FBCSR")
        path = os.path.join(OUT_DIR, name)
        if _mtx_ok(path, M, N, lo, hi):
            print(f"exists {path}")
            continue
        print(f"writing {path}")
        nnz = _write_mtx(path, M, N, 10_000 + i, lo, hi)
        print(f"wrote {path} nnz={nnz:,}")

    with open(os.path.join(OUT_DIR, "list.txt"), "w") as f:
        for name, *_ in MATRIX_SPECS:
            f.write(name + "\n")
    print(f"wrote {OUT_DIR}/list.txt")


if __name__ == "__main__":
    main()


# Transparency in the use of AI tools:
# Parts of this script were fixed and refactored with AI-assisted coding. 
