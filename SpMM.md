SpMM Documentation

-------------------------------------

## Changed & added files

```text
ginkgo/
├── SpMM.md
│
├── benchmark/utils/
│   └── formats.hpp              # csr_spmm_v1..v3, sellp_spmm_v1..v2, fbcsr_spmm_v1..v3
│
├── common/cuda_hip/matrix/
│   ├── csr_kernels.instantiate.cpp             # SpMM instantiations
│   ├── csr_kernels.template.cpp                # spmm stub → spmv
│   ├── sellp_kernels.cpp                       # spmm stub → spmv
│   └── fbcsr_kernels.template.cpp              # spmm stub → spmv
│
├── core/
│   ├── device_hooks/common_kernels.inc.cpp      # GKO_STUB: CSR, SELLP, FBCSR spmm
│   │
│   └── matrix/
│       ├── csr.cpp                       # apply_impl: spmm if nrhs > 2.
│       ├── csr_kernels.hpp               # Declarations.
│       │
│       ├── sellp.cpp                     # apply_impl: spmm if nrhs > 2 + spmm_version_ field.
│       ├── sellp_kernels.hpp             # Declarations.
│       │
│       ├── fbcsr.cpp                     # apply_impl: spmm if nrhs > 2 + spmm_version_ field.
│       └── fbcsr_kernels.hpp             # SpMM declarations
│
├── dpcpp/matrix/
│   ├── csr_kernels.dp.cpp                # spmm stub → spmv
│   ├── sellp_kernels.dp.cpp              # spmm stub → spmv
│   └── fbcsr_kernels.dp.cpp              # spmm stub → spmv
│
├── include/ginkgo/core/matrix/
│   ├── csr.hpp                           # omp_spmm_variant strategy (spmm_v1..v3 on OMP)
│   ├── sellp.hpp                         # set/get_spmm_version (1..2 on OMP)
│   └── fbcsr.hpp                         # set/get_spmm_version (1..3 on OMP)
│
├── matrices/
│   ├── synthetic_moderate/               # .mtx + list.txt (generator script)
│   └── Suite_Sparse_Matrices_SpMM/       # ssget Group/Name entries in list.txt
│
├── omp/matrix/
│   ├── csr_kernels.cpp                # spmm_v1..v3; dispatch via strategy
│   ├── sellp_kernels.cpp              # spmm_v1..v2; dispatch via spmm_version_
│   └── fbcsr_kernels.cpp              # spmm_v1..v3; dispatch via spmm_version_
│
├── omp/test/matrix/
│   ├── csr_spmm_variant_kernels.cpp         # ref vs OMP, variants 1..3, nrhs=16
│   ├── sellp_spmm_variant_kernels.cpp       # ref vs OMP, variants 1..2, nrhs=16
│   └── fbcsr_spmm_variant_kernels.cpp       # ref vs OMP, variants 1..3, nrhs=16
│
├── reference/matrix/
│   ├── csr_kernels.cpp                    # spmm → spmv (reference baseline)
│   ├── sellp_kernels.cpp                  # spmm → spmv (reference baseline)
│   └── fbcsr_kernels.cpp                  # spmm → spmv (reference baseline)
│
└── scripts/
    └── generate_spmm_moderate_sparse_matrices.py
```

-------------------------------------

## Implementations overview

- **CSR-v0** — Reused multi-RHS SpMV: dot product
- **CSR-v1** — Sequential Gustavson
- **CSR-v2** — Gustavson + parallelised over rows
- **CSR-v3** — Gustavson + parallelised over rows + SIMD over dense columns
- **SELL-P-v0** — Reused multi-RHS SpMV
- **SELL-P-v1** — Parallel per-slice Gustavson over the (slice, row) space (static)
- **SELL-P-v2** — SELL-P-v1 with SIMD over the K dense columns
- **FBCSR-v0** — Reused multi-RHS SpMV: sequential Gustavson
- **FBCSR-v1** — Parallel block-row Gustavson
- **FBCSR-v2** — Parallel block-row Gustavson + SIMD over dense columns
- **FBCSR-v3** — Parallel block-row Gustavson + SIMD over dense columns + 2×2 register tile and B-row reuse

-------------------------------------

## Cluster (gpu-nvidia-h200-2, AMD EPYC 9555)

### Synthetic matrices (`matrices/synthetic_moderate/`)

Generate once (from repo root):

```sh
python3 scripts/generate_spmm_moderate_sparse_matrices.py
```

### SuiteSparse matrices

Listed in `matrices/Suite_Sparse_Matrices_SpMM/list.txt` as ssget `Group/Name`. 

### Pre-flight (once per build)

```sh
lscpu | grep 'Model name'    # expect AMD EPYC 9555
```

### Environment

```sh
export GINKGO_BENCHMARK_PRECISION=double
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export REPETITIONS=10
export WARMUP=1
```

### Build

```sh
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -mavx512f -mavx512vl -mavx512dq -mprefer-vector-width=512" \
  -DGINKGO_BUILD_REFERENCE=ON \
  -DGINKGO_BUILD_OMP=ON \
  -DGINKGO_BUILD_TESTS=ON \
  -DGINKGO_BUILD_BENCHMARKS=ON \
  -DGINKGO_BUILD_EXAMPLES=OFF \
  -DGINKGO_BUILD_CUDA=OFF \
  -DGINKGO_BUILD_HIP=OFF \
  -DGINKGO_BUILD_SYCL=OFF \
  -DGINKGO_BUILD_MPI=OFF \
  -DGINKGO_BUILD_DOC=OFF

cmake --build . \
  --target omp_test_matrix_csr_spmm_variant_kernels \
           omp_test_matrix_sellp_spmm_variant_kernels \
           omp_test_matrix_fbcsr_spmm_variant_kernels \
           spmv \
  -j"$(nproc)"
```

### OMP SpMM correctness tests

```sh
./omp/test/matrix/csr_spmm_variant_kernels
./omp/test/matrix/sellp_spmm_variant_kernels
./omp/test/matrix/fbcsr_spmm_variant_kernels
```
