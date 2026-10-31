# NH81 GB300: resolved requirements for PI review

Generated 2026-10-31 on electron-server with uv 0.13.5, **nothing
installed** (resolution only):

```
uv pip compile <set>.in --python-platform aarch64-manylinux_2_35 \
  --python-version 3.12 --generate-hashes \
  --exclude-newer 2026-10-17T00:00:00Z
```

* **Platform tag: `aarch64-manylinux_2_35`, not the requested
  `aarch64-manylinux_2_28`.** With `_2_28` the resolution fails:
  `mlx` 0.31.1 and `mlx-cuda-*` 0.31.1 only ship `manylinux_2_35`
  aarch64 wheels. The GB300 runs Ubuntu 24.04 (glibc 2.39), which
  accepts `manylinux_2_35`.
* **Two-week rule:** `--exclude-newer 2026-10-17` (two weeks before the
  slot); every pinned file below was uploaded before that date.
* **aarch64 wheels:** every package has a Linux aarch64 (or pure Python)
  wheel for CPython 3.12; no sdist build is needed. The tables give the
  file pip will take on the GB300 and its SHA-256, checked to be among
  the hashes of the requirements file. The requirements files also list
  the hashes of the other platforms' files of the same versions.
* Install on the GB300 (operator, after review):
  `uv pip install --require-hashes --no-deps -r <requirements file>`.

## `requirements-gb300-nh81.txt`

Input (`requirements-gb300-nh81.in`):

    mlx==0.31.1
    mlx-cuda-12==0.31.1
    numpy==2.2.6
    nvidia-cuda-runtime-cu12==12.9.*
    nvidia-cuda-cccl-cu12==12.9.*

| Package | Version | Uploaded | Linux aarch64 file | SHA-256 | in file |
|---|---|---|---|---|---|
| mlx | 0.31.1 | 2026-03-12 | `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | `1ed1991c8e39f841d5756c0c543beb819763a2f80fba3f4b150bc6cad4d973de` | yes |
| mlx-cuda-12 | 0.31.1 | 2026-03-12 | `mlx_cuda_12-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | `011d611b5b31a948a5eef5ea78a5b3fba41908817d2f094d9bd125be4abae013` | yes |
| numpy | 2.2.6 | 2025-05-17 | `numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87` | yes |
| nvidia-cublas-cu12 | 12.9.2.10 | 2026-04-08 | `nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_aarch64.whl` | `5572131a59c3eebeeb1c4c8144f772d49372c20124916e072a0e3fc30df421d5` | yes |
| nvidia-cuda-cccl-cu12 | 12.9.27 | 2025-05-01 | `nvidia_cuda_cccl_cu12-12.9.27-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `d7898b38aa68beaa234d48f0868273702342a196d6e2e9d0ef058dca2390ebea` | yes |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | 2025-06-05 | `nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `096d4de6bda726415dfaf3198d4f5c522b8e70139c97feef5cd2ca6d4cd9cead` | yes |
| nvidia-cuda-runtime-cu12 | 12.9.79 | 2025-06-05 | `nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `83469a846206f2a733db0c42e223589ab62fd2fabac4432d2f8802de4bded0a4` | yes |
| nvidia-cudnn-cu12 | 9.27.0.42 | 2026-09-29 | `nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_aarch64.whl` | `5bc22243563107ea06aa60f9116ad800063d5f4617cc75bc6670d5db78d9f336` | yes |
| nvidia-nccl-cu12 | 2.32.3 | 2026-09-22 | `nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_aarch64.whl` | `061af42ae1044816820e16b872075e243d1ac0656f51860fbc8e106c257aa83a` | yes |

## `requirements-gb300-nh81-cuda13.txt`

Input (`requirements-gb300-nh81-cuda13.in`):

    mlx==0.31.1
    mlx-cuda-13==0.31.1
    numpy==2.2.6
    nvidia-cuda-runtime==13.*
    nvidia-cuda-cccl==13.*

| Package | Version | Uploaded | Linux aarch64 file | SHA-256 | in file |
|---|---|---|---|---|---|
| mlx | 0.31.1 | 2026-03-12 | `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | `1ed1991c8e39f841d5756c0c543beb819763a2f80fba3f4b150bc6cad4d973de` | yes |
| mlx-cuda-13 | 0.31.1 | 2026-03-12 | `mlx_cuda_13-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | `fcbc4ac3764b5bfac7814f7fd8ad6369d9b5d1279d0b3b3bc62878b572e8580a` | yes |
| numpy | 2.2.6 | 2025-05-17 | `numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87` | yes |
| nvidia-cublas | 13.8.1.7 | 2026-09-28 | `nvidia_cublas-13.8.1.7-py3-none-manylinux_2_27_aarch64.whl` | `803b9974cd7164a481d0b11c25826498d383d0ef0c372573bc6577a344ddf74e` | yes |
| nvidia-cuda-cccl | 13.3.4.3.1 | 2026-09-16 | `nvidia_cuda_cccl-13.3.4.3.1-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `cd47e7823e3ab369a3e3df3a4c6dbb4cb02176cb85b251eec5f7c9cdfa528ec4` | yes |
| nvidia-cuda-nvrtc | 13.4.92 | 2026-09-16 | `nvidia_cuda_nvrtc-13.4.92-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `24b9f5eccc6a5a19779038cf468aecb7cecfa8716269beaef5640bd989d21c28` | yes |
| nvidia-cuda-runtime | 13.4.92 | 2026-09-16 | `nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `bef071788589550ab02846fcbc732a2b88c2b16f2cd8a0f45d98e68e8c5fb0c8` | yes |
| nvidia-cudnn-cu13 | 9.27.0.42 | 2026-09-29 | `nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_aarch64.whl` | `41b3aae756032ed13b1fcac39c79330cfc5434059d0c5251914e0f6ce582c7ea` | yes |
| nvidia-nccl-cu13 | 2.32.3 | 2026-09-22 | `nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_aarch64.whl` | `a5bee92b2f4af218c109f8d221c3c9adcc752b94ae0ffcb0ed5abf9341a7724c` | yes |

SHA-256 of the requirement files: `requirements-gb300-nh81.sha256`.

## PI approval

**Approved by the PI on 2026-10-31 at 22:48 CET** (HITL review of the
resolved sets `requirements-gb300-nh81.txt` and
`requirements-gb300-nh81-cuda13.txt`, as listed above).
