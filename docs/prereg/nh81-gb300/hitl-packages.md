# NH-81 GB300: packages to hash and have reviewed by the PI

**Status 2026-10-31 (after the PI decisions):** step 1 is done (list
below validated by the PI); step 2, the resolved and hashed set, is in
`requirements-gb300-nh81.txt` (fallback `requirements-gb300-nh81-cuda13.txt`),
with the readable summary `requirements-review.md` and the file hashes
in `requirements-gb300-nh81.sha256`. The PI approved it on
2026-10-31 at 22:48 CET (second review). CUDA headers: `nvidia-cuda-runtime-cu12`
and `nvidia-cuda-cccl-cu12` 12.9 wheels (PI decision 8). uv: 0.13.5.

Repository policy: no blind upgrade, every third-party artefact pinned
and reviewed by a human before use. **No lockfile is generated before
this list is reviewed.** Two review steps, both before any access,
installation or freeze:

1. The PI reviews this list (names, versions, constraints).
2. The set is then resolved for `aarch64` / CPython 3.12 (transitive
   NVIDIA wheels included, and the CUDA-13 fallback set if item 4 is
   accepted); the resolved artefacts, with their SHA-256, are reviewed
   by the PI a second time. Only then is the result committed as
   `docs/prereg/nh81-gb300/requirements-gb300.lock` with
   `--require-hashes` entries and covered by the freeze manifest.

Reviewing names is not reviewing artefacts: every row marked "to
record" must carry a reviewed hash before the freeze.

Hashes below were read from the PyPI JSON API on 2026-10-31 (files not
downloaded). They are re-checked on the GB300 before installation; a
mismatch stops the installation.

| Artefact | Version / file | SHA-256 (PyPI) | Status |
|---|---|---|---|
| uv | exact version TBD by PI (item 8), `manylinux` aarch64 build | to record | to review |
| CPython | 3.12.x, uv-managed aarch64 build (TBD by PI item 8) | to record | to review |
| mlx | 0.31.1, `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | `1ed1991c8e39f841d5756c0c543beb819763a2f80fba3f4b150bc6cad4d973de` | to review |
| mlx-cuda-12 | 0.31.1, `mlx_cuda_12-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | `011d611b5b31a948a5eef5ea78a5b3fba41908817d2f094d9bd125be4abae013` | to review |
| nvidia-cublas-cu12 | `12.9.*` (required by mlx-cuda-12), exact version at resolution | to record | to review |
| nvidia-cuda-nvrtc-cu12 | `12.9.*` (required by mlx-cuda-12; NVRTC must know `sm_103`) | to record | to review |
| nvidia-cudnn-cu12 | `9.*` (required by mlx-cuda-12) | to record | to review |
| nvidia-nccl-cu12 | unpinned by mlx-cuda-12, exact version at resolution | to record | to review |
| numpy | 2.2.6 (repository `uv.lock` version), `numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87` | to review |
| CUDA headers for `CUDA_HOME` | from the host's CUDA 13.1 / 13.2 toolkit or the `nvidia-cuda-*` wheels; source TBD | to record | to review |

Fallback only if TBD-by-PI item 4 is accepted and the CUDA-12 smoke
test fails:

| Artefact | Version / file | SHA-256 (PyPI) | Status |
|---|---|---|---|
| mlx-cuda-13 | 0.31.1, `mlx_cuda_13-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | `fcbc4ac3764b5bfac7814f7fd8ad6369d9b5d1279d0b3b3bc62878b572e8580a` | to review |
| nvidia-cublas, nvidia-cuda-nvrtc, nvidia-cudnn-cu13 `9.*`, nvidia-nccl-cu13 | required by mlx-cuda-13, exact versions at resolution | to record | to review |

Not installed: `tokenizers` (prompt ids are committed), `pynvml` /
`nvidia-ml-py` (NVML through `ctypes`), `torch`.

Data files (SHA-256 already fixed in the prereg, section 1): the two
checkpoints, `test.i32`, `val.i32`; plus `prompts_ids.json` and the
code tree at tag `prereg-nh81-m3pro-20261030` with the GB300 delta.

Driver-side items recorded, not installed: NVIDIA driver version,
`libnvidia-ml.so.1` path and its SHA-256, `nvidia-smi` version.
