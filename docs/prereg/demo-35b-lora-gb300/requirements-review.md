# DEMO GB300: resolved requirements for PI review

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

## `requirements-gb300-demo.txt`

Input (`requirements-gb300-demo.in`):

    mlx==0.31.1
    mlx-cuda-12==0.31.1
    numpy==2.2.6
    nvidia-cuda-runtime-cu12==12.9.*
    nvidia-cuda-cccl-cu12==12.9.*
    mlx-lm==0.31.2

| Package | Version | Uploaded | Linux aarch64 file | SHA-256 | in file |
|---|---|---|---|---|---|
| annotated-doc | 0.0.5 | 2026-07-28 | `annotated_doc-0.0.5-py3-none-any.whl` | `117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101` | yes |
| anyio | 4.15.1 | 2026-09-05 | `anyio-4.15.1-py3-none-any.whl` | `6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101` | yes |
| click | 8.5.0 | 2026-08-26 | `click-8.5.0-py3-none-any.whl` | `255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360` | yes |
| filelock | 4.2.0 | 2026-10-14 | `filelock-4.2.0-py3-none-any.whl` | `2ff5690882e8cdb00ef31fb3d01a3094c29f30985426c59495afb1733f3b7238` | yes |
| fsspec | 2026.9.0 | 2026-09-18 | `fsspec-2026.9.0-py3-none-any.whl` | `8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f` | yes |
| h11 | 0.16.0 | 2025-04-24 | `h11-0.16.0-py3-none-any.whl` | `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86` | yes |
| hf-xet | 1.7.0 | 2026-10-06 | `hf_xet-1.7.0-cp38-abi3-manylinux_2_28_aarch64.whl` | `fcfd6c22418e57dd5b3aea649e813b2e2cfb2aebf317b210d90f1fe4b3018b52` | yes |
| httpcore2 | 2.13.1 | 2026-09-23 | `httpcore2-2.13.1-py3-none-any.whl` | `e1e05d4f25f7d7d496bfb96748f6f4b67657b03da069b3a68c36069f3db73d0a` | yes |
| httpx2 | 2.13.1 | 2026-09-23 | `httpx2-2.13.1-py3-none-any.whl` | `6dff50fabc270ee5fd25d845d0b078ed20564579744d6d962850975996d2f9a4` | yes |
| huggingface-hub | 2.2.0 | 2026-10-08 | `huggingface_hub-2.2.0-py3-none-any.whl` | `1667f145dc56dc210d60966069397df9ecfca9607a5d43db88b308c89dae56b3` | yes |
| idna | 3.20 | 2026-09-17 | `idna-3.20-py3-none-any.whl` | `ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c` | yes |
| jinja2 | 3.1.6 | 2025-03-05 | `jinja2-3.1.6-py3-none-any.whl` | `85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67` | yes |
| markdown-it-py | 4.2.0 | 2026-05-07 | `markdown_it_py-4.2.0-py3-none-any.whl` | `9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a` | yes |
| markupsafe | 3.0.4 | 2026-10-02 | `markupsafe-3.0.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | `6a45c3d514f2436064db00d7fc8778d888f0236ebfed649b53d13a59e69ad51b` | yes |
| mdurl | 0.1.2 | 2022-08-14 | `mdurl-0.1.2-py3-none-any.whl` | `84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8` | yes |
| mlx | 0.31.1 | 2026-03-12 | `mlx-0.31.1-cp312-cp312-manylinux_2_35_aarch64.whl` | `1ed1991c8e39f841d5756c0c543beb819763a2f80fba3f4b150bc6cad4d973de` | yes |
| mlx-cuda-12 | 0.31.1 | 2026-03-12 | `mlx_cuda_12-0.31.1-py3-none-manylinux_2_35_aarch64.whl` | `011d611b5b31a948a5eef5ea78a5b3fba41908817d2f094d9bd125be4abae013` | yes |
| mlx-lm | 0.31.2 | 2026-04-07 | `mlx_lm-0.31.2-py3-none-any.whl` | `a90a7905031b2580b4923f92e0fb156ca287f452bb1a9e41a60dcbc971532d46` | yes |
| numpy | 2.2.6 | 2025-05-17 | `numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87` | yes |
| nvidia-cublas-cu12 | 12.9.2.10 | 2026-04-08 | `nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_aarch64.whl` | `5572131a59c3eebeeb1c4c8144f772d49372c20124916e072a0e3fc30df421d5` | yes |
| nvidia-cuda-cccl-cu12 | 12.9.27 | 2025-05-01 | `nvidia_cuda_cccl_cu12-12.9.27-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `d7898b38aa68beaa234d48f0868273702342a196d6e2e9d0ef058dca2390ebea` | yes |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | 2025-06-05 | `nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `096d4de6bda726415dfaf3198d4f5c522b8e70139c97feef5cd2ca6d4cd9cead` | yes |
| nvidia-cuda-runtime-cu12 | 12.9.79 | 2025-06-05 | `nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | `83469a846206f2a733db0c42e223589ab62fd2fabac4432d2f8802de4bded0a4` | yes |
| nvidia-cudnn-cu12 | 9.27.0.42 | 2026-09-29 | `nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_aarch64.whl` | `5bc22243563107ea06aa60f9116ad800063d5f4617cc75bc6670d5db78d9f336` | yes |
| nvidia-nccl-cu12 | 2.32.3 | 2026-09-22 | `nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_aarch64.whl` | `061af42ae1044816820e16b872075e243d1ac0656f51860fbc8e106c257aa83a` | yes |
| packaging | 26.3 | 2026-08-04 | `packaging-26.3-py3-none-any.whl` | `d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c` | yes |
| protobuf | 7.36.2 | 2026-09-17 | `protobuf-7.36.2-cp310-abi3-manylinux2014_aarch64.whl` | `e11e1f0180583a2af89db6a2ecd9e8dc40aa6d2988ca175bfd0e6d12ea72d74e` | yes |
| pygments | 2.21.0 | 2026-08-17 | `pygments-2.21.0-py3-none-any.whl` | `2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9` | yes |
| pyyaml | 6.0.3 | 2025-09-29 | `pyyaml-6.0.3-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | `9149cad251584d5fb4981be1ecde53a1ca46c891a79788c0df828d2f166bda28` | yes |
| regex | 2026.9.29 | 2026-09-29 | `regex-2026.9.29-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | `f93bc1c3486ef3747e07c9d7c1d0a147b8fbaab975f80e348aed6f71309dfaca` | yes |
| rich | 15.0.0 | 2026-04-12 | `rich-15.0.0-py3-none-any.whl` | `33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb` | yes |
| safetensors | 0.8.0 | 2026-06-09 | `safetensors-0.8.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `7a46e5ff292c356d6991e60942ba7f79817682d3a2cef0702136448cb9c4d235` | yes |
| sentencepiece | 0.2.2 | 2026-07-12 | `sentencepiece-0.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | `cbce24284f51f71d10a42b7b9c964dcb9048b28f1c8e5db40bcbcb6f428cba6a` | yes |
| shellingham | 1.5.4 | 2023-10-24 | `shellingham-1.5.4-py2.py3-none-any.whl` | `7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686` | yes |
| tokenizers | 0.23.3 | 2026-10-09 | `tokenizers-0.23.3-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `ec82e80e65a862275b97c3d90b7a523df8d9519ee48aeb4e9625b2cc909274e0` | yes |
| tqdm | 4.70.1 | 2026-09-11 | `tqdm-4.70.1-py3-none-any.whl` | `c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73` | yes |
| transformers | 5.20.1 | 2026-10-16 | `transformers-5.20.1-py3-none-any.whl` | `5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19` | yes |
| truststore | 0.10.5 | 2026-10-12 | `truststore-0.10.5-py3-none-any.whl` | `9aaaedaefaf06d8b206278cf8b5012bc897f485a874503501e12d776df78951c` | yes |
| typer | 0.27.3 | 2026-10-06 | `typer-0.27.3-py3-none-any.whl` | `e50022f28b82a86313e54501317a1db64bf8f8d036ff8cfe5ca7e47675454aff` | yes |
| typing-extensions | 4.16.0 | 2026-07-02 | `typing_extensions-4.16.0-py3-none-any.whl` | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` | yes |

SHA-256 of the requirement files: `requirements-gb300-demo.sha256`.

## PI approval

**Approved by the PI on 2026-10-31 at 22:48 CET** (HITL review of the
resolved set `requirements-gb300-demo.txt` listed above). The PI
checked that `httpx2` and `httpcore2` are the legitimate packages of
Tom Christie / pydantic, required by `huggingface-hub` 2.2.0.
