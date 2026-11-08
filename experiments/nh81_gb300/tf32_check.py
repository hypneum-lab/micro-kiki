"""TF32 check of KXKM r2 section 2, repeated on the GB300 (not a measurement).

Relative error of a 512x1536 @ 1536x384 float32 product against float64
numpy, under the current MLX_ENABLE_TF32 setting. Run twice by the
launcher (flag 0 and 1); both values are logged.
"""

import json
import os

import mlx.core as mx
import numpy as np

rng = np.random.default_rng(0)
a = rng.standard_normal((512, 1536)).astype(np.float32)
b = rng.standard_normal((1536, 384)).astype(np.float32)
ref = a.astype(np.float64) @ b.astype(np.float64)
got = np.array(mx.array(a) @ mx.array(b)).astype(np.float64)
err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
print(json.dumps({"MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32", "default"),
                  "device": str(mx.default_device()), "rel_error": err}))
