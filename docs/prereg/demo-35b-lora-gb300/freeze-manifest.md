# 35B demo GB300 freeze manifest (prepared 2026-10-31, NOT a freeze)

Demo, not a scientific claim. Prepared for the L1 freeze of
`docs/prereg/demo-35b-lora-gb300/`; it builds on the NH-81 GB300 files
(`docs/prereg/nh81-gb300/freeze-manifest.md`). **No tag, no Software
Heritage save yet:** the freeze waits for the PI review of the resolved
requirements. At freeze time: merge on green CI after the NH-81 PR,
recompute this list on the merge commit, record the SHA-256 of this
file in the annotated tag message `demo-35b-gb300-freeze-YYYYMMDD`,
then save to Software Heritage and record the revision SWHID.

Weights: `sources-sha256.txt` (base `Qwen/Qwen3.6-35B-A3B@995ad96e`,
adapters `hypneum-lab/micro-kiki-v4-sota@9add98c2`). Prompts:
`prompts.json`.

## Files (SHA-256)

```
7934192b467e1f182b371ce70c07cbfbce6e757a456a837a35c641a7e8d5deb6  docs/prereg/demo-35b-lora-gb300/2026-10-31-demo-35b-lora-gb300.md
45392249a696c1be953d2362a519bae121d97fc0c8d801c009be12a3c400ec81  docs/prereg/demo-35b-lora-gb300/prompts.json
cf94740fbafec15be40cc812fb7d6996b9fc4bc2dee867a10164c3dc39f2cb10  docs/prereg/demo-35b-lora-gb300/sources-sha256.txt
1b3d108ddb9c62649aaaa62176cac6ecfc4c13ed34f394bb4ae53eddd4099652  docs/prereg/demo-35b-lora-gb300/requirements-gb300-demo.in
c4692382f9f26337d14c7a15c5452c8e36c9050f7a03e536fa8888e188c445ef  docs/prereg/demo-35b-lora-gb300/requirements-gb300-demo.sha256
0dbb54cca1c06426020cb4d42f62a1f8e4a428102d1af0ea9d61b5437249b7ef  docs/prereg/demo-35b-lora-gb300/requirements-gb300-demo.txt
1793e3194594e3e0f9ffe3448a013927a3c4aa9d4c0059b37c1a7501c3fe1973  docs/prereg/demo-35b-lora-gb300/requirements-review.md
dd24811dbb59da924e20fc4d148f09cde4648f3c7f1731cc5e93b1a0dcf4a324  experiments/demo_35b_gb300/demo_launcher.sh
28d42787ed1c8ddcb5a741d4e807ad3259fca4007518c5ee6ac2ebb0dfb7986a  experiments/demo_35b_gb300/demo_rules.py
0bb6e8780e0753cdf522533ebafe2965102925802a70b70ff8b2e456a73b9676  experiments/demo_35b_gb300/run_demo.py
0df7903e71c71f5b9b8ca7a340fa2bbe1175a704720c85c94101152c788f0b41  tests/test_demo_35b_gb300.py
```
