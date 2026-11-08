# Zenodo L2 deposit — micro-kiki pre-registrations (2026-11-08)

This file records the Level-2 (verifiable deposit date) for the frozen
pre-registrations of the micro-kiki repository. On 2026-11-08 the PI
(ORCID 0000-0002-8414-185X) published a restricted Zenodo record
([https://zenodo.org/records/24053794](https://zenodo.org/records/24053794))
holding the five freeze zips of micro-kiki. Metadata are public;
files are restricted (authenticated users may request access). The
record is released under CC-BY-4.0.

**Version DOI:** [10.5281/zenodo.24053794](https://doi.org/10.5281/zenodo.24053794)
(cite this one for a fixed version).
**Concept DOI:** [10.5281/zenodo.24053793](https://doi.org/10.5281/zenodo.24053793)
(resolves to the latest version).
**Published:** 2026-11-08. **PI act:** explicit "Publish" on Zenodo.

The L1 / L2 definition and the protocol for depositing pre-registrations
are in `hypneum-lab/.github` `docs/PROJECT-MANAGEMENT.md` section 4
(also referenced as `hypneum-lab/.github#5`).

## Studies in this deposit

| Study | Tag | Commit | SWHID | Zip | SHA-256 | Documents in the zip |
|---|---|---|---|---|---|---|
| NH-81 joules per token | `prereg-nh81-20261025` | `52461ee321a1c9bdd27ff2433cbcbd4d3dbe932f` | `swh:1:rev:52461ee321a1c9bdd27ff2433cbcbd4d3dbe932f` | `prereg-nh81-20261025.zip` | `f24ae5984896fcd02a9008a92ea7ac90219ecf0932ac4e3aeb362eedf022e676` | `docs/prereg/2026-10-25-nh81-joules-per-token.md`, `docs/prereg/2026-10-28-nh81-amendment-1.md` (absent at the base tag; extracted from tag `prereg-nh81-kxkm-20261028`, commit `e85cdb8c192b9388f39654d06c02b615ca2f9c7c`, and packed in this zip) |
| NH-81 KXKM replication r1–r4 | `prereg-nh81-kxkm-20261030-r4` | `979771f0cf1f061ad906d8e420bad0664da9c5a0` | `swh:1:rev:979771f0cf1f061ad906d8e420bad0664da9c5a0` | `prereg-nh81-kxkm-20261030-r4.zip` | `09efab1c216925883dc927024950f7c8917cb3820f1d74a1c69669cef843679b` | `docs/prereg/2026-10-28-nh81-kxkm-replication.md`, `docs/prereg/2026-10-29-nh81-kxkm-replication-r3.md`, `docs/prereg/2026-10-30-nh81-kxkm-amendment-r4.md` |
| NH-81 M3 Pro replication | `prereg-nh81-m3pro-20261030` | `4b4dafd9632b1bf8e7210f6ba84e5ae420730042` | `swh:1:rev:4b4dafd9632b1bf8e7210f6ba84e5ae420730042` | `prereg-nh81-m3pro-20261030.zip` | `6df4b866ed534276e4d1c6058088d49f7630c7102df018f2f5d9647dd463311d` | `docs/prereg/2026-10-30-nh81-m3pro-replication.md` |
| NH-81 GB300 descriptive | `nh81-gb300-freeze-20261031b` | `185c5d44b50de76a73c9a2a45ef56c6e28fbcedc` | `swh:1:rev:185c5d44b50de76a73c9a2a45ef56c6e28fbcedc` | `nh81-gb300-freeze-20261031b.zip` | `095353244acaa8e70d33c3132cbc8740d67c979df4a27e8bf8b2e76a68427724` | `docs/prereg/nh81-gb300/2026-10-31-nh81-gb300-descriptive.md`, `docs/prereg/nh81-gb300/amendment-2026-10-31.md`, `docs/prereg/nh81-gb300/freeze-manifest.md`, `docs/prereg/nh81-gb300/hitl-packages.md`, `docs/prereg/nh81-gb300/requirements-review.md` |
| Demo 35B LoRA GB300 | `demo-35b-gb300-freeze-20261031b` | `185c5d44b50de76a73c9a2a45ef56c6e28fbcedc` | `swh:1:rev:185c5d44b50de76a73c9a2a45ef56c6e28fbcedc` | `demo-35b-gb300-freeze-20261031b.zip` | `e1143bdfcdd23f14bd2e62c83f7e246af26aeb455c20580c203b8f2f55a31ad7` | `docs/prereg/demo-35b-lora-gb300/2026-10-31-demo-35b-lora-gb300.md`, `docs/prereg/demo-35b-lora-gb300/amendment-2026-10-31.md`, `docs/prereg/demo-35b-lora-gb300/freeze-manifest.md`, `docs/prereg/demo-35b-lora-gb300/requirements-review.md` |

Note: tags `nh81-gb300-freeze-20261031b` and `demo-35b-gb300-freeze-20261031b`
point to the same merge commit (PR #32); they are distinct annotated tags
covering different document sets.

## Files not in the deposit (created after the tag)

The following files exist on the default branch but were created after
their respective freeze tag and are therefore absent from the zip.

| Tag | Path |
|---|---|
| `nh81-gb300-freeze-20261031b` | `docs/prereg/nh81-gb300/freeze-record.md` |
| `demo-35b-gb300-freeze-20261031b` | `docs/prereg/demo-35b-lora-gb300/freeze-record.md` |
