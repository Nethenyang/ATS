# Temporally Constrained Audio Configuration Selection

Research code and recorded measurements accompanying **Temporally Constrained Audio Configuration Selection with Alignment-Aware Fidelity Evaluation**.

This repository implements an offline, reference-based decision: encode and decode a finite set of existing audio configurationss, evaluate their reconstructions on the full reference time grid, and select the smallest actual file satisfying global and active-frame tail fidelity requirements. 

[data sources](docs/DATA_SOURCES.md)

## What is included

- The actual encoder/decoder pipeline, delay alignment, fidelity metrics and finite-set selector.
- Deterministic acquisition/preprocessing scripts for official public releases.
- Source identifiers, original archive member names, selection rules and SHA-256 hashes.
- All **3,628 recorded candidate measurements**, **552 policy outcomes**, threshold/protocol sensitivity results, paired bootstrap summaries and per-source checkpoints.
- Recorded empirical figures and table rows, plus scripts that recreate record-derived outputs without downloading audio.
- Seven original evaluator/codec tests and three repository workflow tests.

## Evaluation units and candidate counts

| Source group | Independent/source units | Excerpt duration | Reference rate | Candidates per source | Total candidates |
|---|---:|---:|---:|---:|---:|
| LibriSpeech dev-clean | 40 distinct speakers | 6 s | 16 kHz | 27 | 1,080 |
| MUSDB18 sample test split | 50 tracks |              7 s | 44.1 kHz | 49 | 2,450 |
| Supplied competition speech | 1 reference | 6 s | 48 kHz | 49 | 49 |
| Supplied competition music | 1 reference | 10 s | 48 kHz | 49 | 49 |
| **Full recorded experiment** | **92 sources** | | | | **3,628** |
| **Public-only reproduction** | **90 public excerpts** | | | | **3,530** |



The recorded environment used Python **3.12.4** on Windows and the versions pinned in `requirements.txt`. PyAV supplies the tested codec interfaces; a separate FFmpeg executable is not required by these scripts.

From the repository directory:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
```

The supplied workflow tests were executed on Windows/Python 3.12. Other platforms may produce different codec sizes or floating-point results; the recorded environment and bitstream hashes identify the original run.

## Repository layout

```text
audio-temporal-selection/
  README.md / README.zh-CN.md
  CITATION.cff
  requirements.txt
  audio_pipeline.py             # codecs, resampling, alignment, metrics, selection
  download_data.py              # official acquisition and deterministic preprocessing
  run_experiments.py            # actual bitstreams and all policy/sensitivity records
  make_report.py                # record-derived tables, summaries and plots
  verify_results.py             # numerical checks without manuscript/audio dependencies
  draw_vector_schematics.py     # deterministic conceptual diagram variant
  draw_alignment_icons.py       # editable SVG alignment icons
  supplementary_denoising.py    # optional auxiliary analysis
  tests/                        # evaluator and repository integration tests
  configs/                      # documented protocol and enumerated candidates
  manifests/                    # source lists and archive provenance
  results/recorded/              # measured CSVs, summary, environment, checkpoints
  results/supplementary/         # auxiliary numerical records, no audio
  artifacts/recorded/            # figures and table rows already generated
  docs/                         # provenance, interpretation and verification notes
  data/                         # local inputs/cache only, ignored and not bundled
  runs/                         # new experiments/verification only, ignored
  artifacts/reproduced/          # local regenerated outputs, ignored
```
