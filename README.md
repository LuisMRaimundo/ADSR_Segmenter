# ADSR_Segmenter

**Repository:** [github.com/LuisMRaimundo/ADSR_Segmenter](https://github.com/LuisMRaimundo/ADSR_Segmenter)

Desktop tool for **automatic ADSR segmentation** of monophonic or quasi-monophonic audio (orchestral one-shots, sample-library prep, spectral-analysis pipelines). Splits each file into **Attack**, **Sustain**, **Decay**, and **Release** regions with optional manual review, JSON/CSV metadata, and a boundary-error benchmark.

Related research tooling: [Intervallic_Homogeneity](https://github.com/LuisMRaimundo/Intervallic_Homogeneity).

---

## No Python installed? (one-click)

See **[installers/README.md](installers/README.md)**:

| Platform | Launcher |
|----------|----------|
| **Windows 10/11** | Double-click `installers\windows\Install and Run.bat` |
| **macOS** | Double-click `installers/macos/Install and Run.command` (after `chmod +x`) |
| **Linux** | `./installers/linux/install-and-run.sh` |

First run downloads a private Python and libraries (~150–250 MB), then opens the **ADSR_Segmenter** graphical interface. No system Python, pip, or conda required.

---

## Developers (Python already installed)

**Windows:** double-click **`run.bat`** in the project folder (runs `split_audio_segments.py`).

Requires **Python ≥ 3.10**. From the repository root:

```bash
pip install -e ".[dev]"
python split_audio_segments.py          # GUI
python split_audio_cli.py -f ./samples -o ./adsr_out --export-metadata
python run_benchmark.py --generate-corpus && python run_benchmark.py
pytest
```

Entry points after install: `adsr-segmenter-gui`, `adsr-segmenter-cli`, `adsr-segmenter-benchmark`.

`run.bat` (Windows) prefers `py -3`, then `python`, and launches `split_audio_segments.py` from the folder that contains the batch file. The current working directory is not used to find the script.

**MP3/M4A:** install [ffmpeg](https://ffmpeg.org/) on your PATH. Compressed formats are optional; `.wav` is the recommended scientific input.

---

## What it does

| Output folder | Content |
|---------------|---------|
| `_Attacks/` | Onset → attack boundary |
| `_Sustains/` | Attack → decay boundary |
| `_Sustains_Stable/` | Flux- and/or half-integer-stable sustain (only when regime refine is `trim`) |
| `_Decays/` | Decay → end of active sound |
| `_Release_Silence/` | Tail after active energy |
| `_Full_Active_Sound/` | Full trimmed active region |

Detection modes: **smart** (energy + proportional anchors, default), **advanced** (spectral flux + derivatives), **proportional**. Pitch refinement: **expand** (default), **annotate** (full sustain for STFT + metadata), **crop** (tight stable window). Spectral-regime refinement: **annotate** (default, metadata only), **trim** (also writes `_Sustains_Stable/`), or **off**. Regime flux is level-normalised; half-integer bands are relative to \(f_0\). Optional `--flux-sidecar` writes `<stem>.flux.json` on the sustain frame grid.

These four folder names are **operational energy/pitch regions**, not uniquely determined physical ADSR instants. The detector always emits attack / sustain / decay / release intervals (clamped to minimum durations) even when a recording has no synthesizer-style sustain. Metadata `decay_start` / \(t_{\mathrm{dec}}\) is an energy-threshold offset after the peak, not synthesizer decay-to-sustain. Digital silence is rejected per file (`no_active_energy`); other files in the same batch still complete. The GUI writes beside the source folder; CLI `--output` selects another directory.

**Load:** `librosa.load(path, sr=None)` keeps the file’s sample rate. The library default `mono=True` is not overridden, so stereo is downmixed. Filename note tokens must not be preceded by a letter (`Violin_A4` and `B#4` match; `Bagpipe1` / `seg1` do not).

**CLI exit codes:** `0` all files succeeded; `1` folder missing or no audio files; `2` at least one file rejected or failed. See [docs/ADSR_Segmenter_math_formula.md](docs/ADSR_Segmenter_math_formula.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_GUIDE.md](QUICK_GUIDE.md) | Non-specialist workflow |
| [run.bat](run.bat) | Windows launcher (Python already installed) |
| [docs/TECHNICAL_MANUAL.md](docs/TECHNICAL_MANUAL.md) | Full DSP specification, API, tutorials |
| [docs/ADSR_Segmenter_math_formula.md](docs/ADSR_Segmenter_math_formula.md) | Implementation-faithful formula reference |
| [docs/REGIME_REFINE_NOTES.md](docs/REGIME_REFINE_NOTES.md) | Spectral-regime defaults and rationale |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [installers/README.md](installers/README.md) | Autonomous installers (Windows / macOS / Linux) |
| [# Copyright and Use Notice.md](#%20Copyright%20and%20Use%20Notice.md) | Proprietary terms |
| [docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md) | Funding and thanks |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Tests and CI for authorised contributors |

---

## Benchmark

Reproducible synthetic corpus (40 labeled one-shots plus 4 regime items) and per-mode/per-preset mean boundary error (ms):

```bash
python run_benchmark.py --generate-corpus
python run_benchmark.py
```

Reports: `benchmark/results/benchmark_report.txt`. Label your own recordings with `python run_benchmark.py --template my_labels.csv`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

GitHub Actions runs `pytest` on push (see `.github/workflows/ci.yml`). Iowa trombone AIFF fixtures under `tests/fixtures/` are optional; those tests skip when the file is absent.

Declared dependencies are `librosa>=0.10.0` and the rest of `requirements.txt` / `pyproject.toml`. There is **no published lockfile**. A developer virtual environment that reuses system-site packages is **not** a fully isolated install.

**Validation evidence recorded for the merged implementation `8cfce8d` (2026-09-17), not re-run in this documentation pass:**

- Reused venv (`include-system-site-packages = true`): Python 3.10.11, librosa 0.10.2.post1 — 53 passed, 2 skipped (Iowa AIFF absent).
- Isolated venv (`include-system-site-packages = false`): Python 3.10.11, librosa 0.11.0 — same declared pins; YIN may report `tracking_failed` instead of `unvoiced` on noise.
- GitHub Actions `pytest` on Python 3.10 and 3.11 for PR #5.
- Synthetic GUI Run-button check and CLI mixed/all-silent batches (external fixtures). The research corpus and in-repository benchmark were not re-run.

This documentation update verified CLI `--help` / benchmark `--help` and source excerpts against that baseline. It does not claim a new full-suite or corpus run.

---

## Copyright and use

Copyright © 2026 Luís Raimundo. All rights reserved.

This repository and its contents are proprietary research material. **No open-source licence is granted.** No permission to copy, redistribute, modify, publish, or derive works without prior written permission from the copyright holder.

**Contact:** lmr.2020@outlook.pt

---

## Acknowledgements

This project was developed by **Luís Raimundo** with the support and funding of the **Fundação para a Ciência e a Tecnologia (FCT)** and **Universidade NOVA de Lisboa**.

**Funding DOI:** [https://doi.org/10.54499/2020.08817.BD](https://doi.org/10.54499/2020.08817.BD)

The author also gratefully acknowledges **Isabel Pires** for her support throughout the development of this work.
