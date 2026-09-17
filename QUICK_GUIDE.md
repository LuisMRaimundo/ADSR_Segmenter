# Quick Guide — ADSR_Segmenter

For users who want to split instrument one-shots without reading the full technical manual.

## 1. Install and open

**Windows (Python already installed):** double-click **`run.bat`** in the project folder.

**No Python on this PC:** use the one-click installer ([installers/README.md](installers/README.md)).

**Manual install (Python 3.10+):**

```bash
pip install -e .
python split_audio_segments.py
```

## 2. Prepare files

- Put all audio files in **one folder** (`.wav` recommended; MP3 needs ffmpeg).
- Name files with pitch when possible, e.g. `Violin_A4_01.wav` (helps sustain detection). The note letters must not sit inside another word (`Bagpipe1` is not read as E1).

## 3. Run a batch split

1. **Browse** → select your folder.
2. Choose a **Preset** matching average note length (or **Auto-Detect Mean Length**).
3. Leave **Smart Mode** on for most orchestral material.
4. For **spectral analysis / STFT**, set **Pitch Refine** to **annotate** (keeps long sustains). Leave **Regime refine** on **annotate** unless you also want a flux-stable crop.
5. Click **► RUN OPTIMIZED SPLIT**.
6. Use **Review Segmentation** to drag attack (green) and decay (orange) lines if needed.

## 4. Outputs

The GUI writes **next to the source folder** (there is no separate output-folder control). Use the CLI `--output` flag to write elsewhere.

The four region folders are **operational energy/pitch cuts**, not unique physical ADSR instants. “Decay start” is when energy has fallen after the peak, not a synthesizer decay-to-sustain point. Silent files are skipped with a rejection message; other files in the same batch still run.

Next to your source files:

- `_Attacks/`, `_Sustains/`, `_Decays/`, `_Release_Silence/`, `_Full_Active_Sound/`
- `_Sustains_Stable/` when **Regime refine** is **trim** (soft high brass, half-integer onset)
- `segmentation_metadata.json` and `.csv` (plus optional `<stem>.flux.json` sidecar)

Spectral-regime refinement watches the spectrum after level and pitch have already settled. Flux is level-normalised (walk threshold 1.5); half-integer bands are \(\pm 0.15\,f_0\) and a second walk (10 dB above the note’s own middle) catches tails that flux misses. Default **annotate** only records the stable window; **trim** also exports `_Sustains_Stable/`. Use the **soft_high_brass** preset (Very Long profile, pitch σ = 8 ¢) for *pp* high brass.

## 5. Presets at a glance

| Preset | Typical use |
|--------|-------------|
| Very Short | Plucks, staccato |
| Short / Medium | Most single notes |
| Long / Very Long | Sustained bowed notes (5–7 s) |
| Legato / Bow | Long notes with vibrato |
| Staccato / Pluck | Short attacks, advanced detection |
| soft_high_brass | Soft high brass; regime **trim** + 8 ¢ pitch window |

## 6. CLI (write somewhere other than the source folder)

```bash
python split_audio_cli.py -f ./samples -o ./adsr_out --export-metadata
```

Exit `0` if every file succeeds, `2` if any file is rejected (for example digital silence) or fails. Valid files in a mixed batch still export. An all-silent folder still writes metadata when `--export-metadata` is set.

Short recordings still receive ordered operational attack / sustain / decay cuts. The 40-hop analysis floor is a preferred minimum that is clamped, not a reason to skip the file.

## 7. Need help?

See [docs/TECHNICAL_MANUAL.md](docs/TECHNICAL_MANUAL.md) §18 Troubleshooting, [docs/ADSR_Segmenter_math_formula.md](docs/ADSR_Segmenter_math_formula.md) for formulas, and [docs/REGIME_REFINE_NOTES.md](docs/REGIME_REFINE_NOTES.md) for regime defaults.

## Copyright

Copyright © 2026 Luís Raimundo. Proprietary research material — see [# Copyright and Use Notice.md](%23%20Copyright%20and%20Use%20Notice.md).
