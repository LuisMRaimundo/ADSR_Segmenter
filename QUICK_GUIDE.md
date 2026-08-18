# Quick Guide — ADSR_Segmenter

For users who want to split instrument one-shots without reading the full technical manual.

## 1. Install and open

**Windows (Python already installed):** double-click **`run.bat`** in the project folder.

**No Python on this PC:** use the one-click installer ([installers/README.md](../installers/README.md)).

**Manual install (Python 3.10+):**

```bash
pip install -e .
python split_audio_segments.py
```

## 2. Prepare files

- Put all audio files in **one folder** (`.wav` recommended; MP3 needs ffmpeg).
- Name files with pitch when possible, e.g. `Violin_A4_01.wav` (helps sustain detection).

## 3. Run a batch split

1. **Browse** → select your folder.
2. Choose a **Preset** matching average note length (or **Auto-Detect Mean Length**).
3. Leave **Smart Mode** on for most orchestral material.
4. For **spectral analysis / STFT**, set **Pitch Refine** to **annotate** (keeps long sustains). Leave **Regime refine** on **annotate** unless you also want a flux-stable crop.
5. Click **► RUN OPTIMIZED SPLIT**.
6. Use **Review Segmentation** to drag attack (green) and decay (orange) lines if needed.

## 4. Outputs

Next to your source files:

- `_Attacks/`, `_Sustains/`, `_Decays/`, `_Release_Silence/`, `_Full_Active_Sound/`
- `_Sustains_Stable/` when **Regime refine** is **trim** (soft high brass, half-integer onset)
- `segmentation_metadata.json` and `.csv` (plus optional `<stem>.flux.json` sidecar)

Spectral-regime refinement watches the spectrum after level and pitch have already settled. Flux is level-normalised; half-integer bands are \(\pm 0.15\,f_0\) and a second walk (10 dB above the note’s own middle) catches tails that flux misses. Default **annotate** only records the stable window; **trim** also exports `_Sustains_Stable/`. Use the **soft_high_brass** preset (Very Long profile, flux ratio 2.0, pitch σ = 8 ¢) for *pp* high brass.

## 5. Presets at a glance

| Preset | Typical use |
|--------|-------------|
| Very Short | Plucks, staccato |
| Short / Medium | Most single notes |
| Long / Very Long | Sustained bowed notes (5–7 s) |
| Legato / Bow | Long notes with vibrato |
| Staccato / Pluck | Short attacks, advanced detection |
| soft_high_brass | Soft high brass; regime **trim** + 8 ¢ pitch window |

## 6. Need help?

See [docs/TECHNICAL_MANUAL.md](docs/TECHNICAL_MANUAL.md) §18 Troubleshooting and [docs/REGIME_REFINE_NOTES.md](docs/REGIME_REFINE_NOTES.md) for regime defaults.

## Copyright

Copyright © 2026 Luís Raimundo. Proprietary research material — see [# Copyright and Use Notice.md](../#%20Copyright%20and%20Use%20Notice.md).
