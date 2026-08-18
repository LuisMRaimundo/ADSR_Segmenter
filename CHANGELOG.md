# Changelog

## 3.3.0 — 2026-08-18

Make the regime and pitch stages register- and instrument-independent.

- Half-integer bands are \(\pm 0.15\,f_0\); unresolvable or unpitched frames report `half_integer_valid=False`.
- Second inward walk on the HI ratio (10 dB above the note’s own mid-sustain); combined boundary is the inner of flux and HI.
- Regime flux is frame-energy-normalised; Advanced attack flux stays unnormalised.
- Vibrato-robust median on flux / HI tracks; YIN range and frame scale with the filename or expected note; tracking / unvoiced failures keep energy boundaries.
- Floor `n_fft` defaults to the pitch frame. Sidecar gains smoothed tracks and validity flags.
- See `docs/REGIME_REFINE_NOTES.md` and Technical Manual §8.

## 3.2.0 — 2026-08-18

Add an optional spectral-regime refinement stage after energy and pitch.

- New `SegmentConfig` fields and `refine_sustain_by_regime` (default mode `annotate`: metadata only).
- Preset `soft_high_brass` (trim mode) plus regime keys on every existing preset.
- Trim mode writes `_Sustains_Stable/` beside the unchanged `_Sustains/`.
- CLI: `--regime-mode`, `--regime-flux-ratio`, `--regime-analysis-n-fft`, `--flux-sidecar`.
- GUI: regime mode, flux ratio, analysis n_fft, sidecar; review table shows `refused`.
- Benchmark corpus appends four regime-labelled one-shots; existing 40-item lines unchanged.
- See `docs/REGIME_REFINE_NOTES.md` and Technical Manual §8.
