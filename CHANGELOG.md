# Changelog

## 3.2.0 — 2026-08-18

Add an optional spectral-regime refinement stage after energy and pitch.

- New `SegmentConfig` fields and `refine_sustain_by_regime` (default mode `annotate`: metadata only).
- Preset `soft_high_brass` (trim mode) plus regime keys on every existing preset.
- Trim mode writes `_Sustains_Stable/` beside the unchanged `_Sustains/`.
- CLI: `--regime-mode`, `--regime-flux-ratio`, `--regime-analysis-n-fft`, `--flux-sidecar`.
- GUI: regime mode, flux ratio, analysis n_fft, sidecar; review table shows `refused`.
- Benchmark corpus appends four regime-labelled one-shots; existing 40-item lines unchanged.
- See `docs/REGIME_REFINE_NOTES.md` and Technical Manual §8.
