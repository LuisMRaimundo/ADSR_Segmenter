# Spectral-regime refinement — parameter notes

These defaults are the citable settings for the third sustain stage
(`refine_sustain_by_regime`). Energy and pitch stages are unchanged.

## Defaults

| Parameter | Default | Why |
|-----------|---------|-----|
| `use_regime_refine` | `True` | Always compute the stationarity window so metadata is available. Export is unchanged unless mode is `trim`. |
| `regime_refine_mode` | `"annotate"` | Metadata-only. Downstream STFT pipelines keep the full pitch-stable sustain; nothing is cropped by default. |
| `regime_flux_ratio` | `2.0` | On the Iowa tenor trombone *pp* C5 reference, median flux is ~0.39 in 0–0.3 s and ~0.14 in the stable middle (ratio ≈ 2.8). A factor of two clears that onset regime without eating ordinary vibrato or bow noise. |
| `regime_flux_median_frames` | `9` (odd) | ~100 ms at hop 512 / 44.1 kHz. Long enough to ignore a single noisy frame, short enough not to smear a 100 ms interior burst into the edges. |
| `regime_reference_fraction` | `0.5` | Central half of the pitch-stable sustain is the “already settled” region. Avoids contaminating the reference with the very edges being tested. |
| `regime_min_windows` | `20` | Floor in non-overlapping analysis frames: `20 × n_fft / sr`. At `n_fft=8192`, 44.1 kHz this is ~3.7 s; at the core default 1024 it is ~0.46 s and is dominated by `regime_min_duration`. |
| `regime_min_duration` | `1.0` s | Hard time floor so a trim can never leave less than one second — the practical minimum for a useful long-window STFT. |
| `regime_analysis_n_fft` | `None` → `cfg.frame_length` | Downstream STFT size. Set this to the analyser FFT (e.g. 8192 / 16384) so the floor is expressed in *that* tool’s windows, not the segmenter’s 1024-point hop. |
| `regime_half_integer` | `True` | Diagnostic only. Reports 1.5·f₀ + 2.5·f₀ energy vs f₀ (±60 Hz / ±15 Hz). The trombone case is −20 dB at the edges and −50 dB in the middle; the ratio is not used to place the cut. |

## Floor rule

```
floor_seconds = max(regime_min_duration,
                    regime_min_windows * (regime_analysis_n_fft or frame_length) / sr)
```

If the pitch-stable span is already shorter than the floor, or a candidate trim would go below it, the stage sets `refused=True`, `refused_reason='span_below_floor'`, and keeps the incoming boundaries.

## When to use which mode

- **annotate** (default) — keep `_Sustains/` as energy + pitch; record `window_start` / `window_end` and optional `<stem>.flux.json` for weighting frames.
- **trim** — also write `_Sustains_Stable/` from the flux-stable window. Use for sampler cores that must not include the half-integer onset of soft high brass.
- **off** — `use_regime_refine=False`; `regime_refine == {}`.

Preset `soft_high_brass` is the Very Long sustained-tone profile with `regime_refine_mode="trim"`, `regime_flux_ratio=2.0`, `pitch_stability_cents=8.0`.
