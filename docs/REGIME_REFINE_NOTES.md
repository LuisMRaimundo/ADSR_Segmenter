# Spectral-regime refinement — parameter notes

These defaults are the citable settings for the third sustain stage
(`refine_sustain_by_regime`) and the pitch-stage guards it depends on.
Energy export (`_Sustains/`) and the Advanced-mode attack detector are
unchanged from v3.1 / v3.2.

## Defaults (v3.2, kept)

| Parameter | Default | Why |
|-----------|---------|-----|
| `use_regime_refine` | `True` | Always compute the stationarity window so metadata is available. Export is unchanged unless mode is `trim`. |
| `regime_refine_mode` | `"annotate"` | Metadata-only. Downstream STFT pipelines keep the full pitch-stable sustain; nothing is cropped by default. |
| `regime_flux_ratio` | `2.0` | On the Iowa tenor trombone *pp* C5 reference, median flux is ~0.39 in 0–0.3 s and ~0.14 in the stable middle (ratio ≈ 2.8). A factor of two clears that onset regime without eating ordinary vibrato or bow noise. The trim decision stays a **ratio**, now applied to level-normalised flux. |
| `regime_flux_median_frames` | `9` (odd) | ~100 ms at hop 512 / 44.1 kHz. Long enough to ignore a single noisy frame, short enough not to smear a 100 ms interior burst into the edges. |
| `regime_reference_fraction` | `0.5` | Central half of the pitch-stable sustain is the “already settled” region. Avoids contaminating the reference with the very edges being tested. Shared by the flux walk and the half-integer walk. |
| `regime_min_windows` | `20` | Floor in non-overlapping analysis frames: `20 × n_fft / sr`. At `n_fft=8192`, 44.1 kHz this is ~3.7 s; at the core default 1024 it is ~0.46 s and is dominated by `regime_min_duration`. |
| `regime_min_duration` | `1.0` s | Hard time floor so a trim can never leave less than one second — the practical minimum for a useful long-window STFT. Unchanged in v3.3. |
| `regime_half_integer` | `True` | Compute the 1.5·f₀ + 2.5·f₀ vs f₀ ratio (now also used as a second walk when valid). |

## New defaults (v3.3)

| Parameter | Default | Why |
|-----------|---------|-----|
| `regime_hi_rel_bandwidth` | `0.15` | Band = \(\pm\alpha f_0\) around \(f_0\), \(1.5 f_0\), and \(2.5 f_0\). Replaces the ±60 / ±15 Hz constants. At E2 (82 Hz) a ±60 Hz band around 1.5·f₀ spanned 63–183 Hz and contained H1 and H2 (the Iowa *pp* batch read +11 to +14 dB). \(\alpha=0.15\) keeps H1 out of the 1.5·f₀ band at every register where the band is resolvable. |
| `regime_use_half_integer` | `True` | Second inward walk. Iowa C5: flux end ratio ≤ 1.1 while HI rose −46.6 → −16.8 dB. Flux does not see a stationary half-integer tail; the HI walk does. |
| `regime_hi_rise_db` | `10` dB | Walk while `ratio − mid-sustain reference > 10 dB`. Relative to the note’s own middle, so a steady multiphonic or sul-ponticello (constant HI content) is not trimmed. 10 dB is below the C5 tail rise (~30 dB) and above ordinary measurement jitter. |
| `regime_vibrato_robust` | `True` | Same moving-median window as the pitch stage (`vibrato_median_window_s`, 0.12 s) applied to flux and HI **before** the reference and the walks. A 6 Hz vibrato (±20 ¢) + tremolo (±2 dB) must not be trimmed (`flux_edge_ratio_* ≤ 1.3`). |
| `regime_analysis_n_fft` | `None` → **pitch frame** | Was `frame_length` (always 1024). Now defaults to the YIN frame chosen for this note (`≥ 4·sr/fmin`, power of two, cap 8192) so `floor_windows` scales with register. Explicit config still wins (`analysis_n_fft_source ∈ {config, pitch_frame, frame_length}`). |
| `pitch_fmin` / `pitch_fmax` | `30` / `4200` Hz | Fallback YIN range when no filename note and no `expected_note_hz` are available (double bass to piccolo). Context, when present, is the parsed or expected note ±1 octave (exact \(f_0/2\) excluded). |
| YIN frame | `4·sr/fmin` | Power of two, capped at 8192, when a filename note, `expected_note_hz`, or an explicit search range is present. Unnamed files keep the v3.2 1024-sample grain so expand-mode benchmark rows do not move. Hop stays `hop_length`. An E2 window is four times a C6 window, so the low-register floor is not the C5 1024-sample grain. |
| `pitch_fail_cents` | `50` ¢ | If the best-window σ exceeds this, `failed=True`, `failed_reason="tracking_failed"`, energy boundaries kept, regime gets `f0_hz=None`. Iowa E2 *pp* had σ = 437 ¢ and a silent continue. |
| Octave / naming | `300` ¢ | If \(\lvert\mathrm{median}\,f_0 - f_{\mathrm{expected}}\rvert > 300\) ¢ → `octave_error` (Iowa E2 tracked 201 Hz against 82 Hz). Filename wrap spellings (`B#`, `Cb`, `E#`, `Fb`) are resolved (`B#4` = C5) and flagged; a >300 ¢ (or >50 ¢ on a wrap) disagreement also sets `note_name_mismatch`. |
| `pitch_min_voiced_fraction` | `0.5` | Fraction of frames within 200 ¢ of the median. Below this: `failed_reason="unvoiced"`, energy boundaries kept, regime on flux only. |

## Level-independent flux

`compute_spectral_flux(..., normalised=False)` keeps the v3.1 signature. Each frame’s magnitude is divided by its sum when `normalised=True` (regime stage and flux sidecar only). `flux_reference` is then comparable across files and across a −40 dB gain; the trim decision remains a ratio.

## Floor rule

```
n_fft = regime_analysis_n_fft or pitch_frame_length or frame_length
floor_seconds = max(regime_min_duration,
                    regime_min_windows * n_fft / sr)
```

If the pitch-stable span is already shorter than the floor, or a candidate trim would go below it, the stage sets `refused=True`, `refused_reason='span_below_floor'`, and keeps the incoming boundaries. The floor is applied to the **combined** (inner) flux + HI result.

## Two-walk combination

On each side the kept boundary is the later start / earlier end of the flux candidate and the HI candidate. `boundary_source_start` / `boundary_source_end` record which walk won (`flux`, `half_integer`, or `none`).

## When to use which mode

- **annotate** (default) — keep `_Sustains/` as energy + pitch; record `window_start` / `window_end` and optional `<stem>.flux.json` for weighting frames.
- **trim** — also write `_Sustains_Stable/` from the combined stable window. Use for sampler cores that must not include the half-integer onset or tail of soft high brass.
- **off** — `use_regime_refine=False`; `regime_refine == {}`.

Preset `soft_high_brass` is the Very Long sustained-tone profile with `regime_refine_mode="trim"`, `regime_flux_ratio=2.0`, `pitch_stability_cents=8.0`.
