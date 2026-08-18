# ADSR_Segmenter — Comprehensive Technical Manual

**Version:** 3.3.0 (`adsr-segmenter`)  
**Repository:** [github.com/LuisMRaimundo/ADSR_Segmenter](https://github.com/LuisMRaimundo/ADSR_Segmenter)  
**Audience:** Musicologists, acousticians, sound designers, and software engineers  
**Copyright:** © 2026 Luís Raimundo. Proprietary research software — see `# Copyright and Use Notice.md`.

---

## Table of Contents

1. [Interdisciplinary Overview](#1-interdisciplinary-overview)
2. [Purpose and Scope](#2-purpose-and-scope)
3. [System Architecture](#3-system-architecture)
4. [Acoustic Model: ADSR Segmentation](#4-acoustic-model-adsr-segmentation)
5. [Signal Processing Pipeline](#5-signal-processing-pipeline)
6. [Detection Algorithms](#6-detection-algorithms)
7. [Pitch-Based Sustain Refinement](#7-pitch-based-sustain-refinement)
8. [Spectral-regime refinement](#8-spectral-regime-refinement)
9. [Boundary Editing and Click-Free Export](#9-boundary-editing-and-click-free-export)
10. [Output Layout and Metadata](#10-output-layout-and-metadata)
11. [Default Parametrization Reference](#11-default-parametrization-reference)
12. [Mathematical Formalism (LaTeX)](#12-mathematical-formalism-latex)
13. [API Reference](#13-api-reference)
14. [GUI and CLI Applications](#14-gui-and-cli-applications)
15. [Tutorials](#15-tutorials)
16. [Boundary Benchmark](#16-boundary-benchmark)
17. [Testing](#17-testing)
18. [Troubleshooting](#18-troubleshooting)
19. [Dependencies](#19-dependencies)

---

## 1. Interdisciplinary Overview

### 1.1 Musicological framing

The **ADSR envelope** (Attack, Decay, Sustain, Release) is a canonical abstraction in organology and sample-based music production. For **monophonic orchestral one-shots** (single bow stroke, pluck, tongued wind attack), the model maps perceptually to:

| Region | Perceptual correlate | Typical orchestral content |
|--------|----------------------|----------------------------|
| **Attack** | Onset → initial spectral/energy rise | Bow scrape, breath noise, hammer transient |
| **Sustain** | Steady-state excitation | Stable pitch, vibrato, harmonic balance |
| **Decay** | Post-articulation energy fall | Bow lift, tongue release, damping |
| **Release** | Residual tail | Room, string ring-out, silence |

This tool **does not** perform source separation or polyphonic transcription. It assumes **one active event per file** (or quasi-monophonic material where a single dominant envelope is meaningful).

### 1.2 Acoustic / DSP perspective

Boundaries are inferred from **short-time energy** (RMS), optionally **spectral flux** and **RMS derivatives**, then optionally refined by **fundamental-frequency stability** (YIN). The design balances:

- **Physical envelope tracking** (energy thresholds relative to peak)
- **Proportional anchors** (duration-class priors from presets)
- **Musical constraints** (minimum sustain, decay guard, pitch-stable windows)

### 1.3 Software engineering perspective

All detection logic lives in **`audio_segment_core.py`** (pure NumPy/librosa, unit-testable). The Tkinter GUI (`split_audio_segments.py`) and headless CLI (`split_audio_cli.py`) are thin orchestration layers. Times exist in two coordinate systems:

- **File time** \(t\): seconds from sample 0 of the loaded file
- **Trim-relative time** \(t'\): seconds from the start of the active region after silence trimming

Conversion: \(t = t_{\mathrm{start}} + t'\), where \(t_{\mathrm{start}} = n_{\mathrm{start}} / f_s\).

---

## 2. Purpose and Scope

Automatically splits monophonic or quasi-monophonic audio into classical envelope regions plus composites:

| Output folder | Content |
|---------------|---------|
| `_Attacks/` | Onset → attack boundary |
| `_Sustains/` | Attack boundary → decay boundary |
| `_Decays/` | Decay boundary → end of active sound |
| `_Release_Silence/` | Tail after active energy (no fades) |
| `_Full_Active_Sound/` | Full trimmed active region |

**Supported formats:** `.wav`, `.mp3`, `.flac`, `.aif`, `.aiff`, `.ogg`, `.m4a`, `.wma`, `.mp4`, `.mka`.

**Out of scope:** polyphonic separation, beat slicing, phrase-level semantics.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  split_audio_segments.py  (Tkinter GUI + batch orchestration)   │
│  • folder picker, presets, review UI, metadata export           │
└────────────────────────────┬────────────────────────────────────┘
                             │ SegmentConfig, detect/extract calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  audio_segment_core.py  (pure DSP — no GUI, unit-testable)      │
│  trim → detect (smart/advanced/proportional) → pitch refine     │
│  → regime refine (annotate/trim) → zero-crossing snap → fades   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    librosa              numpy              soundfile
  (load, trim, RMS,     (arrays, math)     (write WAV/FLAC/…)
   STFT, YIN)
```

| Module | Role |
|--------|------|
| `audio_segment_core.py` | All detection and extraction logic |
| `split_audio_segments.py` | Desktop GUI, manual review, metadata |
| `split_audio_cli.py` | Headless batch CLI |
| `run_benchmark.py` | Boundary-error benchmark |
| `benchmark/benchmark_core.py` | MAE metrics, annotation I/O |

---

## 4. Acoustic Model: ADSR Segmentation

```
Amplitude
    │     ┌──────── sustain plateau ────────┐
    │    /│                                  │\
    │   / │                                  │ \
    │  /  │                                  │  \ decay
    │ /   │                                  │   \
────┴─────┴──────────────────────────────────┴────\────► time
  silence  attack end              decay start   end   release
           (t_att)                  (t_dec)      (t_end)
```

| Symbol | Meaning |
|--------|---------|
| \(t_{\mathrm{start}}\) | Start of active audio after trim |
| \(t_{\mathrm{att}}\) | End of attack / start of sustain |
| \(t_{\mathrm{dec}}\) | Start of decay / end of sustain |
| \(t_{\mathrm{end}}\) | End of musically active energy |
| Release | \([t_{\mathrm{end}}, t_{\mathrm{EOF}}]\) |

Preset percentages (`attack_pct`, `sustain_pct`, `decay_pct`) define **target proportions of active length** used as anchors, not fixed durations in seconds.

---

## 5. Signal Processing Pipeline

### 5.1 End-to-end stages

1. **Load** — native sample rate preserved: `librosa.load(path, sr=None)`
2. **Preprocess** — optional DC removal
3. **Trim** — `librosa.effects.trim` at `trim_db` below peak
4. **Detect** — proportional / smart / advanced mode
5. **Pitch refine** — optional YIN-based sustain adjustment
6. **Regime refine** — optional spectral-flux stationarity (default: annotate only)
7. **Clamp** — enforce minimum sustain and decay tails
8. **Extract** — zero-crossing snap, cosine/Hann/linear fades

### 5.2 Short-time analysis defaults

| Parameter | Default | Effect @ 44.1 kHz |
|-----------|---------|-------------------|
| `frame_length` \(N\) | 1024 | ~23.2 ms window |
| `hop_length` \(H\) | 512 | 50% overlap; frame period \(H/f_s \approx 11.6\) ms |
| RMS | `librosa.feature.rms` | Short-time energy envelope |
| Spectral flux | positive STFT magnitude differences | Onset emphasis (Advanced mode) |

---

## 6. Detection Algorithms

Mode selection in `detect_segments()`:

```text
if cfg.use_advanced:     → detect_segments_advanced_rel()
elif cfg.use_smart:      → detect_segments_smart_rel()   ← default
else:                     → detect_segments_proportional()
```

### 6.1 Proportional mode

Purely duration-based on active length \(L\). Normalized fractions \(\alpha, \sigma, \delta\) sum to 1. See [§11.4](#114-proportional-mode).

**Use when:** homogeneous libraries needing deterministic, preset-driven splits.

### 6.2 Smart mode (default)

Energy-guided boundaries blended with proportional anchors (weights 0.7 / 0.3). See [§11.5](#115-smart-mode-energy--proportional-blend).

**Use when:** general orchestral one-shots; default recommended setting.

### 6.3 Advanced mode

Combined energy + derivative attack; consecutive negative-ΔRMS decay detection; optional sustain plateau snap. See [§11.6](#116-advanced-mode).

**Use when:** spectrally rich but energetically subtle attacks (bow noise, breath, slow RMS rise).

**Mutual exclusion:** Advanced disables Smart in the GUI and CLI.

### 6.4 Energy detectors (summary)

- **Attack:** first frame with \(\mathrm{RMS}[k] \geq \theta_{\mathrm{att}} \cdot \mathrm{RMS}[k_{\mathrm{peak}}]\) scanning toward peak
- **Decay:** first frame with \(\mathrm{RMS}[k] \leq \theta_{\mathrm{dec}} \cdot \max(\mathrm{RMS})\) after attack/peak, subject to minimum decay time guard

Thresholds are **relative to trimmed peak**, not absolute dBFS.

---

## 7. Pitch-Based Sustain Refinement

### 7.1 Modes

| Mode | Export behaviour | Use case |
|------|------------------|----------|
| **`expand`** (default) | Grows pitch-stable seed outward; reverts if below `pitch_refine_min_fraction` | Long bowed notes |
| **`annotate`** | Keeps energy sustain; records stable window in metadata | STFT / spectral analysis |
| **`crop`** | Exports tightest stable window only | Legacy sampler cores |
| **`off`** | Skips refinement | Noisy / unpitched material |

### 7.2 Algorithm summary

1. Derive YIN `fmin`/`fmax` in this order: (a) note parsed from the filename ±1 octave; (b) `expected_note_hz` ±1 octave; (c) `pitch_fmin`/`pitch_fmax` (defaults 30–4200 Hz). The exact octave-below bin is excluded so the tracker cannot sit on \(f_0/2\).
2. Frame length scales with `fmin`: \(\geq 4\,f_s/f_{\min}\), power of two, capped at 8192. Hop is unchanged.
3. Cents deviation from median F0; vibrato-robust stability (linear detrend + moving-median residual).
4. Sliding-window seed search; optional expansion at \(1.25 \times\) stability tolerance.
5. **Failure / unvoiced:** if the voiced-frame fraction (frames within 200 ¢ of the median) is below `pitch_min_voiced_fraction` (0.5) → `failed_reason="unvoiced"`. If the best-window σ exceeds `pitch_fail_cents` (50) → `"tracking_failed"`. If \(|\mathrm{median}\,f_0 - f_{\mathrm{expected}}| > 300\) ¢ → `"octave_error"`. In all three cases `used=False`, energy boundaries are kept, and the regime stage receives `f0_hz=None`.
6. Octave-wrap filename spellings (`B#`, `Cb`, `E#`, `Fb`) are resolved to the sounding pitch and flagged (`note_name_wrap_spelling`). A tracked median more than 300 ¢ from the parsed note (or more than 50 ¢ on a wrap spelling) sets `note_name_mismatch`.

See [§12.7](#127-pitch-based-sustain-refinement).

---

## 8. Spectral-regime refinement

Level and pitch can be steady while the **oscillation regime** is not. A soft high brass note (Iowa tenor trombone, *pp*, C5) can have pitch σ of 2–5 cents and level flat within ±1 dB while spectral flux and the half-integer bands still move. The energy plateau and `refine_sustain_by_pitch` are blind to that third non-stationarity.

### Why relative

v3.2 used absolute constants (Hz, dB, samples) tuned on a tenor-trombone C5. Those do not travel:

- **Register.** A ±60 Hz band around 1.5·f₀ swallows H1 and H2 at E2 and is only meaningful above ~B3. Bands are now \(\pm\alpha f_0\) with \(\alpha=0.15\).
- **Level.** Raw flux on a −41 dBFS excerpt of the same take is two orders of magnitude smaller than the full-level take. Regime flux is computed on a frame-energy-normalised magnitude (\( |X| / \sum|X| \)). The Advanced-mode attack detector still calls the unnormalised function so v3.1 attack MAEs do not move.
- **Vibrato / tremolo.** A 6 Hz wobble must not look like a regime edge. The same moving-median window as the pitch stage is applied to the flux and half-integer tracks before the reference and the walks (`regime_vibrato_robust`).
- **Multiphonics / sul ponticello.** A *steady* half-integer spectrum is not an edge: the HI walk is relative to the note’s own mid-sustain, so a constant multiphonic is not trimmed.
- **Unpitched.** If pitch tracking fails or the material is unvoiced, `f0_hz=None`, `half_integer_valid=False`, and the stage runs on flux only.

### 8.1 Algorithm

1. Resolve \(N_{\mathrm{STFT}}\): `regime_analysis_n_fft` if set, else the pitch-stage frame length, else `frame_length`. Report `analysis_n_fft_source ∈ {config, pitch_frame, frame_length}`.
2. Compute **normalised** flux on the segmenter frame grid. Optionally low-pass with the vibrato median window, then the 9-frame moving median.
3. Restrict to the pitch-stable sustain `[t_att, t_dec]`. Flux reference = median over the central `regime_reference_fraction` (default 0.5).
4. **Flux walk:** inward from each end while smoothed flux > `regime_flux_ratio` × reference (default 2.0).
5. **Half-integer walk** (if `half_integer_valid` and `regime_use_half_integer`): bands \(\pm\alpha f_0\) around \(f_0\), \(1.5 f_0\), and \(2.5 f_0\). Gate if \(f_0\) is missing (`no_f0`) or \(\alpha f_0 < 2\,f_s/N_{\mathrm{STFT}}\) (`band_below_resolution`). Reference = median HI ratio (dB) over the same central fraction; walk inward while \(\mathrm{ratio} - \mathrm{reference} > \) `regime_hi_rise_db` (10 dB).
6. **Combination:** on each side the final boundary is the **more conservative (inner)** of the flux and HI candidates. `boundary_source_* ∈ {flux, half_integer, none}`.
7. **Floor:** applied to the *combined* result exactly as in v3.2. If the candidate span is shorter than `effective_regime_floor`, do not trim (`refused=True`). If the floor itself exceeds the available span, keep the inputs and set `refused_reason='span_below_floor'`.
8. Mode **annotate** (default): returned times equal the inputs; `window_start` / `window_end` report the candidate. Mode **trim**: returned times are the candidate boundaries. `t_end` and decay/release logic are untouched.

\[
t_{\mathrm{floor}} = \max\!\left(
  t_{\mathrm{min}},\;
  K_{\mathrm{win}}\frac{N_{\mathrm{STFT}}}{f_s}
\right)
\]

See [docs/REGIME_REFINE_NOTES.md](REGIME_REFINE_NOTES.md) for why each default was chosen.

### 8.2 Metadata keys (`regime_refine`)

Always present, `None` when not applicable: `used`, `mode`, `flux_reference`, `flux_edge_ratio_start`, `flux_edge_ratio_end`, `window_start`, `window_end`, `window_duration`, `trimmed_start_s`, `trimmed_end_s`, `refused`, `refused_reason`, `floor_seconds`, `floor_windows`, `analysis_n_fft`, `analysis_n_fft_source`, `half_integer_ratio_db_edges`, `half_integer_ratio_db_middle`, `half_integer_valid`, `half_integer_bandwidth_hz`, `hi_reference_db`, `hi_edge_rise_db_start`, `hi_edge_rise_db_end`, `hi_trimmed_start_s`, `hi_trimmed_end_s`, `boundary_source_start`, `boundary_source_end`.

### 8.3 Flux sidecar (`--flux-sidecar` / GUI checkbox)

`<stem>.flux.json` on the sustain frame grid. Existing keys are unchanged; v3.3 adds `flux_normalised`, `half_integer_valid`, `hi_bandwidth_hz`, `f0_hz`, `pitch_frame_length`, and the smoothed tracks:

```json
{"sr": 44100, "hop_length": 512, "n_fft": 1024,
 "times": [0.0, ...], "flux": [...], "half_integer_ratio_db": [...],
 "flux_normalised": true, "half_integer_valid": true,
 "hi_bandwidth_hz": 78.5, "f0_hz": 523.25, "pitch_frame_length": 1024,
 "flux_smoothed": [...], "half_integer_ratio_db_smoothed": [...]}
```

A downstream analyser can weight or median frames without recomputing STFT.

### 8.4 Annotate versus trim

| Mode | `_Sustains/` | `_Sustains_Stable/` | Use |
|------|----------------|----------------------|-----|
| **annotate** (default) | Energy + pitch (unchanged) | not written | STFT / spectral work; keep every available period |
| **trim** | Energy + pitch (unchanged) | flux- and/or HI-stable window | Sampler cores that must drop a half-integer onset or tail |
| **off** | Energy + pitch | not written | `regime_refine == {}` |

Prefer **annotate** unless the exported sustain itself must exclude the unstable regime. Prefer **trim** for soft high brass (`soft_high_brass` preset).

---

## 9. Boundary Editing and Click-Free Export

### 8.1 Zero-crossing alignment

Cuts moved to nearest sign change within ±100 ms, with linear interpolation for sub-sample placement. See [§11.8](#118-zero-crossing-alignment).

### 8.2 Fades

| `fade_type` | Shape |
|-------------|--------|
| `cosine` (default) | Raised cosine |
| `hann` | Hann window half |
| `linear` | Linear ramp |

Fade length clamped: \(\max(\texttt{fade\_ms}, 50\,\mathrm{ms})\) up to half segment length. See [§11.9](#119-fade-envelopes).

### 8.3 Manual review (GUI)

Drag green (attack) and orange (decay) lines; arrow keys nudge 5 ms (Shift = 25 ms). Edits re-export without re-running detection.

---

## 10. Output Layout and Metadata

For input `Violin_A4.wav`:

```text
source_folder/
├── _Attacks/Violin_A4_Attack.wav
├── _Sustains/Violin_A4_Sustain.wav
├── _Sustains_Stable/Violin_A4_SustainStable.wav   # trim mode only
├── _Decays/Violin_A4_Decay.wav
├── _Release_Silence/Violin_A4_Release.wav
├── _Full_Active_Sound/Violin_A4_FullActive.wav
├── segmentation_metadata.json
└── segmentation_metadata.csv
```

JSON per-file keys include `segments.attack_end`, `decay_start`, `end`, `durations.*`, `pitch_stability`, `regime_refine`, and optional `regime_flux_sidecar`.

---

## 11. Default Parametrization Reference

### 10.1 Global module constants (`audio_segment_core.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_TRIM_DB` | 60.0 | Trim threshold (dB below peak) |
| `DEFAULT_FRAME_LENGTH` | 1024 | STFT / RMS frame size |
| `DEFAULT_HOP_LENGTH` | 512 | Hop size |
| `DEFAULT_MIN_SUSTAIN_FRAMES` | 40 | Frame-based sustain floor |
| `DEFAULT_SUSTAIN_VARIANCE_THRESHOLD` | 0.2 | Plateau CV² threshold (Advanced) |
| `DEFAULT_ZERO_CROSSING_SEARCH_MS` | 100.0 | ZC search radius |
| `SMART_ENERGY_BLEND` | 0.7 | Smart mode energy weight |
| `SMART_PROP_BLEND` | 0.3 | Smart mode proportional weight |
| `DEFAULT_VIBRATO_MEDIAN_WINDOW_S` | 0.12 | Vibrato suppression window (s) |
| `DEFAULT_PITCH_REFINE_MIN_FRACTION` | 0.70 | Min refined / energy sustain ratio |
| `DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY` | 0.75 | Decay guard through proportional sustain |
| `DEFAULT_PITCH_FMIN_HZ` | 30.0 | Fallback YIN lower bound |
| `DEFAULT_PITCH_FMAX_HZ` | 4200.0 | Fallback YIN upper bound |
| `DEFAULT_PITCH_FAIL_CENTS` | 50.0 | Best-window σ that marks tracking failure |
| `DEFAULT_PITCH_OCTAVE_FAIL_CENTS` | 300.0 | Median vs expected → `octave_error` |
| `DEFAULT_PITCH_MIN_VOICED_FRACTION` | 0.5 | Consistent-frame fraction for “voiced” |
| `DEFAULT_REGIME_HI_REL_BANDWIDTH` | 0.15 | \(\alpha\) for \(\pm\alpha f_0\) HI bands |
| `DEFAULT_REGIME_HI_RISE_DB` | 10.0 | HI walk vs mid-sustain reference |

### 10.2 `SegmentConfig` dataclass defaults

These are the **authoritative detection defaults** used by the core library, CLI, and GUI after preset application:

| Field | Default | Unit / type | Description |
|-------|---------|-------------|-------------|
| `trim_db` | 60.0 | dB | Silence trim below peak |
| `attack_threshold` | 0.90 | ratio | Attack end at 90% of peak RMS |
| `decay_threshold` | 0.50 | ratio | Decay start at 50% of peak RMS |
| `attack_pct` | 0.15 | fraction | Proportional attack share |
| `sustain_pct` | 0.60 | fraction | Proportional sustain share |
| `decay_pct` | 0.25 | fraction | Proportional decay share |
| `min_sustain_duration` | 0.35 | s | Minimum sustain length |
| `pitch_window_duration` | 0.5 | s | YIN analysis grain |
| `pitch_stability_cents` | 5.0 | cents | Max σ for stable pitch |
| `use_advanced` | `False` | bool | Derivative + flux mode |
| `use_smart` | `True` | bool | Energy + proportional blend |
| `sustain_variance_threshold` | 0.2 | — | Normalized sustain variance cap |
| `frame_length` | 1024 | samples | Analysis frame |
| `hop_length` | 512 | samples | Hop size |
| `min_sustain_frames` | 40 | frames | Frame sustain floor |
| `vibrato_robust` | `True` | bool | Detrend + median vibrato suppression |
| `vibrato_median_window_s` | 0.12 | s | Moving-median window |
| `remove_dc` | `True` | bool | DC offset removal before analysis |
| `use_pitch_refine` | `True` | bool | Enable pitch refinement |
| `pitch_refine_mode` | `"expand"` | enum | `expand` / `annotate` / `crop` |
| `pitch_refine_min_fraction` | 0.70 | ratio | Revert threshold |
| `sustain_fraction_before_decay` | 0.75 | ratio | Earliest decay guard |
| `use_regime_refine` | `True` | bool | Enable spectral-regime stage |
| `regime_refine_mode` | `"annotate"` | enum | `annotate` / `trim` |
| `regime_flux_ratio` | 2.0 | ratio | Walk inward while flux > ratio × reference |
| `regime_flux_median_frames` | 9 | frames | Odd moving-median on flux |
| `regime_reference_fraction` | 0.5 | fraction | Central span used as flux reference |
| `regime_min_windows` | 20 | windows | Floor in downstream STFT hops |
| `regime_min_duration` | 1.0 | s | Floor in seconds |
| `regime_analysis_n_fft` | `None` | samples | Downstream STFT size (pitch frame if None) |
| `regime_half_integer` | `True` | bool | Compute 1.5·f₀ / 2.5·f₀ vs f₀ (dB) |
| `regime_hi_rel_bandwidth` | 0.15 | fraction of f₀ | Half-width of HI and H1 bands |
| `regime_use_half_integer` | `True` | bool | Second inward walk on the HI ratio |
| `regime_hi_rise_db` | 10.0 | dB | Walk while ratio − mid-sustain reference exceeds this |
| `regime_vibrato_robust` | `True` | bool | Moving-median on flux / HI before walks |
| `pitch_fmin` | 30.0 | Hz | Fallback YIN lower bound |
| `pitch_fmax` | 4200.0 | Hz | Fallback YIN upper bound |
| `pitch_fail_cents` | 50.0 | cents | Best-window σ → `tracking_failed` |
| `pitch_min_voiced_fraction` | 0.5 | fraction | Consistent frames required to treat as voiced |

**Effective minimum sustain:**

\[
t_{\mathrm{sus,min}} = \max\!\left(
  t_{\mathrm{sus,cfg}},\;
  t_{\mathrm{pitch,win}},\;
  \frac{K_{\min} H}{f_s}
\right)
\]

For very short sounds (\(L < t_{\mathrm{sus,min}}\)):

\[
t_{\mathrm{sus,min}} \leftarrow \max\!\left(0.25\,L,\; \frac{K_{\min} H}{f_s},\; 0.02\right)
\]

where \(K_{\min} =\) `min_sustain_frames`, \(H =\) `hop_length`, \(f_s =\) sample rate.

### 10.3 Export / GUI-only defaults

| Parameter | Default | Source |
|-----------|---------|--------|
| `fade_ms` | 50.0 | GUI `DEFAULT_FADE_MS`; CLI uses preset value |
| `fade_type` | `"cosine"` | GUI / CLI |
| CLI `--preset` | `"Medium (1.5-3.0s)"` | `split_audio_cli.py` |
| Benchmark tolerance | 50 ms | `DEFAULT_TOLERANCE_MS` |

**Note:** The GUI initializes `min_sustain_duration` spinbox to 1.0 s before a preset is applied; **Apply Preset** overwrites this from the preset table below.

### 10.4 Duration presets (`PRESETS`)

Percentages are normalized to sum to 1 internally if needed.

| Preset | attack% | sustain% | decay% | fade ms | min sustain (s) | θ_att | θ_dec | Extra |
|--------|---------|----------|--------|---------|-----------------|-------|-------|-------|
| Very Short (< 0.5s) | 0.20 | 0.50 | 0.30 | 30 | 0.06 | 0.85 | 0.45 | — |
| Short (0.5–1.5s) | 0.15 | 0.60 | 0.25 | 40 | 0.15 | 0.90 | 0.50 | — |
| **Medium (1.5–3.0s)** | 0.12 | 0.65 | 0.23 | 50 | 0.35 | 0.90 | 0.50 | CLI default |
| Long (3.0–6.0s) | 0.10 | 0.70 | 0.20 | 60 | 0.60 | 0.90 | 0.45 | expand, min_frac 0.72 |
| Very Long (> 6.0s) | 0.08 | 0.75 | 0.17 | 70 | 1.00 | 0.90 | 0.40 | expand, min_frac 0.75 |
| Custom | 0.15 | 0.60 | 0.25 | 50 | 0.35 | 0.90 | 0.50 | — |
| soft_high_brass | 0.08 | 0.75 | 0.17 | 70 | 1.00 | 0.90 | 0.40 | trim, flux ratio 2.0, pitch σ = 8¢ |

Every preset also carries the regime-annotate keys (`use_regime_refine=True`, `regime_refine_mode="annotate"`, …) except `soft_high_brass`, which sets `trim`.

### 10.5 Articulation presets (`ARTICULATION_PRESETS`)

| Preset | attack% | sustain% | decay% | fade ms | min sus (s) | θ_att | θ_dec | Mode |
|--------|---------|----------|--------|---------|-------------|-------|-------|------|
| Staccato / Pluck | 0.22 | 0.45 | 0.33 | 25 | 0.04 | 0.82 | 0.55 | Advanced |
| Legato / Bow | 0.10 | 0.72 | 0.18 | 55 | 0.45 | 0.88 | 0.42 | Smart; pitch σ = 8¢ |
| Marcato / Accent | 0.18 | 0.52 | 0.30 | 35 | 0.12 | 0.80 | 0.48 | Advanced |

Build: `SegmentConfig.from_preset("Legato / Bow")`.

### 10.6 Auto-detect mean length (GUI)

Scans up to 100 files, trims each at 60 dB, averages active duration, selects matching duration preset:

| Mean active length | Auto-selected preset |
|--------------------|----------------------|
| < 0.5 s | Very Short |
| 0.5 – 1.5 s | Short |
| 1.5 – 3.0 s | Medium |
| 3.0 – 6.0 s | Long |
| > 6.0 s | Very Long |

---

## 12. Mathematical Formalism (LaTeX)

This section gives the complete mathematical apparatus implemented in `audio_segment_core.py`.

### 11.1 Signal representation and preprocessing

Let the loaded mono signal be \(x[n]\), \(n = 0, \ldots, N-1\), sample rate \(f_s\) (Hz).

**DC removal** (when `remove_dc=True`):

\[
\tilde{x}[n] = x[n] - \frac{1}{N}\sum_{m=0}^{N-1} x[m]
\]

### 11.2 Active-region trim

Librosa trim finds the shortest contiguous segment whose envelope exceeds a peak-relative threshold. With peak amplitude \(A_{\mathrm{peak}} = \max_n |\tilde{x}[n]|\) and threshold \(T_{\mathrm{dB}} =\) `trim_db`:

\[
A_{\mathrm{thresh}} = A_{\mathrm{peak}} \cdot 10^{-T_{\mathrm{dB}}/20}
\]

Samples outside the retained interval \([n_{\mathrm{start}}, n_{\mathrm{end}})\) are discarded for detection. Define:

\[
t_{\mathrm{start}} = \frac{n_{\mathrm{start}}}{f_s}, \quad
t_{\mathrm{end}} = \min\!\left(\frac{n_{\mathrm{end}}}{f_s},\; \frac{N}{f_s} - 10^{-3}\right), \quad
L = t_{\mathrm{end}} - t_{\mathrm{start}}
\]

The trimmed signal is \(y[n] = \tilde{x}[n_{\mathrm{start}} + n]\), \(n = 0, \ldots, n_{\mathrm{end}} - n_{\mathrm{start}} - 1\).

### 11.3 Short-time RMS envelope

Frame length \(N_f =\) `frame_length`, hop \(H =\) `hop_length`. Librosa RMS for frame index \(k\):

\[
\mathrm{RMS}[k] = \sqrt{\frac{1}{N_f}\sum_{m=0}^{N_f-1} y^2[m + kH]}
\]

Frame center times (approximate):

\[
t_k = \frac{kH}{f_s} \quad \text{(via \texttt{librosa.times\_like})}
\]

Peak frame index:

\[
k_{\mathrm{peak}} = \arg\max_k \mathrm{RMS}[k]
\]

**Frame period** (for converting frame counts to seconds):

\[
\Delta t_{\mathrm{frame}} = \frac{H}{f_s}
\]

### 11.4 STFT and spectral flux

Short-time Fourier transform with \(N_f\) point FFT, hop \(H\):

\[
X[k, \ell] = \sum_{m=0}^{N_f-1} y[m + \ell H]\, w[m]\, e^{-j2\pi km/N_f}
\]

Magnitude \(|X[k,\ell]|\). **Spectral flux** (half-wave rectified frame difference):

\[
\Phi[\ell] = \sum_{k=0}^{N_f/2} \max\!\left(|X[k,\ell+1]| - |X[k,\ell]|,\; 0\right)
\]

The regime stage first normalises each frame, \(\tilde{X}[k,\ell] = |X[k,\ell]| / \sum_k |X[k,\ell]|\), so \(\Phi\) is level-independent (`normalised=True`). Advanced-mode attack detection keeps the unnormalised call (`normalised=False`) so v3.1 attack MAEs do not move.

### 11.5 Proportional mode

Given raw preset fractions \(a, s, d\) (`attack_pct`, `sustain_pct`, `decay_pct`), normalize:

\[
\alpha = \frac{a}{a+s+d}, \quad
\sigma = \frac{s}{a+s+d}, \quad
\delta = \frac{d}{a+s+d}
\]

Trim-relative boundaries:

\[
t'_{\mathrm{att,prop}} = \alpha L, \quad
t'_{\mathrm{dec,prop}} = (\alpha + \sigma) L
\]

Minimum sustain enforcement:

\[
t_{\mathrm{sus,act}} = \max\!\left(t_{\mathrm{sus,min}},\; 0.4\,\sigma L\right)
\]

\[
t'_{\mathrm{dec}} = \max\!\left(t'_{\mathrm{att}} + t_{\mathrm{sus,act}},\; t'_{\mathrm{dec,prop}}\right)
\]

End margin (5% of active length):

\[
t'_{\mathrm{dec}} \leftarrow \min\!\left(t'_{\mathrm{dec}},\; L - 0.05 L\right)
\]

Fallback if ordering fails:

\[
t'_{\mathrm{dec}} = \min\!\left(t'_{\mathrm{att}} + \sigma L,\; L - 0.02\right)
\]

### 11.6 Smart mode: energy + proportional blend

**Energy-based attack** (threshold \(\theta_{\mathrm{att}} =\) `attack_threshold`):

\[
E'_{\mathrm{att}} = \min\left\{ t_k : \mathrm{RMS}[k] \geq \theta_{\mathrm{att}} \cdot \mathrm{RMS}[k_{\mathrm{peak}}],\; k \in [0, k_{\mathrm{peak}}] \right\}
\]

**Decay guard** (minimum proportional sustain traversal, fraction \(\rho =\) `sustain_fraction_before_decay`):

\[
t'_{\mathrm{dec,min}} = L \cdot \left(\alpha + \sigma \rho\right)
\]

**Energy-based decay** (threshold \(\theta_{\mathrm{dec}} =\) `decay_threshold`):

\[
E'_{\mathrm{dec}} = \min\left\{ t_k : \mathrm{RMS}[k] \leq \theta_{\mathrm{dec}} \cdot \max_j \mathrm{RMS}[j],\; k \geq \max(k_{\mathrm{att}}, k_{\mathrm{peak}}),\; t_k \geq t'_{\mathrm{dec,min}} \right\}
\]

Fallback if no crossing: \(E'_{\mathrm{dec}} = 0.85\, t_{K-1}\) (85% of final frame time).

**Blend** (weights \(w_E = 0.7\), \(w_P = 0.3\)):

\[
t'_{\mathrm{att}} = w_E E'_{\mathrm{att}} + w_P t'_{\mathrm{att,prop}}
\]

\[
t'_{\mathrm{dec}} = \max\!\left(w_E E'_{\mathrm{dec}} + w_P t'_{\mathrm{dec,prop}},\; t'_{\mathrm{dec,min}}\right)
\]

**Clamp** (`_clamp_segment_rel`), with minimum decay tail \(t_{\mathrm{dec,tail}} = \max(0.02, 0.01)\):

\[
t'_{\mathrm{att}} \leftarrow \mathrm{clip}\!\left(t'_{\mathrm{att}},\; 0,\; L - t_{\mathrm{sus,min}} - t_{\mathrm{dec,tail}}\right)
\]

\[
t'_{\mathrm{dec}} \leftarrow \mathrm{clip}\!\left(t'_{\mathrm{dec}},\; t'_{\mathrm{att}} + t_{\mathrm{sus,min}},\; L - t_{\mathrm{dec,tail}}\right)
\]

### 11.7 Advanced mode

#### 11.7.1 Combined attack

Energy attack \(E'_{\mathrm{att}}\) as above. **Derivative attack:**

\[
\Delta\mathrm{RMS}[k] = \mathrm{RMS}[k+1] - \mathrm{RMS}[k]
\]

\[
k^* = \arg\max_{k \leq k_{\mathrm{peak}}} \Delta\mathrm{RMS}[k], \quad
D'_{\mathrm{att}} = t_{k^*}
\]

Spectral flux pull (optional):

\[
D'_{\mathrm{att}} \leftarrow \min\!\left(D'_{\mathrm{att}},\; t_{\arg\max \Phi}\right)
\]

Lower bound and peak cap:

\[
D'_{\mathrm{att}} \leftarrow \max\!\left(D'_{\mathrm{att}},\; 0.05\, t_{K-1}\right), \quad
D'_{\mathrm{att}} \leftarrow \min\!\left(D'_{\mathrm{att}},\; 0.85\, t_{k_{\mathrm{peak}}}\right)
\]

Combined:

\[
t'_{\mathrm{att}} = \min(E'_{\mathrm{att}}, D'_{\mathrm{att}})
\]

#### 11.7.2 Derivative decay

After peak time \(t_{k_{\mathrm{peak}}}\), enforce delay:

\[
\Delta t_{\mathrm{delay}} = \max\!\left(0.05,\; 0.15\,(t_{K-1} - t_{k_{\mathrm{peak}}})\right)
\]

Search for **3 consecutive** frames with \(\Delta\mathrm{RMS}[k] < 0\); decay time assigned to start of that run. Fallback: energy decay detector.

#### 11.7.3 Sustain plateau (Advanced)

For sustain frames \(k \in [k_{\mathrm{att}}, k_{\mathrm{dec}})\):

\[
\mu = \frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \mathrm{RMS}[k], \quad
V = \frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \left(\mathrm{RMS}[k] - \mu\right)^2
\]

Normalized variance:

\[
\hat{V} = \frac{V}{\mu^2}
\]

If \(\hat{V} < \tau_v\) (`sustain_variance_threshold` = 0.2) and duration \(\geq t_{\mathrm{sus,min}}\), snap boundaries to plateau edges \((k_{\mathrm{att}}, k_{\mathrm{dec}})\).

### 11.8 Pitch-based sustain refinement

#### 11.8.1 YIN fundamental frequency

On sustain slice \(y_{\mathrm{sus}}\), librosa YIN estimates \(f_0[k]\). Search bounds come from the filename note or `expected_note_hz` (±1 octave, excluding the exact subharmonic) else `pitch_fmin`–`pitch_fmax` (30–4200 Hz). The analysis frame is \(\geq 4\,f_s/f_{\min}\) (power of two, ≤ 8192).

Valid frames: \(\mathcal{V} = \{k : \mathrm{finite}(f_0[k]) \land f_0[k] > 0\}\).

Median reference:

\[
\bar{f}_0 = \mathrm{median}\{f_0[k] : k \in \mathcal{V}\}
\]

**Cents deviation** from median:

\[
c[k] = 1200 \log_2\!\left(\frac{f_0[k]}{\bar{f}_0}\right), \quad k \in \mathcal{V}
\]

**Cents from expected note** (filename hint \(f_{\mathrm{note}}\)):

\[
c_{\mathrm{note}}[k] = 1200 \log_2\!\left(\frac{f_0[k]}{f_{\mathrm{note}}}\right)
\]

#### 11.8.2 Vibrato-robust stability

Given valid cents sequence \(c_k\) at times \(t_k\), **linear detrend**:

\[
c'_k = c_k - (a (t_k - t_0) + b), \quad (a,b) = \mathrm{polyfit}(t, c, 1)
\]

**Moving-median vibrato suppression** (window \(W =\) `vibrato_median_window_s`):

\[
\tilde{c}_k = c'_k - \mathrm{MedianFilter}_{W}(c'_k)
\]

**Pitch stability metric:**

\[
\sigma_c = \mathrm{std}(\tilde{c}_k)
\]

Window is **stable** if \(\sigma_c \leq \tau_p\) (`pitch_stability_cents`, default 5¢).

#### 11.8.3 Seed window search

Analysis grain:

\[
W_{\mathrm{dur}} = \max\!\left(t_{\mathrm{pitch,win}},\; \min(0.5,\; t_{\mathrm{sus,min}})\right)
\]

\[
K_W = \left\lceil \frac{W_{\mathrm{dur}} f_s}{H} \right\rceil
\]

Sliding windows \(i = 0, \ldots, K - K_W\). Require \(\geq 60\%\) valid frames. **Score:**

\[
S_i = \sigma_c^{(i)} + \overline{|c_{\mathrm{note}}|}^{(i)}
\]

(second term zero if no note hint). Choose \(i^* = \arg\min S_i\).

#### 11.8.4 Expand mode

If \(\sigma_c^{(i^*)} \leq \tau_p\), grow seed \([i_{\mathrm{lo}}, i_{\mathrm{hi}})\) left/right while:

\[
\sigma_c(\text{expanded window}) \leq 1.25\,\tau_p
\]

Export boundaries (trim-relative):

\[
t'_{\mathrm{att,new}} = t'_{\mathrm{sus,start}} + t_{i_{\mathrm{lo}}}, \quad
t'_{\mathrm{dec,new}} = t'_{\mathrm{sus,start}} + t_{i_{\mathrm{hi}}-1}
\]

**Revert guard** (energy sustain duration \(T_{\mathrm{energy}} = t'_{\mathrm{dec}} - t'_{\mathrm{att}}\)):

\[
\text{if } (t'_{\mathrm{dec,new}} - t'_{\mathrm{att,new}}) < \eta T_{\mathrm{energy}} \quad (\eta = \texttt{pitch\_refine\_min\_fraction})
\]

then keep energy boundaries and set `kept_energy_boundaries: true`.

**Annotate mode:** same search but always returns energy boundaries; stable window stored in metadata only.

### 11.9 Zero-crossing alignment

Target sample index \(n_0\). Search window \(\mathcal{W} = [n_0 - \Delta n,\; n_0 + \Delta n]\) where \(\Delta n = \lfloor f_s \cdot T_{\mathrm{search}} \rfloor\), \(T_{\mathrm{search}} = 0.1\) s.

Find sign-change indices \(n_j\) where \(\mathrm{sign}(y[n_j]) \neq \mathrm{sign}(y[n_j+1])\). Pick crossing nearest \(n_0\).

**Sub-sample interpolation** between \(y_1 = y[n_j]\), \(y_2 = y[n_j+1]\):

\[
\tau = \frac{-y_1}{y_2 - y_1} \quad (|y_2 - y_1| > 10^{-10})
\]

\[
n_{\mathrm{ZC}} = n_{\mathrm{start}} + \mathrm{round}(n_j + \tau)
\]

### 11.10 Fade envelopes

Fade length in samples:

\[
N_{\mathrm{fade}} = \mathrm{clip}\!\left(\left\lfloor f_s \frac{T_{\mathrm{fade,ms}}}{1000}\right\rfloor,\; \left\lfloor\frac{f_s}{20}\right\rfloor,\; \left\lfloor\frac{N_{\mathrm{seg}}}{2}\right\rfloor\right)
\]

Parameter \(u \in [0,1]\), \(u_m = m/(N_{\mathrm{fade}}-1)\).

**Cosine (raised cosine, default):**

\[
g_{\mathrm{in}}(m) = \tfrac{1}{2}\left(1 - \cos(\pi u_m)\right), \quad
g_{\mathrm{out}}(m) = \tfrac{1}{2}\left(1 + \cos(\pi u_m)\right)
\]

**Linear:**

\[
g_{\mathrm{in}}(m) = u_m, \quad g_{\mathrm{out}}(m) = 1 - u_m
\]

**Hann:** first/second half of \(\mathrm{hanning}(2 N_{\mathrm{fade}})\).

Applied: \(y_{\mathrm{out}}[m] = y[m] \cdot g_{\mathrm{in}}(m)\) for \(m < N_{\mathrm{fade}}\) and \(y_{\mathrm{out}}[N-m-1] = y[N-m-1] \cdot g_{\mathrm{out}}(m)\).

If click verification fails, fades re-applied at \(1.5 \times T_{\mathrm{fade,ms}}\).

### 11.11 Click severity metric

Reference interior difference scale:

\[
d_{\mathrm{ref}} = \mathrm{median}\left(\left|y[n+1] - y[n]\right|_{n \in \mathcal{M}}\right) + \epsilon
\]

Edge severity:

\[
S = \max\!\left(
\frac{|y[0]|}{d_{\mathrm{ref}}},\;
\frac{|y[N-1]|}{d_{\mathrm{ref}}},\;
\frac{\max|\Delta y|_{\mathrm{start}}}{2 d_{\mathrm{ref}}},\;
\frac{\max|\Delta y|_{\mathrm{end}}}{2 d_{\mathrm{ref}}}
\right)
\]

Segment passes if \(|y[0]|, |y[N-1]| \leq 0.01\) and \(S \leq 4.0\).

### 11.12 File-time conversion and final constraints

After trim-relative detection and pitch refine:

\[
t_{\mathrm{att}} = t_{\mathrm{start}} + t'_{\mathrm{att}}, \quad
t_{\mathrm{dec}} = t_{\mathrm{start}} + t'_{\mathrm{dec}}, \quad
t_{\mathrm{end}} = \text{trim end}
\]

Final file-time clamp:

\[
t_{\mathrm{dec}} \leftarrow \min(t_{\mathrm{dec}},\; t_{\mathrm{end}} - \max(0.02,\; 0.05 L))
\]

\[
t_{\mathrm{att}} \leftarrow \min(t_{\mathrm{att}},\; t_{\mathrm{dec}} - \min(t_{\mathrm{sus,min}},\; 0.5 L))
\]

### 11.13 Benchmark error metrics

Per boundary \(b \in \{\mathrm{att}, \mathrm{dec}, \mathrm{end}\}\), ground truth \(\hat{t}_b\), prediction \(\tilde{t}_b\):

\[
e_b\;[\mathrm{ms}] = 1000 \left|\tilde{t}_b - \hat{t}_b\right|
\]

Mean error per sample:

\[
\bar{e} = \frac{1}{3}(e_{\mathrm{att}} + e_{\mathrm{dec}} + e_{\mathrm{end}})
\]

Aggregate MAE over \(N\) samples:

\[
\mathrm{MAE}_b = \frac{1}{N}\sum_{i=1}^{N} e_b^{(i)}, \quad
\mathrm{MAE}_{\mathrm{mean}} = \frac{1}{N}\sum_{i=1}^{N} \bar{e}^{(i)}
\]

Tolerance hit rate (default \(\tau = 50\) ms):

\[
P_{\leq\tau} = \frac{100}{N}\left|\left\{i : \max_b e_b^{(i)} \leq \tau\right\}\right|
\]

### 11.14 Time-index diagram

```text
File:     |---- leading silence ----|==== ACTIVE (trimmed) ====|---- release ----|
          0                    n_start              n_end              N-1

Trim-relative (t' = 0 at active start):
          0        t'_att              t'_dec              L
          |-- A ---|------ S ---------|------ D ---------|

File time:
          t_start  t_att               t_dec               t_end
```

Sample indices after zero-crossing snap:

\[
[n_{\mathrm{start}} : n_{\mathrm{att}}) \rightarrow \text{Attack},\;
[n_{\mathrm{att}} : n_{\mathrm{dec}}) \rightarrow \text{Sustain},\;
[n_{\mathrm{dec}} : n_{\mathrm{end}}) \rightarrow \text{Decay},\;
[n_{\mathrm{end}} : N) \rightarrow \text{Release}
\]

---

## 13. API Reference

### 12.1 Primary entry points

```python
from pathlib import Path
import librosa
import audio_segment_core as core

y, sr = librosa.load("note.wav", sr=None)
cfg = core.SegmentConfig(use_smart=True, attack_threshold=0.9)

result = core.detect_segments(y, sr, cfg, file_path=Path("Violin_A4.wav"))
# result.t_att, result.t_dec, result.t_end  — absolute times (seconds)
# result.trim — TrimInfo
# result.pitch_refine — dict

parts, idx_att, idx_dec, idx_end = core.extract_and_fade_segments(
    y, sr,
    result.t_att, result.t_dec, result.t_end,
    result.trim,
    fade_ms=50.0,
    fade_type="cosine",
)
```

### 12.2 Key functions

| Function | Purpose |
|----------|---------|
| `trim_active_region(y, sr, trim_db)` | Silence gate |
| `compute_rms_envelope(y, sr, …)` | Energy envelope |
| `compute_spectral_flux(y, sr, …)` | Onset-sensitive flux (`n_fft` optional) |
| `compute_half_integer_ratio_db(y, sr, f0, cfg)` | 1.5·f₀+2.5·f₀ vs f₀ (dB) |
| `refine_sustain_by_regime(…)` | Spectral-stationarity sustain window |
| `detect_segments(y, sr, cfg, file_path)` | Full detection pipeline |
| `validate_segments(t_att, t_dec, t_end)` | Ordering / min duration check |
| `extract_and_fade_segments(…)` | Slice + ZC + fade |
| `find_zero_crossing(y, idx, sr, search_ms)` | Nearest ZC sample index |
| `apply_fades(audio, sr, fade_ms, fade_type)` | Edge ramps |
| `parse_note_hz_from_filename(path)` | Expected F0 from name |
| `SegmentConfig.from_preset(name, **overrides)` | Preset builder |
| `process_audio_file(path, out_dir, cfg, …)` | Headless single-file pipeline |
| `batch_process_folder(folder, cfg, …)` | Batch wrapper |

### 12.3 Validation rules

`validate_segments()` requires \(t_{\mathrm{att}} < t_{\mathrm{dec}} < t_{\mathrm{end}}\), sustain \(\geq 10\) ms, decay tail \(\geq 10\) ms.

---

## 14. GUI and CLI Applications

### 13.1 Launch

**Windows:** double-click `run.bat` or:

```bash
pip install -e ".[dev]"
python split_audio_segments.py          # GUI
python split_audio_cli.py -f ./samples  # headless batch
```

Entry points: `adsr-segmenter-gui`, `adsr-segmenter-cli`, `adsr-segmenter-benchmark`.

### 13.2 CLI flags

| Flag | Description |
|------|-------------|
| `--folder`, `-f` | Input directory (required) |
| `--preset`, `-p` | Duration or articulation preset |
| `--fade-ms` | Fade length (default: from preset) |
| `--fade-type` | `cosine` / `hann` / `linear` |
| `--attack-threshold` | Override θ_att |
| `--decay-threshold` | Override θ_dec |
| `--min-sustain` | Override minimum sustain (s) |
| `--advanced` | Advanced detection mode |
| `--proportional` | Proportional-only mode |
| `--no-pitch-refine` | Disable pitch refinement |
| `--pitch-refine-mode` | `expand` / `annotate` / `crop` |
| `--no-vibrato-robust` | Disable vibrato suppression |
| `--regime-mode` | `annotate` / `trim` / `off` |
| `--regime-flux-ratio` | Flux edge / reference threshold |
| `--regime-analysis-n-fft` | Downstream STFT size for the floor |
| `--flux-sidecar` | Write `<stem>.flux.json` |
| `--export-metadata` | Write JSON + CSV |
| `--output` | Output directory (default: source folder) |

**MP3/M4A:** requires **ffmpeg** on PATH.

### 13.3 GUI workflow

1. **Source Folder** — select input directory
2. **Preset Configuration** — duration class; **Auto-Detect Mean Length** or manual mean length; **Apply Preset**
3. **Segmentation Parameters** — thresholds, fades, Smart/Advanced, pitch refine, regime refine (mode / flux ratio / n_fft / sidecar), optional thread pool
4. **► RUN OPTIMIZED SPLIT** — batch process
5. **Review Segmentation** — manual boundary adjustment
6. **Clear** — reset UI state (does not delete outputs)

---

## 15. Tutorials

### Tutorial A — First batch split (GUI)

**Goal:** Split a folder of violin one-shots into ADSR stems.

1. **Install**

   ```bash
   pip install -e ".[dev]"
   ```

   Install [ffmpeg](https://ffmpeg.org/) for compressed formats.

2. **Prepare files**

   - Place `.wav` files in one folder, e.g. `D:\Samples\Violin\`.
   - Name with pitch hints: `Violin_A4_01.wav`, `Violin_G3_02.wav`.

3. **Start**

   ```bash
   python split_audio_segments.py
   ```

4. **Configure**

   - **Browse** → select folder.
   - **Auto-Detect Mean Length** (optional).
   - For ~2 s notes, use **Medium (1.5–3.0s)** → **Apply Preset**.

5. **Parameters**

   - **Smart Mode** on.
   - Attack 0.90, Decay 0.50.
   - Fade 50 ms, cosine.
   - Pitch Stability 5¢; tighten to 3¢ if sustains shrink too much.

6. **Run** → **► RUN OPTIMIZED SPLIT**

   Log example: `Att: 0.18s | Sus: 1.42s | Dec: 0.35s | PitchWin: 0.52-1.05s (σ=3.21¢)`

7. **Review**

   - Drag green (attack) and orange (decay) lines.
   - Arrow keys: 5 ms; Shift+arrow: 25 ms.

8. **Metadata**

   - Open `segmentation_metadata.csv` for QA.

---

### Tutorial B — Legato string library (CLI)

**Goal:** Batch-process long bowed notes with metadata for a sample library.

```bash
python split_audio_cli.py \
  --folder "D:/Samples/Violin_Legato" \
  --preset "Legato / Bow" \
  --pitch-refine-mode expand \
  --export-metadata
```

Expected behaviour: Smart mode, relaxed decay threshold (0.42), longer fades (55 ms), pitch tolerance 8¢.

---

### Tutorial C — Staccato / pluck articulations

**Goal:** Short attacks with minimal sustain.

```bash
python split_audio_cli.py \
  --folder "D:/Samples/Pizz" \
  --preset "Staccato / Pluck" \
  --fade-ms 25
```

Advanced mode activates automatically via preset (derivative + flux attack).

---

### Tutorial D — Spectral analysis pipeline (annotate mode)

**Goal:** Keep full energy-based sustains for STFT while recording stable pitch windows.

```bash
python split_audio_cli.py \
  --folder "D:/Analysis/LongBows" \
  --preset "Very Long (> 6.0s)" \
  --pitch-refine-mode annotate \
  --export-metadata
```

Read `pitch_stability.window_start` / `window_end` from JSON for analysis ROI.

Soft high brass (half-integer onset after a flat level):

```bash
python split_audio_cli.py -f ./trombone_pp \
  --preset soft_high_brass \
  --regime-mode trim \
  --flux-sidecar \
  --export-metadata
```

`_Sustains/` is unchanged. `_Sustains_Stable/` is the flux-stable window. `<stem>.flux.json` and `segments.regime_refine` record the cut and the floor decision.

---

### Tutorial E — Python scripting (single file)

```python
from pathlib import Path
import librosa
import soundfile as sf
import audio_segment_core as core

input_path = Path("Violin_A4.wav")
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

y, sr = librosa.load(input_path, sr=None)
cfg = core.SegmentConfig.from_preset("Medium (1.5-3.0s)")

result = core.detect_segments(y, sr, cfg, file_path=input_path)
parts, _, _, _ = core.extract_and_fade_segments(
    y, sr,
    result.t_att, result.t_dec, result.t_end,
    result.trim,
    fade_ms=50.0,
    fade_type="cosine",
)

for folder, audio in parts.items():
    if len(audio) == 0:
        continue
    tag = folder.strip("_").replace("Release_Silence", "Release")
    if folder == "_Full_Active_Sound":
        tag = "FullActive"
    target = out_dir / folder.strip("_")
    target.mkdir(parents=True, exist_ok=True)
    sf.write(target / f"{input_path.stem}_{tag}.wav", audio, sr)

print(f"Boundaries: att={result.t_att:.3f}s, dec={result.t_dec:.3f}s, end={result.t_end:.3f}s")
```

---

### Tutorial F — Mode selection guide

| Symptom | Suggested change |
|---------|------------------|
| Attack includes too much steady tone | Lower θ_att (e.g. 0.85) or enable Advanced |
| Decay starts too early on long bows | Lower θ_dec (0.40), Very Long preset, or annotate mode |
| Sustain too short for STFT (5–7 s) | **annotate** or **expand**; avoid **crop** |
| Short plucks get empty sustain | Very Short / Staccato preset; min sustain ~0.06 s |
| Noisy recordings, unstable splits | Pitch refine **off** or proportional-only |
| Multiple envelope peaks | Manual review; pre-trim files |

---

### Tutorial G — Running tests

```bash
cd "path/to/ADSR_Segmenter"
pytest
```

Unit tests across trim, smart ordering, vibrato robustness, annotate mode, regime refine, and benchmark smoke tests.

---

## 16. Boundary Benchmark

Reproducible evaluation against labeled ground truth:

```bash
python run_benchmark.py --generate-corpus   # 40 synthetic one-shots + 4 regime items
python run_benchmark.py                     # → benchmark/results/benchmark_report.txt
python run_benchmark.py --template my_labels.csv
python run_benchmark.py --annotations my_labels.csv --audio-dir D:/labeled
```

Metrics: MAE for \(t_{\mathrm{att}}\), \(t_{\mathrm{dec}}\), \(t_{\mathrm{end}}\) (ms); % within ±50 ms. See [§11.13](#1113-benchmark-error-metrics).

---

## 17. Testing

| Module | Coverage |
|--------|----------|
| `test_segment_detection.py` | Trim, attack energy, smart ordering, short sounds, ZC extract |
| `test_advanced_features.py` | Vibrato robust, Hann vs cosine, long-note guard, annotate, batch I/O |
| `test_benchmark.py` | Corpus generation, MAE aggregation, template |
| `test_regime_refine.py` | Steady tone, half-integer onset, floor refusal, annotate, sidecar |
| `test_regime_generalise.py` | Relative HI bands, dual walk, vibrato, level, pitch failure, unvoiced, floor scaling |

Key regression: 6 s note sustain ≥ 2.8 s; annotate mode preserves energy boundaries.

---

## 18. Troubleshooting

| Issue | Cause | Remedy |
|-------|-------|--------|
| `NoBackendError` | Missing ffmpeg / soundfile | Install ffmpeg; use WAV |
| No files found | Wrong folder | Files must be directly in folder |
| Clicks at edges | Cut away from ZC | Increase fade; cosine; manual nudge |
| Sustain too short | Pitch crop / tight refine | annotate or expand; Very Long preset |
| Soft brass still has “fizz” after attack | Unstable half-integer regime | Regime **trim** or `soft_high_brass`; or keep annotate + sidecar |
| `regime_refine.refused` is true | Candidate shorter than STFT floor | Lower `regime_analysis_n_fft`, or keep annotate and use the sidecar |
| `half_integer_valid` is false | No f₀, or \(\alpha f_0\) below bin resolution | Expected on E2-class notes at `n_fft=1024`; flux walk still runs |
| `pitch_refine.failed` / `unvoiced` | Tracker lost the note or the take is unpitched | Energy boundaries kept; regime uses flux only |
| `note_name_mismatch` | Filename spelling disagrees with tracked f₀ | Check wrap spellings (`B#4` = C5) and file naming |
| All segments similar length | Proportional-only | Enable Smart Mode |
| MP3 slow/fails | Codec | Convert to WAV for batch jobs |

---

## 19. Dependencies

| Package | Version | Role |
|---------|---------|------|
| `librosa` | ≥ 0.10 | Load, trim, RMS, STFT, YIN |
| `numpy` | ≥ 1.23 | Numerical arrays |
| `soundfile` | ≥ 0.12 | WAV/FLAC/AIFF/OGG write |
| `matplotlib` | ≥ 3.7 | Review waveform plots |
| `pytest` | ≥ 7.0 | Unit tests (dev) |

Python ≥ 3.10. Standard library: `tkinter`, `threading`, `json`, `csv`, `pathlib`, `logging`, `concurrent.futures`.

---

## Appendix — Quick constant lookup

```python
# audio_segment_core.py — detection defaults
SMART_ENERGY_BLEND = 0.7
SMART_PROP_BLEND = 0.3
DEFAULT_TRIM_DB = 60.0
DEFAULT_FRAME_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_ZERO_CROSSING_SEARCH_MS = 100.0
DEFAULT_PITCH_REFINE_MIN_FRACTION = 0.70
DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY = 0.75
DEFAULT_PITCH_FMIN_HZ = 30.0
DEFAULT_PITCH_FMAX_HZ = 4200.0
DEFAULT_REGIME_HI_REL_BANDWIDTH = 0.15
DEFAULT_REGIME_HI_RISE_DB = 10.0
```

---

*Document for ADSR_Segmenter v3.3.0. Synchronized with `audio_segment_core.py`, `ALL_PRESETS`, and `SegmentConfig`. Last updated: August 2026.*
