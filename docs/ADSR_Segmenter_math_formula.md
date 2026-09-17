# ADSR_Segmenter — Implementation-Faithful Mathematical Reference

**Document type:** formula catalogue for the production detector  
**Language:** English  
**Rendering:** Markdown + LaTeX (`$...$` inline, `$$...$$` display). No custom macros.  
**Project version:** 3.3.2 plus unreleased validation, parser, silence, and documentation fixes  
**Do not treat this file as a uniqueness proof.** Operational boundaries are heuristics.

---

## 0. Provenance and working-tree status

| Item | Value |
|------|--------|
| Repository | `E:\PYTHON CODES\Pacore de preparação dos sons\ADSR_Segmenter` |
| GitHub alignment SHA | `8ad432f0e2da366dba045732f6cf581e7a4bbfa9` |
| Branch | `fix/adsr-validation-and-documentation` (local only; not pushed) |
| Documented HEAD | `8ad432f` plus the files hashed in §0.1 (this refresh) |
| Generated | 2026-09-17 |

GitHub `origin/main` at generation time is `8ad432f`. This file describes the reviewed working tree immediately before the fix-branch commit.

### 0.1 SHA-256 of documented first-party Python (refreshed 2026-09-17)

| File | Lines | SHA-256 |
|------|------:|---------|
| `audio_segment_core.py` | 1871 | `0805cd12a81630a0b9453355ecbe87d61c0f4fe2b47dfe67c9934d50311e49a5` |
| `split_audio_cli.py` | 236 | `a2abfce9618bd86098b4ef1ad11599027ab1bfdedb7b21e9b7cb19221442b49e` |
| `split_audio_segments.py` | 1339 | `65b900500ac21f8f50d411651366e5b3b48c7d86a3ef06c1b8871b1d4d6e9ec7` |
| `run_benchmark.py` | 214 | `d9fc84bd002cf46c2e34c29639cf36e06cc464750d92a089b67f8545be9f3df2` |
| `benchmark/__init__.py` | 1 | `5a8a93cc01d0789009b18ccfcd9c60beb70071008dbb2f5e67c4bcad8e4635fb` |
| `benchmark/benchmark_core.py` | 367 | `b45174312b89060f4f66e940c951f130c423b4577bab5a32d4355f23c8404ca2` |
| `benchmark/generate_corpus.py` | 236 | `a2079e270c5b1f6e8f382d5515af57d6a6576c0ad1197b72f0958535d25ee172` |
| `installers/common/bootstrap.py` | 242 | `32f26734b05542f876d0fe63c4e5828e5293871f3e70eb91a2a1e0e87090390f` |
| `installers/common/config.py` | 72 | `3f60b8816166b49532a9ef60730a4ae7f44c7d6b2693712188926a55742d0789` |
| `tests/test_segment_detection.py` | 111 | `c2d5049bbf66e1b3c65c09cf909510980a8cd2c2cf2edb4bcd073c8465f5e21b` |
| `tests/test_advanced_features.py` | 203 | `e69de1f1453247d8965a144d55871a45db1d77ec811d92c972c69e3f7806e179` |
| `tests/test_benchmark.py` | 73 | `854e17557af011ebe7f9091232e4f4df0815e9c39e3e3ab73b665a4e9056f98c` |
| `tests/test_regime_refine.py` | 219 | `b77e90a751c0d08df5cde05d826cf02d2be5d535e7b944077dcd63e669c3b29f` |
| `tests/test_regime_generalise.py` | 339 | `9b59114b1c502b61a9697cd72a21281d920aa902def3d0a2c1a2caf11ffecc1f` |
| `tests/test_silence_and_policy.py` | 127 | `69ebd5e1777d0cba753f7ccf1046e53dfe4a5997e9a0841bdab7fd2dadaa3e7d` |

### 0.2 Coverage classification

| File | Role | Mathematics coverage |
|------|------|----------------------|
| `audio_segment_core.py` | Production DSP | Full (M-001…M-038, L-001…L-010) |
| `split_audio_cli.py` | Production CLI | Orchestration only; calls core |
| `split_audio_segments.py` | Production GUI | Same core detector; fade/export wrappers; no independent detector math |
| `run_benchmark.py` | Optional benchmark driver | No original DSP; writes default `benchmark/results/` if invoked that way |
| `benchmark/benchmark_core.py` | Optional benchmark | M-039 error metrics |
| `benchmark/generate_corpus.py` | Optional / test corpus | M-040, M-041 synthetic envelopes |
| `benchmark/__init__.py` | Package marker | None |
| `installers/common/bootstrap.py` | Installer | None (downloads Python) |
| `installers/common/config.py` | Installer | None |
| `tests/*.py` | Test-only | Exercise production maths; no separate production formulae |

### 0.3 Axes, units, and the two clocks

- Discrete time index $n$ is in **samples**. File time is $t = n / f_s$ seconds, $f_s$ = `sr`.
- Detection inside `detect_segments` uses **trim-relative** time $t'$ on the trimmed array, then $t = t_{\mathrm{start}} + t'$.
- RMS / flux / YIN live on a **frame** axis $k$ with hop $H$ = `hop_length` (default 512). Frame period $\Delta t = H/f_s$ (about $23.2\,\mathrm{ms}$ at $22\,050\,\mathrm{Hz}$, $11.6\,\mathrm{ms}$ at $44\,100\,\mathrm{Hz}$).
- Export times in `process_audio_file` are **zero-crossing-snapped sample indices** divided by $f_s$, not the raw detector floats.

---

## 1. Conceptual ADSR versus operational intervals

A classical synthesizer ADSR has an attack rise, a decay *down to a sustain level*, a gated sustain, and a release after note-off. This program does **not** estimate those four physical phases as unique acoustic events.

Operational outputs (absolute file seconds):

| Symbol | Code | Operational meaning |
|--------|------|---------------------|
| $t_{\mathrm{start}}$ | `trim.t_start` | Start of `librosa.effects.trim` region |
| $t_{\mathrm{att}}$ | `t_att` | End of the exported attack / start of sustain |
| $t_{\mathrm{dec}}$ | `t_dec` | End of sustain / start of the falling active tail |
| $t_{\mathrm{end}}$ | `t_end` / `trim.t_end` | End of the active (trimmed) region |
| release | $t > t_{\mathrm{end}}$ | Residual samples after the trim |

The detector **always constructs** $t_{\mathrm{att}} < t_{\mathrm{dec}} < t_{\mathrm{end}}$ when it can, using minimum-duration clamps. A recording without a plateau still receives a sustain interval.

---

## 2. Default analysis grain

Implemented in `audio_segment_core.py` lines 20–22.

```python
DEFAULT_TRIM_DB = 60.0
DEFAULT_FRAME_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
```

Timing tolerances in tests should be at least one hop, typically $2$–$3$ hops, because energy thresholds are evaluated on that grid.

Smart-mode blend constants (lines 26–27):

```python
SMART_ENERGY_BLEND = 0.7
SMART_PROP_BLEND = 0.3
```

---

## M-001 — DC removal

1. **Name / status:** Mean subtraction. Production, default on (`remove_dc=True`).
2. **Site:** `audio_segment_core.py`, `preprocess_signal`, lines 304–308.
3. **Excerpt:**

```python
def preprocess_signal(y: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    if not remove_dc or len(y) == 0:
        return y
    return y - float(np.mean(y))
```

4. **Formula:**

$$
\tilde{x}[n] = x[n] - \frac{1}{N}\sum_{m=0}^{N-1} x[m], \quad N=\mathrm{len}(y)
$$

5. **Symbols:** $x$ = `y` (dimensionless PCM after librosa load, typically $[-1,1]$); $N$ samples. Empty or `remove_dc=False` returns $x$ unchanged.
6. **Layman:** Subtract the average sample so a constant offset is not treated as sound.
7. **Specialist:** First-moment centering. Does not high-pass filter; very low-frequency musical content is largely preserved.
8. **Conditions:** Skipped if $N=0$ or `remove_dc` is false. No clipping.
9. **Downstream:** Fed to trim and all envelopes. Tests: `tests/test_segment_detection.py` (trim/detect on bursts).

---

## L-001 — Audio load (library)

- **Package:** `librosa.load`, version used in validation: `0.10.2.post1` (constraint `librosa>=0.10.0`).
- **Call sites:** `process_audio_file` line 1773; GUI `split_audio_segments.py` lines 589, 749, 1037; benchmark `benchmark_core.py` load site.
- **Arguments:** `librosa.load(path, sr=None)` — native rate. Default **`mono=True` is not overridden**, so stereo is downmixed.
- **Do not reproduce** librosa resampling/downmix internals.
- **Layman:** The file is read as one channel at its own sample rate.
- **Specialist:** `sr=None` disables resampling. Channel mix is librosa’s default mono reduction.
- **Project mathematics around the call:** none besides later DC removal.

---

## L-002 — Silence trim (library)

- **Package:** `librosa.effects.trim`.
- **Call site:** `trim_active_region`, line 311: `librosa.effects.trim(y, top_db=trim_db)` with default `trim_db=60`.
- **Layman:** Drop leading/trailing material that is far below the peak.
- **Specialist:** `top_db` is referenced to the peak of the *current* signal (librosa default behaviour). Do not treat $60\,\mathrm{dB}$ as an absolute SPL threshold.

---

## M-002 — Trim index to time and end safeguard

1. **Name / status:** Production.
2. **Site:** `trim_active_region`, lines 311–319.
3. **Excerpt:**

```python
    y_trimmed, index = librosa.effects.trim(y, top_db=trim_db)
    idx_start, idx_end = int(index[0]), int(index[1])
    t_start = idx_start / sr
    t_end_trimmed = idx_end / sr
    t_end_signal = len(y) / sr
    t_end = min(t_end_trimmed, t_end_signal - 1e-3)
    active_len = max(0.0, t_end - t_start)
```

4. **Formula:**

$$
t_{\mathrm{start}}=\frac{n_{\mathrm{start}}}{f_s},\quad
t_{\mathrm{end}}=\min\!\left(\frac{n_{\mathrm{end}}}{f_s},\; \frac{N}{f_s}-10^{-3}\right),\quad
L=\max(0,\, t_{\mathrm{end}}-t_{\mathrm{start}})
$$

5. **Symbols:** $n_{\mathrm{start}},n_{\mathrm{end}}$ = `idx_start`, `idx_end` (samples, exclusive end from librosa); $L$ = `active_len` (s).
6. **Layman:** Convert the kept region to seconds and leave at least one millisecond before the file’s last sample.
7. **Specialist:** The $1\,\mathrm{ms}$ cap is a safeguard, not an acoustic offset. `y_trimmed` is the array returned by librosa (may not match the $1\,\mathrm{ms}$-capped $t_{\mathrm{end}}$ exactly).
8. **Fallback:** If trim yields an empty/short region, `detect_segments` uses proportional fractions of $L$ (lines 1501–1507).
9. **Tests:** `test_trim_active_region`, `test_extract_starts_at_trim_not_file_start`.

---

## L-003 — RMS envelope (library)

- **Package:** `librosa.feature.rms`.
- **Call site:** `compute_rms_envelope`, line 322: `librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]`.
- **Defaults:** `frame_length=1024`, `hop_length=512`. Centered frames follow librosa defaults (not overridden).
- **Layman:** A moving loudness curve.
- **Specialist:** Project code uses the returned vector as-is. Frame times via L-005.

---

## L-005 — Frame times (library)

- **Package:** `librosa.times_like`.
- **Sites:** RMS line 322; flux line 330; YIN line 789.
- **Arguments:** `times_like(arr, sr=sr, hop_length=hop_length)`.
- **Layman:** Attach a clock to each analysis frame.
- **Specialist:** Flux is one frame shorter than the STFT (because of `diff`); `times_like(flux)` timestamps that shorter vector from $t=0$.

---

## M-003 — Half-wave-rectified spectral flux

1. **Name / status:** Production. Used in advanced attack (`normalised=False`) and regime refine (`normalised=True` by default).
2. **Site:** `compute_spectral_flux`, lines 330–352.
3. **Excerpt:**

```python
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    if normalised:
        magnitude = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-12)
    diff = np.diff(magnitude, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
```

4. **Formula:** Let $X[b,k]$ be the STFT (L-004). $|X|$ is optionally column-normalised:

$$
\hat{X}[b,k]=\frac{|X[b,k]|}{\sum_b |X[b,k]|+10^{-12}}
\quad\text{or}\quad \hat{X}=|X|
$$

$$
\Phi[k]=\sum_b \max\!\bigl(\hat{X}[b,k+1]-\hat{X}[b,k],\,0\bigr),\quad k=0,\ldots,K-2
$$

5. **Symbols:** $b$ bin, $k$ frame, $\Phi$ = `flux`. `n_fft` defaults to `frame_length`.
6. **Layman:** Measure how much the spectrum grows from one frame to the next. Ignoring decreases emphasises onsets.
7. **Specialist:** Standard half-wave spectral flux. Normalisation makes $\Phi$ approximately level-invariant. Advanced-mode attack keeps the unnormalised path so published v3.1 attack MAEs stay comparable.
8. **Safeguards:** $10^{-12}$ in the denominator. Empty/short signals handled by callers.
9. **Tests:** `test_spectral_flux_vectorized`; regime tests; advanced-mode tests.

---

## L-004 — STFT (library)

- **Package:** `librosa.stft`.
- **Sites:** flux in `330+`; half-integer ratio line 1022.
- **Arguments:** `n_fft`, `hop_length`; other librosa defaults (window, center) are **not** set in project code.
- **Do not reproduce** the STFT window formula here.

---

## M-004 — Odd-length moving median

1. **Name / status:** Production helper.
2. **Site:** `_moving_median`, lines 355–363.
3. **Excerpt:**

```python
    win = max(3, win | 1)
    half = win // 2
    out[i] = float(np.median(arr[lo:hi]))
```

4. **Formula:** Force odd width $W=\max(3,\, w\text{ OR }1)$. Then

$$
m[i]=\mathrm{median}\bigl\{a[j]: j\in[\max(0,i-\lfloor W/2\rfloor),\, \min(N, i+\lfloor W/2\rfloor+1))\bigr\}
$$

5. **Symbols:** `win` may be even; `| 1` sets the lowest bit. Units follow the input (cents, flux, dB).
6. **Layman:** Replace each point by the middle value of its neighbours so wiggles shrink.
7. **Specialist:** Causal/anti-causal centered median; edge windows shrink. Used for vibrato suppression and flux smoothing.
8. **Domain:** $W\ge 3$, odd.
9. **Tests:** `test_vibrato_robust_stability_lower_than_raw`.

---

## M-005 — Pitch-stability standard deviation (cents)

1. **Name / status:** Production.
2. **Site:** `pitch_stability_std_cents`, lines 366–389.
3. **Steps (no closed form for the whole routine):**
   - Keep finite cents samples $(t_i,c_i)$. If fewer than 3, return undefined (`None`).
   - If $\ge 3$ points, linear detrend: fit $c\approx \alpha (t-t_0)+\beta$ with `np.polyfit(..., 1)` and subtract.
   - If `vibrato_robust` and $\ge 5$ points, subtract M-004 of the residual with $W$ from `vibrato_median_window_s / median(\Delta t)` (default $0.12\,\mathrm{s}$).
   - Return $\mathrm{std}(c)$ (population `np.std` default, ddof=0).
4. **Residual after detrend:**

$$
c_i' = c_i - \hat\alpha (t_i-t_0)-\hat\beta
$$

5. **Symbols:** $c$ in **cents**; $t$ in seconds; output in cents.
6. **Layman:** After removing slow pitch drift and fast vibrato wobble, how much pitch still jumps around.
7. **Specialist:** Ordinary-least-squares linear trend; median high-pass for ~4–7 Hz vibrato. `np.std` uses $N$, not $N-1$.
8. **Undefined:** `<3` finite samples, or `<3` after masking. Default `vibrato_robust=True`.
9. **Downstream:** seed-window search and expand/crop. Tests: vibrato unit test; pitch refine tests.

---

## M-006 — Energy attack time

1. **Name / status:** Production. Smart mode and combined attack.
2. **Site:** `detect_attack_energy`, lines 392–406.
3. **Excerpt:**

```python
    peak_val = float(rms[peak_idx])
    if peak_val < 1e-12:
        return float(times[0])
    level = threshold * peak_val
    for i in range(search_end):
        if rms[i] >= level:
            return float(times[i])
```

4. **Formula:** Let $k^\star=\arg\max_k r[k]$ (`peak_idx` from caller). Search $k=0,\ldots,k^\star$:

$$
t_{\mathrm{att,E}}=\min\{ t[k]: k\le k^\star,\; r[k]\ge \theta_{\mathrm{att}}\, r[k^\star] \}
$$

If $r[k^\star]<10^{-12}$, return $t[0]$. If no crossing, return $t[\min(k^\star,K-1)]$.

5. **Symbols:** $r$ = RMS; $\theta_{\mathrm{att}}$ = `attack_threshold` (default $0.9$). Time in seconds (trim-relative).
6. **Layman:** Mark attack end at the first frame that has reached 90% of the loudest frame, looking only up to that peak.
7. **Specialist:** Relative threshold on the RMS trajectory. Not a physical “end of transient” identifier. Resolution $=\Delta t$.
8. **Defaults:** Medium preset $\theta_{\mathrm{att}}=0.90$.
9. **Tests:** `test_energy_attack_before_peak`, smart ordering tests.

---

## M-007 — Derivative / flux attack time

1. **Name / status:** Production, advanced mode.
2. **Site:** `detect_attack_derivative`, lines 409–429.
3. **Steps:**
   - $d[k]=r[k+1]-r[k]$.
   - $k_d=\arg\max_{k<k^\star} d[k]$, $t\leftarrow t[k_d]$.
   - If flux is provided, $t\leftarrow\min(t, t_\Phi[\arg\max \Phi])$ using the **global** flux peak.
   - $t\leftarrow\max(t,\, t[\lfloor 0.05 K\rfloor])$.
   - $t\leftarrow\min(t,\, 0.85\, t[k^\star])$.
4. **Layman:** Prefer the fastest loudness rise, but not at the very start and not after most of the way to the peak. If the spectrum jumps later, take the earlier of the two clocks.
5. **Specialist:** Heuristic, not a unique onset. Global $\arg\max\Phi$ can sit after the attack; `min` then keeps the earlier derivative time.
6. **Tests:** advanced-mode tests in `tests/test_advanced_features.py`.

---

## M-008 — Combined attack

1. **Status:** Production.
2. **Site:** `detect_attack_combined`, lines 432–445.
3. **Formula:** If `use_derivative` is false, return M-006. Else

$$
t_{\mathrm{att}}=\min(t_{\mathrm{att,E}},\, t_{\mathrm{att,D}})
$$

---

## M-009 — Normalised ADSR fractions

1. **Site:** `_normalized_adsr_fractions`, lines 448–454.
2. **Formula:** $S=p_A+p_S+p_D$. If $S\le 0$, return $(0.15,0.60,0.25)$. Else $(p_A/S,\,p_S/S,\,p_D/S)$.
3. **Layman:** Treat the three percentages as a pie chart even if they do not add to 100%.
4. **Note:** The short-signal branch in `detect_segments` (lines 1501–1507) uses **raw** `attack_pct` / `sustain_pct` and does **not** call this helper.

---

## M-010 — Earliest allowed decay time

1. **Site:** `min_decay_time_proportional`, lines 457–462.
2. **Formula:**

$$
t_{\mathrm{dec,min}} = L\cdot\bigl(f_A + f_S\cdot \rho\bigr)
$$

with $(f_A,f_S,f_D)$ from M-009 and $\rho=$ `sustain_fraction_before_decay` (default $0.75$).
3. **Layman:** Do not start “decay” until at least three-quarters of the planned sustain slice has elapsed.
4. **Tests:** smart/advanced decay behaviour; regime tests keep energy sustain when annotate.

---

## M-011 — Energy decay time

1. **Site:** `detect_decay_energy`, lines 465–486.
2. **Formula:** $R_{\max}=\max_k r[k]$. Search from $\max(k_{\mathrm{att}},k^\star)$ (and not before $t_{\mathrm{dec,min}}$):

$$
t_{\mathrm{dec,E}}=\min\{ t[k]: r[k]\le \theta_{\mathrm{dec}} R_{\max} \}
$$

If none, return $t[\lfloor 0.85 K\rfloor]$. If $R_{\max}<10^{-12}$, return $t[-1]$.
3. **Default** $\theta_{\mathrm{dec}}=0.5$.
4. **Layman:** Decay starts when loudness has fallen to half the peak (after the peak and after the sustain guard).
5. **Specialist:** This is **not** the synthesizer “decay to sustain level”. It is an offset-like energy crossing after the peak.

---

## M-012 — Derivative decay time

1. **Site:** `detect_decay_derivative`, lines 489–510.
2. **Steps:** Start search after $t[k^\star]+\max(0.05,\, 0.15(t[-1]-t[k^\star]))$. Count consecutive negative $d[k]$. When the count reaches 3, return $t[k-2]$. Else fall back to M-011.
3. **Layman:** Look for three frames in a row that get quieter, then mark the start of that run.
4. **Specialist:** Run-length heuristic on the first difference. Not an exponential time-constant fit.

---

## M-013 — Proportional boundaries

1. **Site:** `detect_segments_proportional`, lines 513–533.
2. **Formula:** Normalise percentages as in M-009. Then

$$
t_{\mathrm{att}}=L f_A,\quad
t_{\mathrm{dec}}=L(f_A+f_S)
$$

Adjust: $t_{\mathrm{dec}}\leftarrow\max(t_{\mathrm{att}}+\max(d_{\min},\, 0.4 L f_S),\, t_{\mathrm{dec}})$, then $t_{\mathrm{dec}}\leftarrow\min(t_{\mathrm{dec}},\, L-0.05L)$. If order breaks, $t_{\mathrm{dec}}\leftarrow\min(t_{\mathrm{att}}+L f_S,\, L-0.02)$.
3. **Layman:** Cut the kept sound into attack / sustain / decay by the preset pie chart, then nudge so sustain is not tiny.
4. **Tests:** `test_proportional_percentages_sum`.

---

## M-014 — Sustain plateau flag

1. **Site:** `detect_sustain_plateau`, lines 536–556.
2. **Formula:** On $r[k_{\mathrm{att}}:k_{\mathrm{dec}}]$,

$$
v=\frac{\mathrm{Var}(r)}{(\bar r)^2}
$$

Accept the interval if $v<\theta_v$ (default $0.2$) and duration $\ge d_{\min}$. Else return `(None, None)`.
3. **Layman:** If the sustain slice is both long enough and not too wrinkly, keep those frames.
4. **Used only in advanced mode** after derivative times (`detect_segments_advanced_rel`, lines 1427–1455).

---

## M-015 — Effective minimum sustain

1. **Site:** `effective_min_sustain_duration`, lines 559–572.
2. **Formula:**

$$
d_{\mathrm{frames}}=\frac{N_{\mathrm{min,frames}}\, H}{f_s},\quad
d=\max(d_{\mathrm{cfg}},\, T_{\mathrm{pitch}},\, d_{\mathrm{frames}})
$$

If $0<L<d$, then $d\leftarrow\max(0.25L,\, d_{\mathrm{frames}},\, 0.02)$.
3. **Defaults:** $N_{\mathrm{min,frames}}=40$, $H=512$, $T_{\mathrm{pitch}}=0.5\,\mathrm{s}$. At $22\,050\,\mathrm{Hz}$, $d_{\mathrm{frames}}\approx 0.929\,\mathrm{s}$, which **dominates** the Medium $0.35\,\mathrm{s}$ preset.
4. **Layman:** Demand a sustain at least 40 hops long, unless the file is shorter than that demand.
5. **Specialist:** The 40-hop term is a **preferred minimum that is clamped**, not a reject-the-file gate. The short-file branch cannot go below $d_{\mathrm{frames}}$, so at $f_s\le 22\,050\,\mathrm{Hz}$ it does not relax that grain. Detector numbers were not changed.
6. **Tests:** short-sound test still expects $\ge 0.02\,\mathrm{s}$ sustain.

---

## M-015b — Silence / export rejection

1. **Site:** `describe_rejection`, lines 575–587.
2. **Rule:** After the same DC removal as detection, if $\max|\tilde x|<10^{-8}$ or `trim.active_len` $\le 10^{-6}$, return `no_active_energy`. Else if M-032 fails, return `invalid_segment_boundaries`. Else accept.
3. **Layman:** All-zero files are invalid scientific input even when clamps invent ordered times.
4. **Downstream:** `process_audio_file` raises `ValueError`; CLI `batch_process_folder` records the error and continues; GUI `process_file` records `batch_failures` and continues. CLI exit `2` if any file fails.
5. **Tests:** `tests/test_silence_and_policy.py`.

---

## M-015c — Boundary-policy metadata

1. **Site:** `_boundary_policy`, lines 590–607; attached on every `SegmentResult`.
2. **Fields:** `decay_definition=energy_threshold_after_peak`; `min_sustain_s` (effective); `min_sustain_cfg_s`; `min_sustain_frames_s`; `active_len_s`; `sustain_assignment` (`detector` / `operational_clamp` / `operational_proportional` / `pitch_refined` / `fallback_proportional`); `frame_floor_limits_short_file`.
3. **Layman:** Say whether the sustain was measured or assigned, and that “decay” means an energy offset.
4. **Does not change detector numbers.**

---

## M-016 — Filename note parse

1. **Site:** `parse_note_from_filename`, lines 610–640 (boundary regex at 626).
2. **Excerpt:**

```python
    match = re.search(r"(?:^|[^A-Za-z])([A-Ga-g])(#|b)?(\d+)", path.stem)
```

3. **Procedure:** First match of pitch-class + optional `#`/`b` + octave, **not** preceded by a letter. Convert with L-007. Wrap spellings `{B#, Cb, E#, Fb}` set `note_name_wrap_spelling`.
4. **Layman:** Read `A4` out of `Violin_A4.wav`. Do not treat `Bagpipe1` as note E1.
5. **Specialist:** Regular-language heuristic, not OCR of a score. First match wins.
6. **Tests:** `test_parse_note_from_filename` (A4, B#4, `seg1`, `Bagpipe1`).

---

## L-007 — Note name to hertz (library)

- **Package:** `librosa.note_to_hz`.
- **Site:** `parse_note_from_filename`, line 636.
- **Layman:** Turn `A4` into $440\,\mathrm{Hz}$ (equal temperament as implemented by librosa).

---

## M-017 — Next power of two

1. **Site:** `next_pow2`, lines 648–653.
2. **Formula:** Smallest $p=2^m$ with $p\ge \lceil n\rceil$ and $p\ge 1$.

---

## M-018 — YIN frame length from $f_{\min}$

1. **Site:** `pitch_frame_length_for_fmin`, lines 656–659.
2. **Formula:**

$$
N_{\mathrm{YIN}}=\mathrm{clip}\bigl(\mathrm{next\_pow2}(4 f_s / f_{\min}),\, [64,\, 8192]\bigr)
$$

3. **Layman:** Low notes need a longer analysis window so the tracker can see several periods.
4. **Used when** a filename note, `expected_note_hz`, or explicit `search_range` is present; unnamed files keep `frame_length=1024` (comment: preserve v3.2 expand-mode benchmark rows).

---

## M-019 — Pitch search range

1. **Site:** `pitch_search_range`, lines 662–683.
2. **Formula:** If a center $f_c$ exists and no explicit range:

$$
f_{\min}=\max(f_{\min}^{\mathrm{cfg}},\, 1.06\cdot f_c/2),\quad
f_{\max}=\min(f_{\max}^{\mathrm{cfg}},\, f_c\cdot 2/1.06)
$$

Defaults $f_{\min}^{\mathrm{cfg}}=30\,\mathrm{Hz}$, $f_{\max}^{\mathrm{cfg}}=4200\,\mathrm{Hz}$. The factor $1.06$ keeps the exact subharmonic $f_c/2$ outside the legal interval.
3. **Layman:** Listen about one octave around the named note, but do not allow the exact octave below.

---

## M-020 — Expand a pitch-stable seed (iterative)

1. **Site:** `_expand_stable_pitch_window`, lines 686–709.
2. **Objective:** Grow $[k_{\mathrm{lo}},k_{\mathrm{hi}})$ while M-005 of the window stays $\le 1.25\,\theta_{\mathrm{cents}}$ (`pitch_stability_cents * 1.25`).
3. **Steps:** Decrement `lo` while the condition holds; increment `hi` while it holds; stop at array edges or first failure.
4. **Layman:** Stretch the steady-pitch island until the pitch gets too restless.
5. **No closed form.**

---

## M-021 — Pitch-based sustain refinement (iterative)

1. **Site:** `refine_sustain_by_pitch`, lines 712–949.
2. **Objective:** Optionally replace energy sustain $[t_{\mathrm{att}}',t_{\mathrm{dec}}']$ by a pitch-stable sub-window.
3. **Constraints / fallbacks (keep energy bounds):** `use_pitch_refine=False`; sustain shorter than M-015; YIN exception; $<3$ finite $f_0$; voiced fraction $<$ `pitch_min_voiced_fraction` (default $0.5$); median $f_0$ more than $300\,\mathrm{¢}$ from the expected note; best window std $>$ `pitch_fail_cents` (default $50$); refined duration $<$ `pitch_refine_min_fraction` of energy sustain (default $0.70$); mode `annotate`.
4. **Steps:**
   1. YIN on the energy-sustain slice (L-006) with M-018/M-019.
   2. Median $f_0$ on finite positive frames. Keep frames within $200\,\mathrm{¢}$ of that median for voicing.
   3. Cents vs median and vs expected note (M-022).
   4. Slide a window of duration $\max(T_{\mathrm{pitch}}, \min(0.5,d_{\min}))$; score $=\sigma_{\mathrm{¢}} + \overline{|c_{\mathrm{note}}|}$.
   5. Mode `expand`: M-020. Mode `crop`: keep seed. Mode `annotate`: record window, return energy times.
5. **Layman:** If the pitch sits still on one note, trim (or just annotate) the sustain to that still part.
6. **Tests:** `tests/test_regime_generalise.py`, `tests/test_advanced_features.py`.

---

## M-022 — Cents

Used in pitch refine (lines 815, 826, 839–842):

$$
c(f,f_{\mathrm{ref}})=1200\log_2(f/f_{\mathrm{ref}})
$$

implemented as `1200.0 * np.log2(...)`. Floor $f$ at $10^{-12}$ when comparing to the median.

---

## M-023 — Voiced-fraction gate

$$
\text{voiced\_frac}=\frac{\#\{k: f_0[k]\text{ finite},\ |c(f_0[k],f_{\mathrm{med}})|<200\}}{K}
$$

Fail if $<0.5$ (`pitch_min_voiced_fraction`).

---

## L-006 — YIN (library)

- **Package:** `librosa.yin`.
- **Site:** `refine_sustain_by_pitch`, YIN call at line 789.
- **Arguments:** `fmin`, `fmax`, `sr`, `frame_length=pitch_n`, `hop_length=cfg.hop_length`. Other YIN defaults (trough threshold, etc.) are librosa’s.
- **Do not reproduce** de Cheveigné & Kawahara internals.
- **Layman:** Guess the sung/played note from repeating waveform cycles.

---

## M-024 — Half-integer analysis $n_{\mathrm{FFT}}$

1. **Site:** `resolve_hi_n_fft`, lines 962–992.
2. **Need:** $2 f_s / n_{\mathrm{FFT}} \le \alpha f_0$ i.e. $n_{\mathrm{FFT}}\ge 2 f_s/(\alpha f_0)$, $\alpha=$ `regime_hi_rel_bandwidth` (default $0.15$).
3. **Then:** next power of two; raise to `pitch_frame_length` if larger; cap $16384$; cap $\le \lfloor N_{\mathrm{sus}}/4\rfloor$ (power-of-two after that cap).
4. **Layman:** Pick an FFT long enough that a band $\pm 0.15 f_0$ covers more than two bins.
5. **Tests:** `tests/test_regime_generalise.py`.

---

## M-025 — Half-integer band ratio

1. **Site:** `compute_half_integer_ratio_db`, lines 995–1044.
2. **Formula:** Bandwidth $w=\alpha f_0$. Band energy $E(f_c)=\sum_{b: |F[b]-f_c|\le w} |X[b,k]|^2$. Then

$$
\rho_{\mathrm{dB}}[k]=10\log_{10}\frac{\max(E(1.5 f_0)+E(2.5 f_0),\,10^{-20})}{E(f_0)}
$$

when $E(f_0)>10^{-20}$; else NaN.
3. **Invalid if** no $f_0$, or $w < 2 f_s/n_{\mathrm{FFT}}$ (`band_below_resolution`).
4. **Layman:** Compare energy sitting at one-and-a-half and two-and-a-half times the note to energy at the note itself.
5. **Specialist:** Relative inharmonic / half-integer diagnostic, not a physical “brassiness” uniqueness theorem. Frequencies from L-008.
6. **Tests:** regime refine / generalise.

---

## L-008 — FFT bin frequencies (library)

- **Package:** `librosa.fft_frequencies`.
- **Site:** line 1022: `librosa.fft_frequencies(sr=sr, n_fft=n_fft)`.

---

## M-026 — Regime duration floor

1. **Site:** `effective_regime_floor`, lines 1062–1069.
2. **Formula:**

$$
T_{\mathrm{floor}}=\max\bigl(T_{\mathrm{min}},\, N_{\mathrm{win}}\, n_{\mathrm{FFT}} / f_s\bigr)
$$

Defaults $T_{\mathrm{min}}=1.0\,\mathrm{s}$, $N_{\mathrm{win}}=8$. `n_fft` from `resolve_analysis_n_fft` (config, else pitch frame, else 1024).
3. **Layman:** Do not *apply* a regime trim if the remaining island would be shorter than about a second or eight analysis windows.
4. **Specialist:** The floor gates **trim application only**. Diagnostics are still filled (`span_below_floor`).

---

## M-027 — Inward walk (iterative)

1. **Site:** `_walk_inward`, lines 1130–1138.
2. **Steps:** Increment `start_i` while `above_fn` is true; decrement `end_i` while true and `end_i>start_i`.
3. **Used for** flux $v>\theta$ and HI $(v-\rho_{\mathrm{ref}})>\Delta_{\mathrm{dB}}$.
4. **No closed form.**

---

## M-028 — Spectral-regime refinement (iterative)

1. **Site:** `refine_sustain_by_regime`, lines 1151–1184.
2. **Objective:** Find a more stationary sub-interval of the pitch-stage sustain using normalised flux and optional HI rise.
3. **Steps:**
   1. Flux on the full trim (M-003, `normalised=True` by default). Restrict to $[t_{\mathrm{att}}',t_{\mathrm{dec}}']$.
   2. Vibrato median (M-004) then median of width `regime_flux_median_frames` (default 9).
   3. Reference = median of the central `regime_reference_fraction` (default $0.5$).
   4. Threshold $\theta = r_{\mathrm{applied}}\cdot \max(\mathrm{ref},10^{-12})$ with $r_{\mathrm{applied}}=1.5$ (normalised) or `regime_flux_ratio=2.0` (unnormalised).
   5. M-027 on flux. Repeat for HI with rise $\Delta=10\,\mathrm{dB}$ above mid-sustain HI.
   6. Combined candidate: $t_{\mathrm{att}}^{\mathrm{c}}=\max(t_{\mathrm{flux}},t_{\mathrm{HI}})$, $t_{\mathrm{dec}}^{\mathrm{c}}=\min(t_{\mathrm{flux}},t_{\mathrm{HI}})$.
   7. If input span or candidate span $<T_{\mathrm{floor}}$, refuse trim.
   8. Mode `annotate`: return original times. Mode `trim`: return candidate.
4. **Layman:** If the start or end of the sustain is spectrally “busier” than the middle, optionally shave those ends.
5. **Tests:** `tests/test_regime_refine.py`, `tests/test_regime_generalise.py`. Iowa trombone fixture tests skip when the optional AIFF is absent.

---

## M-029 — Relative boundary clamp

1. **Site:** `_clamp_segment_rel`, lines 1414–1424.
2. **Formula:** $t_{\mathrm{tail}}=\max(0.02,0.01)$. Then

$$
t_{\mathrm{att}}\leftarrow \mathrm{clip}(t_{\mathrm{att}},\, [0,\, L-d_{\min}-t_{\mathrm{tail}}]),\quad
t_{\mathrm{dec}}\leftarrow \mathrm{clip}(t_{\mathrm{dec}},\, [t_{\mathrm{att}}+d_{\min},\, L-t_{\mathrm{tail}}])
$$

with extra repairs if order collapses.
3. **Layman:** Force a sustain long enough and leave a little tail, even if the detectors disagree.
4. **This is the mechanism that invents a sustain on no-sustain signals.**

---

## M-030 — Smart blend

1. **Site:** `detect_segments_smart_rel`, lines 1458–1480.
2. **Formula:**

$$
t_{\mathrm{att}}=0.7\, t_{\mathrm{att,E}}+0.3\, t_{\mathrm{att,P}},\quad
t_{\mathrm{dec}}=0.7\, t_{\mathrm{dec,E}}+0.3\, t_{\mathrm{dec,P}}
$$

then $t_{\mathrm{dec}}\leftarrow\max(t_{\mathrm{dec}}, t_{\mathrm{dec,min}})$ and M-029.
3. **Layman:** Trust the loudness curve most of the way, but keep a 30% pull toward the preset pie chart.
4. **Tests:** `test_detect_segments_smart_ordering`.

---

## M-031 — `detect_segments` orchestration

1. **Site:** `detect_segments`, lines 1483–1580.
2. **Steps:** preprocess → trim → if too short, raw-percentage proportional times → else advanced / smart / proportional → pitch refine → clamp → optional regime refine → convert to file time → squeeze $t_{\mathrm{dec}}$ by $\max(0.02,0.05L)$ from $t_{\mathrm{end}}$ → squeeze $t_{\mathrm{att}}$ to keep min sustain.
3. **Exception fallback:** proportional on the *untrimmed* file (lines 1570–1580).
4. **Layman:** Run the recipe in order; if something throws, fall back to the pie chart on the whole file.

---

## M-032 — Segment validation

1. **Site:** `validate_segments`, lines 1583–1590.
2. **Formula:** Require $t_{\mathrm{att}}<t_{\mathrm{dec}}<t_{\mathrm{end}}$ and both $(t_{\mathrm{dec}}-t_{\mathrm{att}})$ and $(t_{\mathrm{end}}-t_{\mathrm{dec}})$ $\ge d_{\mathrm{val}}$ (default $0.01\,\mathrm{s}$).
3. **`process_audio_file` raises `ValueError` if this fails** (via `describe_rejection` in `process_audio_file`, lines 1759–1777). CLI records the error; GUI records `batch_failures` and continues.

---

## M-033 — Zero-crossing snap

1. **Site:** `find_zero_crossing`, lines 1593–1622.
2. **Steps:** Search $\pm 100\,\mathrm{ms}$ (default) for `signbit` changes. If none, expand to $\min(200\,\mathrm{ms}, N/4)$. Linear interpolate $t=-y_1/(y_2-y_1)$ on the nearest crossing; `round` to a sample.
3. **Layman:** Move the cut to the nearest place the waveform crosses zero so the splice clicks less.
4. **Specialist:** `numpy.signbit` treats $+0$ as non-negative. Interpolation is first-order.

---

## M-034 / M-035 — Fade curves and application

1. **Sites:** `_fade_curve` 1625–1636; `apply_fades` 1639–1657.
2. **Length:**

$$
N_{\mathrm{req}}=\lfloor f_s T_{\mathrm{ms}}/1000\rfloor,\quad
N_{\mathrm{floor}}=\min(\lfloor f_s/20\rfloor,\, \lfloor N/4\rfloor),\quad
N_{\mathrm{fade}}=\min(\max(N_{\mathrm{req}},N_{\mathrm{floor}}),\, \lfloor N/2\rfloor)
$$

$\lfloor f_s/20\rfloor$ is $50\,\mathrm{ms}$ at any $f_s$.
3. **Shapes** ($u=m/(N-1)$ via `linspace(0,1,N)`):
   - linear: $u$ / $1-u$
   - cosine: $\tfrac12(1-\cos\pi u)$ / $\tfrac12(1+\cos\pi u)$
   - hann: first / second half of `np.hanning(2N)` (L-010)
4. **After multiply:** if $|y[0]|<0.001$ set $0$; same for the last sample.
5. **Release folder is not faded.**
6. **Tests:** `test_hann_differs_from_cosine`.

---

## L-010 — Hann window (library)

- **Package:** `numpy.hanning`.
- **Site:** `_fade_curve`, line 1632.
- **Do not reproduce** NumPy’s endpoint convention beyond: project uses `hanning(2n)[:n]` and `hanning(2n)[n:]`.

---

## M-036 — Edge-click severity

1. **Site:** `edge_click_severity`, lines 1660–1676.
2. **Formula:** Compare $|y[0]|$, $|y[-1]|$, and max $|\Delta y|$ on the first/last `edge` samples to the median $|\Delta y|$ of the interior. Return the max of those ratios (end spikes divided by $2\mathrm{ref}$).
3. **`verify_no_clicks`:** $|y[0]|,|y[-1]|\le 0.01$ and severity $\le 4$. Failed segments are faded again at $1.5\times$ `fade_ms`.

---

## M-037 — Extraction indices

1. **Site:** `extract_and_fade_segments`, lines 1687–1727.
2. **Formula:** $n=\mathrm{clip}(\lfloor t f_s\rfloor,\, \text{ordered chain})$. Then M-033 on attack/decay/end. Decay snap is forced at least $20\,\mathrm{ms}$ after the snapped attack.
3. **Partitions:**
   - `_Attacks`: $[n_{\mathrm{start}}, n_{\mathrm{att}})$
   - `_Sustains`: $[n_{\mathrm{att}}, n_{\mathrm{dec}})$
   - `_Decays`: $[n_{\mathrm{dec}}, n_{\mathrm{end}})$
   - `_Release_Silence`: $[n_{\mathrm{end}}, N)$ (no fade)
   - `_Full_Active_Sound`: $[n_{\mathrm{start}}, n_{\mathrm{end}})$
4. **Empty segments:** writing skips `len==0` (line 1793). Half-open sample intervals; a zero-length hop is possible if clamps collide.
5. **Regime trim** writes `_Sustains_Stable` from `result.t_att/t_dec` while standard folders use pre-regime `source_att/source_dec` (lines 1779–1815).

---

## M-038 — Reported durations

1. **Site:** `process_audio_file`, lines 1829–1847.
2. **Formula:**

$$
t_{\mathrm{att}}^{\mathrm{out}}=n_{\mathrm{att}}/f_s,\quad
d_{\mathrm{att}}=(n_{\mathrm{att}}-n_{\mathrm{start}})/f_s,\quad \ldots
$$

so $t_{\mathrm{att}}^{\mathrm{out}}=t_{\mathrm{start}}+d_{\mathrm{att}}$ up to integer sample rounding.
3. **Tests / validation:** synthetic `sample_time_conversion` (external fixtures).

---

## L-009 — Audio write (library)

- **Package:** `soundfile.write`.
- **Site:** `write_audio`, lines 1740–1747. Format map: WAV/AIFF/FLAC/OGG from suffix; otherwise soundfile default.
- **Subtype** (PCM bit depth) is **not** set; library default applies.

---

## M-039 — Benchmark boundary MAE

1. **Site:** `benchmark/benchmark_core.py`, `boundary_errors_ms`, lines 134–141; aggregates 283–287.
2. **Formula:**

$$
e_{\mathrm{att}}=1000\,|t_{\mathrm{att}}^{\mathrm{pred}}-t_{\mathrm{att}}^{\mathrm{gt}}|
$$

(and likewise for $t_{\mathrm{dec}}$, $t_{\mathrm{end}}$), in milliseconds. MAE is the arithmetic mean over samples. “Within tolerance” if $\max(e_{\mathrm{att}},e_{\mathrm{dec}},e_{\mathrm{end}})\le 50\,\mathrm{ms}$ by default.
3. **Layman:** How many milliseconds the cuts missed the labelled cuts, on average.
4. **Specialist:** Labels are synthesizer-style envelope breakpoints (`generate_corpus`), **not** re-derived operational energy crossings. Comparing them to $t_{\mathrm{dec}}$ (energy threshold) is an evaluation convention.
5. **Do not run** the in-repo `benchmark/corpus` or overwrite `benchmark/results/` as part of ordinary validation.

---

## M-040 — Synthetic corpus envelope

1. **Site:** `benchmark/generate_corpus.py`, `_synthesize`, lines 70–148.
2. **Formula:** Linear attack $0\to 1$, sustain $1$, linear decay $1\to 0$, plus leading/trailing zeros. Optional vibrato (M-041), attack noise, brass/flute harmonics, unstable onset, burst.
3. **Ground-truth:** $t_{\mathrm{att}}=t_{\mathrm{gap}}+T_A$, $t_{\mathrm{dec}}=t_{\mathrm{att}}+T_S$, $t_{\mathrm{end}}=t_{\mathrm{dec}}+T_D$.
4. **Layman:** Build a toy note whose intended cuts are known because we drew them.
5. **Tests:** `tests/test_benchmark.py` writes into a **temporary** directory.

---

## M-041 — Corpus vibrato

1. **Site:** `generate_corpus.py`, lines 95–98.
2. **Formula:** $f(t)=f_0\, 2^{(\delta/1200)\sin(2\pi f_{\mathrm{vib}} t)}$, phase $=\sum 2\pi f(t)/f_s$.
3. **Same construction** as `tests/test_advanced_features.py` `_vibrato_burst`.

---

## 3. GUI / CLI mathematics

`split_audio_cli.py` and `split_audio_segments.py` do not implement a second detector. They call `detect_segments` / `extract_and_fade_segments` / `write_audio`.

GUI-only notes (not separate M-ids):

- `_config_from_ui` (`split_audio_segments.py` 127–148) builds `SegmentConfig` from widgets; unspecified fields use dataclass defaults (pitch refine on, regime annotate defaults).
- `DEFAULT_MIN_SUSTAIN_DURATION` on the GUI class is **not** read by `_config_from_ui`. The widget default is $0.35\,\mathrm{s}$ (Medium). The leftover constant was corrected to $0.35$ so it does not imply a $1.0\,\mathrm{s}$ detector floor.
- Before **Run**, `_snapshot_run_settings` copies widget values on the UI thread. The worker uses that snapshot and must not call Tk `.get()`.
- The GUI writes beside the source folder; there is no separate output-folder control.
- Review plot label: “Decay (energy offset)”.


---

## 4. Omissions and limitations

- Internal formulae of librosa YIN, STFT window, RMS centering, `effects.trim`, and soundfile codecs are **not** restated (L-ids only).
- No closed-form “true ADSR” inversion exists in this codebase; iterative walks are documented as procedures.
- `effective_min_sustain_duration`’s short-file branch cannot undercut the 40-hop floor (M-015). Policy: clamp / assign, do not reject the file. Detector numbers unchanged.
- Installer scripts contain no DSP mathematics.
- Optional Iowa trombone AIFF under `tests/fixtures/` was **not** present; those tests were skipped.
- This document describes the **current working tree**, not every historical export in the wild.

---

## 5. References actually consulted

- Working tree modules listed in §0.1 (read in full for `audio_segment_core.py`; targeted reads for CLI, GUI, benchmark, tests, installers).
- `README.md`, `docs/TECHNICAL_MANUAL.md` §§5–12, `docs/REGIME_REFINE_NOTES.md`, `CHANGELOG.md`, `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`.
- librosa $0.10.2.post1$ in the reused sibling venv; librosa $0.11.0$ in the isolated sibling venv (declared `>=0.10.0`). Not librosa source.
- Unit tests under `tests/` and synthetic checks written **outside** this repository.

No external web service was used to render this file.
