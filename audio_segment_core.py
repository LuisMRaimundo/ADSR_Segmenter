"""
Pure audio segmentation logic (no GUI).
Times are trim-relative inside detection, converted to absolute file times at the end.
"""

from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRIM_DB = 60.0
DEFAULT_FRAME_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_SUSTAIN_FRAMES = 40
DEFAULT_SUSTAIN_VARIANCE_THRESHOLD = 0.2
DEFAULT_ZERO_CROSSING_SEARCH_MS = 100.0
SMART_ENERGY_BLEND = 0.7
SMART_PROP_BLEND = 0.3
DEFAULT_VIBRATO_MEDIAN_WINDOW_S = 0.12
DEFAULT_PITCH_REFINE_MIN_FRACTION = 0.70
DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY = 0.75  # min % through proportional sustain before decay
DEFAULT_PITCH_FMIN_HZ = 30.0
DEFAULT_PITCH_FMAX_HZ = 4200.0
DEFAULT_PITCH_FAIL_CENTS = 50.0
DEFAULT_PITCH_OCTAVE_FAIL_CENTS = 300.0
DEFAULT_PITCH_MIN_VOICED_FRACTION = 0.5
DEFAULT_REGIME_HI_REL_BANDWIDTH = 0.15
DEFAULT_REGIME_HI_RISE_DB = 10.0
DEFAULT_REGIME_MIN_WINDOWS = 8
DEFAULT_REGIME_FLUX_RATIO_NORMALISED = 1.5
DEFAULT_HI_N_FFT_FLOOR = 4096
NOTE_WRAP_SPELLINGS = {("B", "#"), ("C", "b"), ("E", "#"), ("F", "b")}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a", ".wma", ".mp4", ".mka",
}

OUTPUT_FOLDERS = (
    "_Attacks",
    "_Sustains",
    "_Decays",
    "_Release_Silence",
    "_Full_Active_Sound",
)

PRESETS = {
    "Very Short (< 0.5s)": {
        "attack_pct": 0.20,
        "sustain_pct": 0.50,
        "decay_pct": 0.30,
        "fade_ms": 30.0,
        "min_sustain_duration": 0.06,
        "attack_threshold": 0.85,
        "decay_threshold": 0.45,
    },
    "Short (0.5-1.5s)": {
        "attack_pct": 0.15,
        "sustain_pct": 0.60,
        "decay_pct": 0.25,
        "fade_ms": 40.0,
        "min_sustain_duration": 0.15,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
    "Medium (1.5-3.0s)": {
        "attack_pct": 0.12,
        "sustain_pct": 0.65,
        "decay_pct": 0.23,
        "fade_ms": 50.0,
        "min_sustain_duration": 0.35,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
    "Long (3.0-6.0s)": {
        "attack_pct": 0.10,
        "sustain_pct": 0.70,
        "decay_pct": 0.20,
        "fade_ms": 60.0,
        "min_sustain_duration": 0.60,
        "attack_threshold": 0.90,
        "decay_threshold": 0.45,
        "pitch_refine_mode": "expand",
        "pitch_refine_min_fraction": 0.72,
    },
    "Very Long (> 6.0s)": {
        "attack_pct": 0.08,
        "sustain_pct": 0.75,
        "decay_pct": 0.17,
        "fade_ms": 70.0,
        "min_sustain_duration": 1.00,
        "attack_threshold": 0.90,
        "decay_threshold": 0.40,
        "pitch_refine_mode": "expand",
        "pitch_refine_min_fraction": 0.75,
    },
    "Custom": {
        "attack_pct": 0.15,
        "sustain_pct": 0.60,
        "decay_pct": 0.25,
        "fade_ms": 50.0,
        "min_sustain_duration": 0.35,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
}

# Orchestral articulation profiles (same ADSR objective, tuned thresholds)
ARTICULATION_PRESETS = {
    "Staccato / Pluck": {
        "attack_pct": 0.22,
        "sustain_pct": 0.45,
        "decay_pct": 0.33,
        "fade_ms": 25.0,
        "min_sustain_duration": 0.04,
        "attack_threshold": 0.82,
        "decay_threshold": 0.55,
        "use_advanced": True,
        "use_smart": False,
    },
    "Legato / Bow": {
        "attack_pct": 0.10,
        "sustain_pct": 0.72,
        "decay_pct": 0.18,
        "fade_ms": 55.0,
        "min_sustain_duration": 0.45,
        "attack_threshold": 0.88,
        "decay_threshold": 0.42,
        "use_advanced": False,
        "use_smart": True,
        "pitch_stability_cents": 8.0,
    },
    "Marcato / Accent": {
        "attack_pct": 0.18,
        "sustain_pct": 0.52,
        "decay_pct": 0.30,
        "fade_ms": 35.0,
        "min_sustain_duration": 0.12,
        "attack_threshold": 0.80,
        "decay_threshold": 0.48,
        "use_advanced": True,
        "use_smart": False,
    },
}

_REGIME_ANNOTATE_KEYS = {
    "use_regime_refine": True,
    "regime_refine_mode": "annotate",
    "regime_flux_ratio": 2.0,
    "regime_flux_ratio_normalised": DEFAULT_REGIME_FLUX_RATIO_NORMALISED,
    "regime_flux_median_frames": 9,
    "regime_reference_fraction": 0.5,
    "regime_min_windows": DEFAULT_REGIME_MIN_WINDOWS,
    "regime_min_duration": 1.0,
    "regime_analysis_n_fft": None,
    "regime_half_integer": True,
    "regime_use_half_integer": True,
    "regime_hi_rel_bandwidth": DEFAULT_REGIME_HI_REL_BANDWIDTH,
    "regime_hi_rise_db": DEFAULT_REGIME_HI_RISE_DB,
    "regime_vibrato_robust": True,
}

for _preset in PRESETS.values():
    for _key, _val in _REGIME_ANNOTATE_KEYS.items():
        _preset.setdefault(_key, _val)
for _preset in ARTICULATION_PRESETS.values():
    for _key, _val in _REGIME_ANNOTATE_KEYS.items():
        _preset.setdefault(_key, _val)

# Sustained-tone profile (Very Long) with regime trim for soft high brass.
PRESETS["soft_high_brass"] = {
    **PRESETS["Very Long (> 6.0s)"],
    "regime_refine_mode": "trim",
    "regime_flux_ratio": 2.0,
    "pitch_stability_cents": 8.0,
}

ALL_PRESETS = {**PRESETS, **ARTICULATION_PRESETS}


@dataclass
class SegmentConfig:
    trim_db: float = DEFAULT_TRIM_DB
    attack_threshold: float = 0.9
    decay_threshold: float = 0.5
    attack_pct: float = 0.15
    sustain_pct: float = 0.60
    decay_pct: float = 0.25
    min_sustain_duration: float = 0.35
    pitch_window_duration: float = 0.5
    pitch_stability_cents: float = 5.0
    use_advanced: bool = False
    use_smart: bool = True
    sustain_variance_threshold: float = DEFAULT_SUSTAIN_VARIANCE_THRESHOLD
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH
    min_sustain_frames: int = DEFAULT_MIN_SUSTAIN_FRAMES
    vibrato_robust: bool = True
    vibrato_median_window_s: float = DEFAULT_VIBRATO_MEDIAN_WINDOW_S
    remove_dc: bool = True
    use_pitch_refine: bool = True
    # annotate = keep energy sustain, record stable window in metadata only (best for STFT)
    # expand   = grow stable seed outward (default; keeps long sustains for spectral work)
    # crop     = tightest stable window only (legacy sampler-style)
    pitch_refine_mode: str = "expand"
    pitch_refine_min_fraction: float = DEFAULT_PITCH_REFINE_MIN_FRACTION
    sustain_fraction_before_decay: float = DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY
    use_regime_refine: bool = True
    regime_refine_mode: str = "annotate"        # annotate | trim
    regime_flux_ratio: float = 2.0              # unnormalised flux walk (advanced attack path)
    regime_flux_ratio_normalised: float = DEFAULT_REGIME_FLUX_RATIO_NORMALISED
    regime_flux_median_frames: int = 9          # odd; moving-median length on the flux track
    regime_reference_fraction: float = 0.5      # central fraction of the pitch-stable sustain used as reference
    regime_min_windows: int = DEFAULT_REGIME_MIN_WINDOWS  # floor: min non-overlapping analysis windows after trim
    regime_min_duration: float = 1.0            # floor: minimum seconds after trim
    regime_analysis_n_fft: Optional[int] = None # downstream STFT size; if None use pitch frame
    regime_half_integer: bool = True            # also compute the half-integer band ratio (needs f0)
    regime_hi_rel_bandwidth: float = DEFAULT_REGIME_HI_REL_BANDWIDTH
    regime_use_half_integer: bool = True
    regime_hi_rise_db: float = DEFAULT_REGIME_HI_RISE_DB
    regime_vibrato_robust: bool = True
    pitch_fmin: float = DEFAULT_PITCH_FMIN_HZ
    pitch_fmax: float = DEFAULT_PITCH_FMAX_HZ
    pitch_fail_cents: float = DEFAULT_PITCH_FAIL_CENTS
    pitch_min_voiced_fraction: float = DEFAULT_PITCH_MIN_VOICED_FRACTION

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "SegmentConfig":
        """Build config from a named preset with optional field overrides."""
        preset = ALL_PRESETS.get(name)
        if preset is None:
            raise ValueError(f"Unknown preset: {name!r}")
        fields = {
            "attack_pct": preset.get("attack_pct", 0.15),
            "sustain_pct": preset.get("sustain_pct", 0.60),
            "decay_pct": preset.get("decay_pct", 0.25),
            "min_sustain_duration": preset.get("min_sustain_duration", 0.35),
            "attack_threshold": preset.get("attack_threshold", 0.9),
            "decay_threshold": preset.get("decay_threshold", 0.5),
            "use_advanced": preset.get("use_advanced", False),
            "use_smart": preset.get("use_smart", True),
            "pitch_stability_cents": preset.get("pitch_stability_cents", 5.0),
            "pitch_refine_mode": preset.get("pitch_refine_mode", "expand"),
            "pitch_refine_min_fraction": preset.get(
                "pitch_refine_min_fraction", DEFAULT_PITCH_REFINE_MIN_FRACTION
            ),
            "use_regime_refine": preset.get("use_regime_refine", True),
            "regime_refine_mode": preset.get("regime_refine_mode", "annotate"),
            "regime_flux_ratio": preset.get("regime_flux_ratio", 2.0),
            "regime_flux_ratio_normalised": preset.get(
                "regime_flux_ratio_normalised", DEFAULT_REGIME_FLUX_RATIO_NORMALISED
            ),
            "regime_flux_median_frames": preset.get("regime_flux_median_frames", 9),
            "regime_reference_fraction": preset.get("regime_reference_fraction", 0.5),
            "regime_min_windows": preset.get("regime_min_windows", DEFAULT_REGIME_MIN_WINDOWS),
            "regime_min_duration": preset.get("regime_min_duration", 1.0),
            "regime_analysis_n_fft": preset.get("regime_analysis_n_fft"),
            "regime_half_integer": preset.get("regime_half_integer", True),
            "regime_use_half_integer": preset.get("regime_use_half_integer", True),
            "regime_hi_rel_bandwidth": preset.get(
                "regime_hi_rel_bandwidth", DEFAULT_REGIME_HI_REL_BANDWIDTH
            ),
            "regime_hi_rise_db": preset.get("regime_hi_rise_db", DEFAULT_REGIME_HI_RISE_DB),
            "regime_vibrato_robust": preset.get("regime_vibrato_robust", True),
            "pitch_fmin": preset.get("pitch_fmin", DEFAULT_PITCH_FMIN_HZ),
            "pitch_fmax": preset.get("pitch_fmax", DEFAULT_PITCH_FMAX_HZ),
            "pitch_fail_cents": preset.get("pitch_fail_cents", DEFAULT_PITCH_FAIL_CENTS),
            "pitch_min_voiced_fraction": preset.get(
                "pitch_min_voiced_fraction", DEFAULT_PITCH_MIN_VOICED_FRACTION
            ),
        }
        fields.update(overrides)
        return cls(**fields)


@dataclass
class TrimInfo:
    idx_start: int
    idx_end: int
    t_start: float
    t_end: float
    active_len: float


@dataclass
class SegmentResult:
    t_att: float
    t_dec: float
    t_end: float
    trim: TrimInfo
    pitch_refine: Dict = field(default_factory=dict)
    regime_refine: Dict = field(default_factory=dict)


def preprocess_signal(y: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """Optional DC removal before envelope analysis."""
    if not remove_dc or len(y) == 0:
        return y
    return y - float(np.mean(y))


def trim_active_region(y: np.ndarray, sr: int, trim_db: float = DEFAULT_TRIM_DB) -> Tuple[np.ndarray, TrimInfo]:
    y_trimmed, index = librosa.effects.trim(y, top_db=trim_db)
    idx_start, idx_end = int(index[0]), int(index[1])
    t_start = idx_start / sr
    t_end_trimmed = idx_end / sr
    t_end_signal = len(y) / sr
    t_end = min(t_end_trimmed, t_end_signal - 1e-3)
    active_len = max(0.0, t_end - t_start)
    return y_trimmed, TrimInfo(idx_start, idx_end, t_start, t_end, active_len)


def compute_rms_envelope(
    y: np.ndarray, sr: int, frame_length: int = DEFAULT_FRAME_LENGTH, hop_length: int = DEFAULT_HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    return rms, times


def compute_spectral_flux(
    y: np.ndarray,
    sr: int,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: Optional[int] = None,
    normalised: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Half-wave-rectified spectral flux.

    When ``normalised`` is True, each frame's magnitude is divided by its sum
    so flux is level-independent. The attack detector in advanced mode keeps
    the default ``normalised=False`` so v3.1 attack MAEs do not move.
    """
    n_fft = int(frame_length if n_fft is None else n_fft)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    if normalised:
        magnitude = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-12)
    diff = np.diff(magnitude, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    times = librosa.times_like(flux, sr=sr, hop_length=hop_length)
    return flux, times


def _moving_median(arr: np.ndarray, win: int) -> np.ndarray:
    """Odd-length moving median (used to suppress vibrato in pitch stability scoring)."""
    win = max(3, win | 1)
    half = win // 2
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = float(np.median(arr[lo:hi]))
    return out


def pitch_stability_std_cents(
    cents: np.ndarray,
    times: np.ndarray,
    cfg: SegmentConfig,
) -> Optional[float]:
    """
    Pitch stability in cents after linear detrend and optional vibrato suppression.
    Vibrato (~4–7 Hz on strings) is attenuated via a short moving-median residual.
    """
    valid = np.isfinite(cents)
    if valid.sum() < 3:
        return None
    c = cents[valid].astype(np.float64)
    t = times[valid].astype(np.float64)
    if len(c) >= 3:
        slope, intercept = np.polyfit(t - t[0], c, 1)
        c = c - np.polyval([slope, intercept], t - t[0])
    if cfg.vibrato_robust and len(c) >= 5:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else cfg.vibrato_median_window_s
        if dt > 0:
            win = max(3, int(round(cfg.vibrato_median_window_s / dt)) | 1)
            if len(c) >= win:
                c = c - _moving_median(c, win)
    return float(np.std(c))


def detect_attack_energy(
    rms: np.ndarray, times: np.ndarray, peak_idx: int, threshold: float
) -> float:
    """Attack end: first frame at/above threshold * peak while ascending toward peak."""
    if len(rms) < 2:
        return float(times[0])
    peak_val = float(rms[peak_idx])
    if peak_val < 1e-12:
        return float(times[0])
    level = threshold * peak_val
    search_end = max(1, peak_idx + 1)
    for i in range(search_end):
        if rms[i] >= level:
            return float(times[i])
    return float(times[min(peak_idx, len(times) - 1)])


def detect_attack_derivative(
    rms: np.ndarray,
    times: np.ndarray,
    peak_idx: int,
    spectral_flux: Optional[np.ndarray] = None,
    flux_times: Optional[np.ndarray] = None,
) -> float:
    if len(rms) < 2:
        return float(times[0])
    drms = np.diff(rms)
    search_end = max(1, min(peak_idx, len(drms)))
    max_derivative_idx = int(np.argmax(drms[:search_end]))
    attack_time = float(times[max_derivative_idx])
    if spectral_flux is not None and flux_times is not None and len(spectral_flux) > 0:
        flux_peak_time = float(flux_times[int(np.argmax(spectral_flux))])
        attack_time = min(attack_time, flux_peak_time)
    min_attack_time = float(times[int(len(times) * 0.05)])
    attack_time = max(min_attack_time, attack_time)
    peak_time = float(times[peak_idx])
    attack_time = min(attack_time, peak_time * 0.85)
    return attack_time


def detect_attack_combined(
    rms: np.ndarray,
    times: np.ndarray,
    peak_idx: int,
    threshold: float,
    spectral_flux: Optional[np.ndarray] = None,
    flux_times: Optional[np.ndarray] = None,
    use_derivative: bool = False,
) -> float:
    energy_t = detect_attack_energy(rms, times, peak_idx, threshold)
    if not use_derivative:
        return energy_t
    deriv_t = detect_attack_derivative(rms, times, peak_idx, spectral_flux, flux_times)
    return min(energy_t, deriv_t)


def _normalized_adsr_fractions(
    attack_pct: float, sustain_pct: float, decay_pct: float
) -> Tuple[float, float, float]:
    total = attack_pct + sustain_pct + decay_pct
    if total <= 0:
        return 0.15, 0.60, 0.25
    return attack_pct / total, sustain_pct / total, decay_pct / total


def min_decay_time_proportional(
    active_len: float, attack_pct: float, sustain_pct: float, decay_pct: float, sustain_fraction: float
) -> float:
    """Earliest allowed decay start: after sustain_fraction of the proportional sustain zone."""
    att_f, sus_f, _ = _normalized_adsr_fractions(attack_pct, sustain_pct, decay_pct)
    return active_len * (att_f + sus_f * sustain_fraction)


def detect_decay_energy(
    rms: np.ndarray,
    times: np.ndarray,
    attack_idx: int,
    peak_idx: int,
    threshold: float,
    min_decay_time: Optional[float] = None,
) -> float:
    peak_val = float(np.max(rms))
    if peak_val < 1e-12:
        return float(times[-1])
    level = threshold * peak_val
    search_start = max(attack_idx, peak_idx)
    if min_decay_time is not None:
        search_start = max(
            search_start,
            int(np.searchsorted(times, min_decay_time, side="left")),
        )
    for i in range(search_start, len(rms)):
        if rms[i] <= level:
            return float(times[i])
    return float(times[int(len(times) * 0.85)])


def detect_decay_derivative(
    rms: np.ndarray, times: np.ndarray, attack_idx: int, peak_idx: int, threshold: float
) -> float:
    if len(rms) < 2:
        return float(times[-1])
    drms = np.diff(rms)
    search_start = max(attack_idx, peak_idx)
    if peak_idx < len(times):
        peak_time = float(times[peak_idx])
        min_decay_delay = max(0.05, (times[-1] - peak_time) * 0.15)
        min_decay_time = peak_time + min_decay_delay
        search_start = max(search_start, int(np.searchsorted(times, min_decay_time, side="right")))
    search_start = min(search_start, len(drms) - 1)
    negative_count = 0
    for i in range(search_start, len(drms)):
        if drms[i] < 0:
            negative_count += 1
            if negative_count >= 3:
                return float(times[i - 2])
        else:
            negative_count = 0
    return detect_decay_energy(rms, times, attack_idx, peak_idx, threshold)


def detect_segments_proportional(
    active_len: float,
    attack_pct: float,
    sustain_pct: float,
    decay_pct: float,
    min_sustain_duration: float,
) -> Tuple[float, float]:
    total = attack_pct + sustain_pct + decay_pct
    if total > 0:
        attack_pct /= total
        sustain_pct /= total
        decay_pct /= total
    t_attack_end = active_len * attack_pct
    t_decay_start = active_len * (attack_pct + sustain_pct)
    min_sustain_actual = max(min_sustain_duration, active_len * sustain_pct * 0.4)
    t_decay_start = max(t_attack_end + min_sustain_actual, t_decay_start)
    margin = active_len * 0.05
    t_decay_start = min(t_decay_start, active_len - margin)
    if t_decay_start <= t_attack_end:
        t_decay_start = min(t_attack_end + active_len * sustain_pct, active_len - 0.02)
    return t_attack_end, t_decay_start


def detect_sustain_plateau(
    rms: np.ndarray,
    times: np.ndarray,
    attack_idx: int,
    decay_idx: int,
    min_duration: float,
    variance_threshold: float = DEFAULT_SUSTAIN_VARIANCE_THRESHOLD,
) -> Tuple[Optional[int], Optional[int]]:
    if decay_idx <= attack_idx + 1:
        return None, None
    sustain_rms = rms[attack_idx:decay_idx]
    if len(sustain_rms) < 2:
        return None, None
    mean_rms = np.mean(sustain_rms)
    if mean_rms < 1e-10:
        return None, None
    variance = np.var(sustain_rms) / (mean_rms ** 2)
    duration = times[decay_idx] - times[attack_idx]
    if variance < variance_threshold and duration >= min_duration:
        return attack_idx, decay_idx
    return None, None


def effective_min_sustain_duration(
    cfg: SegmentConfig, sr: int, active_len: Optional[float] = None
) -> float:
    min_by_frames = (cfg.min_sustain_frames * cfg.hop_length) / max(float(sr), 1.0)
    min_required = max(cfg.min_sustain_duration, cfg.pitch_window_duration, min_by_frames)
    if active_len is not None and 0 < active_len < min_required:
        min_required = max(active_len * 0.25, min_by_frames, 0.02)
    return min_required


def parse_note_from_filename(path: Optional[Path]) -> Tuple[Optional[float], Dict]:
    """Parse a pitch-class + octave from a filename.

    Octave-wrap spellings (B#, Cb, E#, Fb) are resolved to the sounding pitch
    (B#4 = C5, Cb5 = B4) and flagged in the returned metadata.
    """
    info: Dict = {
        "note_name": None,
        "note_name_wrap_spelling": False,
        "expected_note_hz": None,
    }
    if path is None:
        return None, info
    match = re.search(r"([A-Ga-g])(#|b)?(\d+)", path.stem)
    if not match:
        return None, info
    letter, accidental, octave = match.group(1).upper(), match.group(2), match.group(3)
    note_str = letter + (accidental or "") + octave
    info["note_name"] = note_str
    if accidental and (letter, accidental) in NOTE_WRAP_SPELLINGS:
        info["note_name_wrap_spelling"] = True
        logger.warning("Octave-wrap note spelling %s in %s", note_str, path.name)
    try:
        hz = float(librosa.note_to_hz(note_str))
    except Exception:
        return None, info
    info["expected_note_hz"] = hz
    return hz, info


def parse_note_hz_from_filename(path: Optional[Path]) -> Optional[float]:
    hz, _ = parse_note_from_filename(path)
    return hz


def next_pow2(n: float) -> int:
    p = 1
    need = max(1, int(np.ceil(float(n))))
    while p < need:
        p *= 2
    return p


def pitch_frame_length_for_fmin(sr: int, fmin_hz: float, cap: int = 8192) -> int:
    """Power-of-two frame length ≥ 4·sr/fmin, capped at ``cap``."""
    need = 4.0 * float(sr) / max(float(fmin_hz), 1.0)
    return int(min(max(next_pow2(need), 64), cap))


def pitch_search_range(
    cfg: SegmentConfig,
    expected_note_hz: Optional[float] = None,
    file_path: Optional[Path] = None,
    search_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Optional[float]]:
    """Return (fmin, fmax, filename_hz). Explicit search_range wins."""
    file_hz, _ = parse_note_from_filename(file_path)
    if search_range is not None:
        return float(search_range[0]), float(search_range[1]), file_hz
    center = file_hz if file_hz and file_hz > 0 else None
    if center is None and expected_note_hz is not None and expected_note_hz > 0:
        center = float(expected_note_hz)
    if center is not None:
        # Slightly inside a full octave so yin/pyin cannot sit on the exact
        # subharmonic (center/2), which is otherwise a legal ±1-octave bound.
        fmin = max(float(cfg.pitch_fmin), center / 2.0 * 1.06)
        fmax = min(float(cfg.pitch_fmax), center * 2.0 / 1.06)
        if fmax <= fmin:
            fmin, fmax = float(cfg.pitch_fmin), float(cfg.pitch_fmax)
        return fmin, fmax, file_hz
    return float(cfg.pitch_fmin), float(cfg.pitch_fmax), file_hz


def _expand_stable_pitch_window(
    cents: np.ndarray,
    times: np.ndarray,
    seed_lo: int,
    seed_hi: int,
    cfg: SegmentConfig,
) -> Tuple[int, int]:
    """Grow a pitch-stable seed window outward while stability stays within tolerance."""
    lo, hi = seed_lo, seed_hi
    threshold = cfg.pitch_stability_cents * 1.25

    while lo > 0:
        std = pitch_stability_std_cents(cents[lo - 1 : hi], times[lo - 1 : hi], cfg)
        if std is None or std > threshold:
            break
        lo -= 1

    while hi < len(cents):
        std = pitch_stability_std_cents(cents[lo : hi + 1], times[lo : hi + 1], cfg)
        if std is None or std > threshold:
            break
        hi += 1

    return lo, hi


def refine_sustain_by_pitch(
    y_trimmed: np.ndarray,
    sr: int,
    t_att_rel: float,
    t_dec_rel: float,
    cfg: SegmentConfig,
    expected_note_hz: Optional[float] = None,
    file_path: Optional[Path] = None,
    search_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, Dict]:
    info: Dict = {
        "used": False,
        "std_cents": None,
        "window_start": None,
        "window_end": None,
        "window_duration": None,
        "expected_note_hz": expected_note_hz,
        "mean_abs_cents_from_note": None,
        "mode": cfg.pitch_refine_mode,
        "energy_sustain_duration": None,
        "kept_energy_boundaries": False,
        "median_f0_hz": None,
        "failed": False,
        "failed_reason": None,
        "note_name_wrap_spelling": False,
        "note_name_mismatch": False,
        "pitch_frame_length": None,
        "pitch_fmin": None,
        "pitch_fmax": None,
    }
    file_hz, file_meta = parse_note_from_filename(file_path)
    info["note_name_wrap_spelling"] = bool(file_meta.get("note_name_wrap_spelling"))
    if expected_note_hz is None and file_hz is not None:
        expected_note_hz = file_hz
        info["expected_note_hz"] = expected_note_hz
    if not cfg.use_pitch_refine:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    energy_att, energy_dec = t_att_rel, t_dec_rel
    energy_sustain_dur = max(0.0, energy_dec - energy_att)
    info["energy_sustain_duration"] = energy_sustain_dur

    min_duration = effective_min_sustain_duration(cfg, sr, len(y_trimmed) / sr)
    # Analysis grain for seed search — not the export length in expand mode
    window_duration = max(cfg.pitch_window_duration, min(0.5, min_duration))
    total_len = len(y_trimmed) / sr
    sustain_start = max(0.0, min(t_att_rel, total_len))
    sustain_end = max(sustain_start, min(t_dec_rel, total_len))
    if sustain_end - sustain_start < min_duration:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    start_idx = int(sustain_start * sr)
    end_idx = int(sustain_end * sr)
    y_sustain = y_trimmed[start_idx:end_idx]
    fmin_hz, fmax_hz, file_hz = pitch_search_range(cfg, expected_note_hz, file_path, search_range)
    has_note_context = search_range is None and (
        (file_hz is not None and file_hz > 0)
        or (expected_note_hz is not None and expected_note_hz > 0)
    )
    # Scale the YIN frame with register when a note or an explicit search
    # range is given. Unnamed files keep the historical 1024-sample grain
    # so v3.2 expand-mode benchmark rows do not move.
    pitch_n = (
        pitch_frame_length_for_fmin(sr, fmin_hz)
        if (has_note_context or search_range is not None)
        else int(cfg.frame_length)
    )
    info["pitch_fmin"] = float(fmin_hz)
    info["pitch_fmax"] = float(fmax_hz)
    info["pitch_frame_length"] = int(pitch_n)
    if len(y_sustain) < min(cfg.frame_length, pitch_n):
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    try:
        f0 = librosa.yin(
            y_sustain,
            fmin=float(fmin_hz),
            fmax=float(fmax_hz),
            sr=sr,
            frame_length=int(pitch_n),
            hop_length=cfg.hop_length,
        )
    except Exception:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    times = librosa.times_like(f0, sr=sr, hop_length=cfg.hop_length)
    valid = np.isfinite(f0) & (f0 > 0)
    if valid.sum() < 3:
        info["failed"] = True
        info["failed_reason"] = "unvoiced"
        info["kept_energy_boundaries"] = True
        info["median_f0_hz"] = None
        return t_att_rel, t_dec_rel, info

    f0_med = float(np.median(f0[valid]))
    if f0_med <= 0:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    consistent = valid & (np.abs(1200.0 * np.log2(np.maximum(f0, 1e-12) / f0_med)) < 200.0)
    voiced_frac = float(consistent.sum()) / max(len(f0), 1)
    if voiced_frac < float(cfg.pitch_min_voiced_fraction):
        info["failed"] = True
        info["failed_reason"] = "unvoiced"
        info["kept_energy_boundaries"] = True
        info["median_f0_hz"] = None
        return t_att_rel, t_dec_rel, info
    info["median_f0_hz"] = f0_med

    if expected_note_hz is not None and expected_note_hz > 0:
        cents_off = abs(1200.0 * np.log2(f0_med / float(expected_note_hz)))
        if cents_off > DEFAULT_PITCH_OCTAVE_FAIL_CENTS:
            info["failed"] = True
            info["failed_reason"] = "octave_error"
            info["note_name_mismatch"] = True
            info["kept_energy_boundaries"] = True
            info["std_cents"] = None
            info["median_f0_hz"] = f0_med
            return t_att_rel, t_dec_rel, info
        if info["note_name_wrap_spelling"] and cents_off > 50.0:
            info["note_name_mismatch"] = True

    cents = np.full_like(f0, np.nan, dtype=np.float64)
    cents[valid] = 1200.0 * np.log2(f0[valid] / f0_med)
    cents_from_note = np.full_like(f0, np.nan, dtype=np.float64)
    if expected_note_hz is not None and expected_note_hz > 0:
        cents_from_note[valid] = 1200.0 * np.log2(f0[valid] / expected_note_hz)

    window_frames = max(1, int(np.ceil(window_duration * sr / cfg.hop_length)))
    best_std = None
    best_start = None
    best_mean_abs_from_note = None
    best_score = None

    for i in range(0, len(cents) - window_frames + 1):
        window = cents[i : i + window_frames]
        w_times = times[i : i + window_frames]
        w_valid = np.isfinite(window)
        if w_valid.sum() < max(3, int(0.6 * window_frames)):
            continue
        std = pitch_stability_std_cents(window, w_times, cfg)
        if std is None:
            continue
        mean_abs_from_note = 0.0
        if expected_note_hz is not None:
            w_note = np.isfinite(cents_from_note[i : i + window_frames])
            if w_note.sum() >= max(3, int(0.6 * window_frames)):
                mean_abs_from_note = float(
                    np.mean(np.abs(cents_from_note[i : i + window_frames][w_note]))
                )
        score = std + mean_abs_from_note
        if best_score is None or score < best_score:
            best_score = score
            best_std = std
            best_start = i
            best_mean_abs_from_note = mean_abs_from_note if expected_note_hz else None

    if best_std is None or best_start is None:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    fail_std = best_std
    if fail_std is not None and fail_std > float(cfg.pitch_fail_cents):
        info["std_cents"] = float(fail_std)
        info["failed"] = True
        info["failed_reason"] = "tracking_failed"
        info["kept_energy_boundaries"] = True
        info["median_f0_hz"] = f0_med
        return t_att_rel, t_dec_rel, info

    seed_lo = best_start
    seed_hi = best_start + window_frames

    if best_std <= cfg.pitch_stability_cents:
        if cfg.pitch_refine_mode == "expand":
            seed_lo, seed_hi = _expand_stable_pitch_window(cents, times, seed_lo, seed_hi, cfg)
        elif cfg.pitch_refine_mode == "crop":
            pass  # keep seed window only

        win_start_t = sustain_start + float(times[seed_lo])
        win_end_t = sustain_start + float(times[min(seed_hi - 1, len(times) - 1)])
        win_end_t = min(max(win_end_t, win_start_t + min_duration), sustain_end)

        refined_dur = win_end_t - win_start_t
        min_allowed = energy_sustain_dur * cfg.pitch_refine_min_fraction

        if refined_dur < min_allowed:
            info.update(
                {
                    "std_cents": best_std,
                    "window_start": win_start_t,
                    "window_end": win_end_t,
                    "window_duration": refined_dur,
                    "mean_abs_cents_from_note": best_mean_abs_from_note,
                    "kept_energy_boundaries": True,
                }
            )
            return energy_att, energy_dec, info

        if cfg.pitch_refine_mode == "annotate":
            info.update(
                {
                    "used": True,
                    "std_cents": pitch_stability_std_cents(
                        cents[seed_lo:seed_hi], times[seed_lo:seed_hi], cfg
                    ),
                    "window_start": win_start_t,
                    "window_end": win_end_t,
                    "window_duration": refined_dur,
                    "mean_abs_cents_from_note": best_mean_abs_from_note,
                    "kept_energy_boundaries": True,
                }
            )
            return energy_att, energy_dec, info

        info.update(
            {
                "used": True,
                "std_cents": pitch_stability_std_cents(
                    cents[seed_lo:seed_hi], times[seed_lo:seed_hi], cfg
                ),
                "window_start": win_start_t,
                "window_end": win_end_t,
                "window_duration": win_end_t - win_start_t,
                "mean_abs_cents_from_note": best_mean_abs_from_note,
                "kept_energy_boundaries": False,
            }
        )
        return win_start_t, win_end_t, info

    info["std_cents"] = best_std
    info["mean_abs_cents_from_note"] = best_mean_abs_from_note
    info["kept_energy_boundaries"] = True
    return t_att_rel, t_dec_rel, info


def resolve_analysis_n_fft(
    cfg: SegmentConfig, pitch_frame_length: Optional[int] = None
) -> Tuple[int, str]:
    if cfg.regime_analysis_n_fft is not None and int(cfg.regime_analysis_n_fft) > 0:
        return int(cfg.regime_analysis_n_fft), "config"
    if pitch_frame_length is not None and int(pitch_frame_length) > 0:
        return int(pitch_frame_length), "pitch_frame"
    return int(cfg.frame_length), "frame_length"


def resolve_hi_n_fft(cfg: SegmentConfig, pitch_frame_length: Optional[int] = None) -> int:
    """STFT size for the half-integer track: pitch frame, else max(frame_length, 4096)."""
    if pitch_frame_length is not None and int(pitch_frame_length) > 0:
        return int(pitch_frame_length)
    return max(int(cfg.frame_length), DEFAULT_HI_N_FFT_FLOOR)


def compute_half_integer_ratio_db(
    y: np.ndarray,
    sr: int,
    f0_hz: Optional[float],
    cfg: SegmentConfig,
    n_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Per-frame 1.5·f0 + 2.5·f0 energy vs f0, in dB, with bands ±α·f0.

    Returns (ratio_db, times, status). NaN where unresolvable or f0 is missing.
    """
    n_fft = int(cfg.frame_length if n_fft is None else n_fft)
    hop = int(cfg.hop_length)
    alpha = float(cfg.regime_hi_rel_bandwidth)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(magnitude, sr=sr, hop_length=hop)
    n_frames = int(magnitude.shape[1])
    status: Dict = {
        "half_integer_valid": False,
        "reason": None,
        "half_integer_bandwidth_hz": None,
        "n_fft": n_fft,
    }
    nan = np.full(n_frames, np.nan, dtype=np.float64)
    if f0_hz is None or not np.isfinite(f0_hz) or float(f0_hz) <= 0:
        status["reason"] = "no_f0"
        return nan, times, status

    f0 = float(f0_hz)
    bandwidth = alpha * f0
    status["half_integer_bandwidth_hz"] = bandwidth
    bin_hz = float(sr) / float(n_fft)
    if bandwidth < 2.0 * bin_hz:
        status["reason"] = "band_below_resolution"
        return nan, times, status

    def _band_energy(center_hz: float, half_width_hz: float) -> np.ndarray:
        mask = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
        if not np.any(mask):
            return np.zeros(n_frames, dtype=np.float64)
        return np.sum(magnitude[mask, :] ** 2, axis=0)

    e_f0 = _band_energy(f0, bandwidth)
    e_half = _band_energy(1.5 * f0, bandwidth) + _band_energy(2.5 * f0, bandwidth)
    ratio_db = np.full(n_frames, np.nan, dtype=np.float64)
    valid = e_f0 > 1e-20
    ratio_db[valid] = 10.0 * np.log10(np.maximum(e_half[valid], 1e-20) / e_f0[valid])
    status["half_integer_valid"] = True
    status["reason"] = None
    return ratio_db, times, status


def effective_regime_floor(
    cfg: SegmentConfig, sr: int, analysis_n_fft: Optional[int] = None
) -> float:
    """Seconds: max(regime_min_duration, regime_min_windows * analysis_n_fft / sr)."""
    if analysis_n_fft is None:
        analysis_n_fft, _ = resolve_analysis_n_fft(cfg)
    by_windows = float(cfg.regime_min_windows) * float(analysis_n_fft) / max(float(sr), 1.0)
    return max(float(cfg.regime_min_duration), float(by_windows))


def _regime_info_template(cfg: SegmentConfig) -> Dict:
    n_fft, src = resolve_analysis_n_fft(cfg)
    return {
        "used": False,
        "mode": cfg.regime_refine_mode,
        "flux_reference": None,
        "flux_edge_ratio_start": None,
        "flux_edge_ratio_end": None,
        "window_start": None,
        "window_end": None,
        "window_duration": None,
        "trimmed_start_s": None,
        "trimmed_end_s": None,
        "refused": None,
        "refused_reason": None,
        "floor_seconds": None,
        "floor_windows": int(cfg.regime_min_windows),
        "analysis_n_fft": int(n_fft),
        "analysis_n_fft_source": src,
        "half_integer_ratio_db_edges": None,
        "half_integer_ratio_db_middle": None,
        "half_integer_valid": None,
        "half_integer_invalid_reason": None,
        "half_integer_bandwidth_hz": None,
        "hi_n_fft": None,
        "hi_reference_db": None,
        "flux_normalised": None,
        "flux_ratio_applied": None,
        "hi_edge_rise_db_start": None,
        "hi_edge_rise_db_end": None,
        "hi_trimmed_start_s": None,
        "hi_trimmed_end_s": None,
        "boundary_source_start": None,
        "boundary_source_end": None,
    }


def _vibrato_smooth_track(values: np.ndarray, times: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Low-pass a flux/HI track with the pitch-stage vibrato median window."""
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if finite.sum() < 5 or not cfg.regime_vibrato_robust:
        return out
    t = np.asarray(times, dtype=np.float64)
    dt = float(np.median(np.diff(t[finite]))) if finite.sum() > 1 else cfg.vibrato_median_window_s
    if dt <= 0:
        return out
    win = max(3, int(round(cfg.vibrato_median_window_s / dt)) | 1)
    if finite.sum() < win:
        return out
    filled = out.copy()
    if not np.all(finite):
        idx = np.arange(len(filled))
        filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    return _moving_median(filled, win)


def _walk_inward(values: np.ndarray, times: np.ndarray, above_fn) -> Tuple[int, int]:
    n = len(values)
    start_i = 0
    while start_i < n and above_fn(values[start_i], start_i):
        start_i += 1
    end_i = n - 1
    while end_i > start_i and above_fn(values[end_i], end_i):
        end_i -= 1
    return start_i, end_i


def _median_finite(arr: np.ndarray) -> Optional[float]:
    if arr is None or len(arr) == 0:
        return None
    valid = np.asarray(arr, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def refine_sustain_by_regime(
    y_trimmed: np.ndarray,
    sr: int,
    t_att_rel: float,
    t_dec_rel: float,
    cfg: SegmentConfig,
    f0_hz: Optional[float] = None,
    pitch_frame_length: Optional[int] = None,
    flux_normalised: bool = True,
) -> Tuple[float, float, Dict]:
    """
    Third refinement stage. Uses spectral flux stationarity on the pitch-stable sustain,
    optionally combined with a relative half-integer walk.
    In mode 'annotate' the returned times equal the inputs and the window is reported
    in info only; in mode 'trim' the returned times are the candidate boundaries.
    The floor gates applying a trim only: diagnostics are always filled when frames exist.
    """
    info = _regime_info_template(cfg)
    analysis_n_fft, nfft_src = resolve_analysis_n_fft(cfg, pitch_frame_length)
    hi_n_fft = resolve_hi_n_fft(cfg, pitch_frame_length)
    info["analysis_n_fft"] = int(analysis_n_fft)
    info["analysis_n_fft_source"] = nfft_src
    info["hi_n_fft"] = int(hi_n_fft)
    info["flux_normalised"] = bool(flux_normalised)
    info["flux_ratio_applied"] = float(
        cfg.regime_flux_ratio_normalised if flux_normalised else cfg.regime_flux_ratio
    )
    floor_s = effective_regime_floor(cfg, sr, analysis_n_fft)
    info["floor_seconds"] = float(floor_s)
    info["used"] = True
    info["boundary_source_start"] = "none"
    info["boundary_source_end"] = "none"

    t_att_rel = float(t_att_rel)
    t_dec_rel = float(t_dec_rel)
    span = max(0.0, t_dec_rel - t_att_rel)

    def _set_hi_invalid(reason: str) -> None:
        info["half_integer_valid"] = False
        info["half_integer_invalid_reason"] = reason
        info["half_integer_reason"] = reason

    def _keep(reason: Optional[str]) -> Tuple[float, float, Dict]:
        info["refused"] = True
        info["refused_reason"] = reason
        info["window_start"] = t_att_rel
        info["window_end"] = t_dec_rel
        info["window_duration"] = span
        info["trimmed_start_s"] = 0.0
        info["trimmed_end_s"] = 0.0
        info["hi_trimmed_start_s"] = 0.0
        info["hi_trimmed_end_s"] = 0.0
        if info.get("half_integer_invalid_reason") is None:
            _set_hi_invalid("no_f0" if not (f0_hz and f0_hz > 0) else "refused")
        return t_att_rel, t_dec_rel, info

    if span <= 0.0 or len(y_trimmed) < cfg.frame_length:
        return _keep("insufficient_frames")

    flux, times = compute_spectral_flux(
        y_trimmed,
        sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        normalised=bool(flux_normalised),
    )
    mask = (times >= t_att_rel) & (times <= t_dec_rel)
    if int(np.count_nonzero(mask)) < 3:
        return _keep("insufficient_frames")

    flux_s = np.asarray(flux[mask], dtype=np.float64)
    times_s = np.asarray(times[mask], dtype=np.float64)
    n = len(flux_s)
    frac = min(1.0, max(0.05, float(cfg.regime_reference_fraction)))
    lo = int(n * (1.0 - frac) / 2.0)
    hi = max(lo + 1, int(n * (1.0 + frac) / 2.0))

    flux_lp = _vibrato_smooth_track(flux_s, times_s, cfg)
    smoothed = _moving_median(flux_lp, int(cfg.regime_flux_median_frames))
    reference = float(np.median(smoothed[lo:hi]))
    info["flux_reference"] = reference

    denom = max(reference, 1e-12)
    n_edge = max(1, n // 8)
    info["flux_edge_ratio_start"] = float(np.median(smoothed[:n_edge]) / denom)
    info["flux_edge_ratio_end"] = float(np.median(smoothed[-n_edge:]) / denom)

    threshold = float(info["flux_ratio_applied"]) * denom
    flux_start_i, flux_end_i = _walk_inward(smoothed, times_s, lambda v, _i: v > threshold)
    if flux_start_i >= n:
        flux_att, flux_dec = t_att_rel, t_dec_rel
    else:
        flux_att = float(times_s[flux_start_i])
        flux_dec = float(times_s[flux_end_i])
        if flux_dec < flux_att:
            flux_att, flux_dec = t_att_rel, t_dec_rel

    hi_att, hi_dec = t_att_rel, t_dec_rel
    hi_db, hi_times, hi_status = compute_half_integer_ratio_db(
        y_trimmed, sr, f0_hz, cfg, n_fft=hi_n_fft
    )
    info["half_integer_valid"] = bool(hi_status.get("half_integer_valid"))
    info["half_integer_bandwidth_hz"] = hi_status.get("half_integer_bandwidth_hz")
    info["hi_n_fft"] = int(hi_status.get("n_fft") or hi_n_fft)
    if info["half_integer_valid"]:
        info["half_integer_invalid_reason"] = None
        info["half_integer_reason"] = None
    else:
        reason = hi_status.get("reason") or "refused"
        if reason not in ("band_below_resolution", "no_f0", "refused"):
            reason = "refused"
        _set_hi_invalid(reason)
    hi_mask = (hi_times >= t_att_rel) & (hi_times <= t_dec_rel)
    hi_s = np.asarray(hi_db[hi_mask], dtype=np.float64)
    if hi_s.size and cfg.regime_half_integer:
        e = max(1, len(hi_s) // 8)
        mid_lo = int(len(hi_s) * (1.0 - frac) / 2.0)
        mid_hi = max(mid_lo + 1, int(len(hi_s) * (1.0 + frac) / 2.0))
        info["half_integer_ratio_db_edges"] = (
            _median_finite(hi_s[:e]),
            _median_finite(hi_s[-e:]),
        )
        info["half_integer_ratio_db_middle"] = _median_finite(hi_s[mid_lo:mid_hi])

    if (
        info["half_integer_valid"]
        and cfg.regime_use_half_integer
        and cfg.regime_half_integer
        and hi_s.size >= 3
    ):
        hi_times_s = np.asarray(hi_times[hi_mask], dtype=np.float64)
        hi_lp = _vibrato_smooth_track(hi_s, hi_times_s, cfg)
        hi_sm = _moving_median(hi_lp, int(cfg.regime_flux_median_frames))
        n_hi = len(hi_sm)
        hlo = int(n_hi * (1.0 - frac) / 2.0)
        hhi = max(hlo + 1, int(n_hi * (1.0 + frac) / 2.0))
        hi_ref = _median_finite(hi_sm[hlo:hhi])
        info["hi_reference_db"] = hi_ref
        if hi_ref is not None:
            rise = float(cfg.regime_hi_rise_db)
            info["hi_edge_rise_db_start"] = float(np.nanmedian(hi_sm[: max(1, n_hi // 8)]) - hi_ref)
            info["hi_edge_rise_db_end"] = float(np.nanmedian(hi_sm[-max(1, n_hi // 8) :]) - hi_ref)
            hs, he = _walk_inward(hi_sm, hi_times_s, lambda v, _i: (v - hi_ref) > rise)
            if hs < n_hi:
                hi_att = float(hi_times_s[hs])
                hi_dec = float(hi_times_s[he])
            info["hi_trimmed_start_s"] = max(0.0, hi_att - t_att_rel)
            info["hi_trimmed_end_s"] = max(0.0, t_dec_rel - hi_dec)
        else:
            info["hi_trimmed_start_s"] = 0.0
            info["hi_trimmed_end_s"] = 0.0
    else:
        if not info["half_integer_valid"]:
            info["half_integer_valid"] = False
        info["hi_trimmed_start_s"] = 0.0
        info["hi_trimmed_end_s"] = 0.0

    cand_att = max(flux_att, hi_att)
    cand_dec = min(flux_dec, hi_dec)
    if cand_dec < cand_att:
        cand_att, cand_dec = t_att_rel, t_dec_rel

    flux_trim_s = flux_att - t_att_rel
    hi_trim_s = hi_att - t_att_rel
    if max(flux_trim_s, hi_trim_s) <= 1e-6:
        info["boundary_source_start"] = "none"
    elif hi_trim_s > flux_trim_s + 1e-6:
        info["boundary_source_start"] = "half_integer"
    else:
        info["boundary_source_start"] = "flux"

    flux_trim_e = t_dec_rel - flux_dec
    hi_trim_e = t_dec_rel - hi_dec
    if max(flux_trim_e, hi_trim_e) <= 1e-6:
        info["boundary_source_end"] = "none"
    elif hi_trim_e > flux_trim_e + 1e-6:
        info["boundary_source_end"] = "half_integer"
    else:
        info["boundary_source_end"] = "flux"

    cand_span = max(0.0, cand_dec - cand_att)
    info["window_start"] = cand_att
    info["window_end"] = cand_dec
    info["window_duration"] = cand_span
    info["trimmed_start_s"] = max(0.0, cand_att - t_att_rel)
    info["trimmed_end_s"] = max(0.0, t_dec_rel - cand_dec)

    if span < floor_s or cand_span < floor_s:
        info["refused"] = True
        info["refused_reason"] = "span_below_floor"
        return t_att_rel, t_dec_rel, info

    info["refused"] = False
    info["refused_reason"] = None
    if cfg.regime_refine_mode == "trim":
        return cand_att, cand_dec, info
    return t_att_rel, t_dec_rel, info


def build_regime_flux_sidecar(
    y: np.ndarray,
    sr: int,
    cfg: SegmentConfig,
    t_start: float,
    t_end: float,
    f0_hz: Optional[float] = None,
    pitch_frame_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Flux + half-integer tracks on the sustain frame grid (times relative to t_start)."""
    i0 = max(0, int(round(float(t_start) * sr)))
    i1 = min(len(y), max(i0 + 1, int(round(float(t_end) * sr))))
    y_s = y[i0:i1]
    analysis_n_fft, _ = resolve_analysis_n_fft(cfg, pitch_frame_length)
    hi_n_fft = resolve_hi_n_fft(cfg, pitch_frame_length)
    flux, times = compute_spectral_flux(
        y_s, sr, frame_length=cfg.frame_length, hop_length=cfg.hop_length, normalised=True
    )
    hi_status: Dict = {"half_integer_valid": False, "half_integer_bandwidth_hz": None, "n_fft": hi_n_fft}
    if cfg.regime_half_integer:
        hi_db, hi_times, hi_status = compute_half_integer_ratio_db(
            y_s, sr, f0_hz, cfg, n_fft=hi_n_fft
        )
        n = min(len(flux), len(hi_db), len(times), len(hi_times))
        flux, times, hi_db = flux[:n], times[:n], hi_db[:n]
    else:
        hi_db = np.full(len(flux), np.nan, dtype=np.float64)
        n = len(flux)
        times = times[:n]
    flux_sm = _moving_median(
        _vibrato_smooth_track(np.asarray(flux, dtype=np.float64), np.asarray(times), cfg),
        int(cfg.regime_flux_median_frames),
    )
    hi_sm = _moving_median(
        _vibrato_smooth_track(np.asarray(hi_db, dtype=np.float64), np.asarray(times), cfg),
        int(cfg.regime_flux_median_frames),
    )
    return {
        "sr": int(sr),
        "hop_length": int(cfg.hop_length),
        "n_fft": int(cfg.frame_length),
        "times": [float(t) for t in times],
        "flux": [float(v) for v in flux],
        "half_integer_ratio_db": [
            None if not np.isfinite(v) else float(v) for v in hi_db
        ],
        "flux_normalised": True,
        "half_integer_valid": bool(hi_status.get("half_integer_valid")),
        "hi_bandwidth_hz": hi_status.get("half_integer_bandwidth_hz"),
        "hi_n_fft": int(hi_status.get("n_fft") or hi_n_fft),
        "f0_hz": None if f0_hz is None else float(f0_hz),
        "pitch_frame_length": None if pitch_frame_length is None else int(pitch_frame_length),
        "flux_smoothed": [float(v) for v in flux_sm],
        "half_integer_ratio_db_smoothed": [
            None if not np.isfinite(v) else float(v) for v in hi_sm
        ],
    }


def _clamp_segment_rel(
    t_att: float, t_dec: float, active_len: float, min_sustain: float, min_decay: float = 0.02
) -> Tuple[float, float]:
    min_tail = max(min_decay, 0.01)
    t_att = max(0.0, min(t_att, active_len - min_sustain - min_tail))
    t_dec = max(t_att + min_sustain, min(t_dec, active_len - min_tail))
    if t_dec <= t_att:
        t_dec = min(t_att + min_sustain, active_len - min_tail)
    if t_dec >= active_len - 1e-4:
        t_dec = max(t_att + min_sustain, active_len - min_tail)
    return t_att, t_dec


def detect_segments_advanced_rel(
    y_trimmed: np.ndarray, sr: int, cfg: SegmentConfig, min_sustain: float
) -> Tuple[float, float]:
    rms, times = compute_rms_envelope(y_trimmed, sr, cfg.frame_length, cfg.hop_length)
    peak_idx = int(np.argmax(rms))
    flux, flux_times = compute_spectral_flux(y_trimmed, sr, cfg.frame_length, cfg.hop_length)

    attack_time = detect_attack_combined(
        rms, times, peak_idx, cfg.attack_threshold, flux, flux_times, use_derivative=True
    )
    attack_idx = min(int(np.searchsorted(times, attack_time, side="right")), len(rms) - 1)

    active_len = len(y_trimmed) / sr
    min_decay_t = min_decay_time_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.sustain_fraction_before_decay
    )
    decay_time = detect_decay_derivative(rms, times, attack_idx, peak_idx, cfg.decay_threshold)
    decay_time = max(decay_time, min_decay_t)
    decay_idx = min(int(np.searchsorted(times, decay_time, side="right")), len(rms) - 1)

    plateau = detect_sustain_plateau(
        rms, times, attack_idx, decay_idx, min_sustain, cfg.sustain_variance_threshold
    )
    if plateau[0] is not None:
        attack_time = float(times[plateau[0]])
        decay_time = float(times[plateau[1]])

    active_len = len(y_trimmed) / sr
    return _clamp_segment_rel(attack_time, decay_time, active_len, min_sustain)


def detect_segments_smart_rel(
    y_trimmed: np.ndarray, sr: int, cfg: SegmentConfig, min_sustain: float
) -> Tuple[float, float]:
    """Energy-guided boundaries blended with proportional anchors."""
    active_len = len(y_trimmed) / sr
    prop_att, prop_dec = detect_segments_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, min_sustain
    )
    rms, times = compute_rms_envelope(y_trimmed, sr, cfg.frame_length, cfg.hop_length)
    peak_idx = int(np.argmax(rms))
    energy_att = detect_attack_energy(rms, times, peak_idx, cfg.attack_threshold)
    attack_idx = min(int(np.searchsorted(times, energy_att, side="right")), len(rms) - 1)
    min_decay_t = min_decay_time_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.sustain_fraction_before_decay
    )
    energy_dec = detect_decay_energy(
        rms, times, attack_idx, peak_idx, cfg.decay_threshold, min_decay_time=min_decay_t
    )

    t_att = SMART_ENERGY_BLEND * energy_att + SMART_PROP_BLEND * prop_att
    t_dec = SMART_ENERGY_BLEND * energy_dec + SMART_PROP_BLEND * prop_dec
    t_dec = max(t_dec, min_decay_t)
    return _clamp_segment_rel(t_att, t_dec, active_len, min_sustain)


def detect_segments(
    y: np.ndarray, sr: int, cfg: SegmentConfig, file_path: Optional[Path] = None
) -> SegmentResult:
    empty_pitch = {
        "used": False,
        "std_cents": None,
        "window_start": None,
        "window_end": None,
        "window_duration": None,
        "expected_note_hz": None,
        "mean_abs_cents_from_note": None,
        "vibrato_robust": cfg.vibrato_robust,
    }
    y = preprocess_signal(y, remove_dc=cfg.remove_dc)
    try:
        y_trimmed, trim = trim_active_region(y, sr, cfg.trim_db)
        active_len = trim.active_len

        if active_len <= 0 or len(y_trimmed) < cfg.frame_length * 2:
            t_att = trim.t_start + cfg.attack_pct * active_len
            t_dec = trim.t_start + (cfg.attack_pct + cfg.sustain_pct) * active_len
            return SegmentResult(t_att, t_dec, trim.t_end, trim, empty_pitch)

        min_sustain = effective_min_sustain_duration(cfg, sr, active_len)

        if cfg.use_advanced:
            t_att_rel, t_dec_rel = detect_segments_advanced_rel(y_trimmed, sr, cfg, min_sustain)
        elif cfg.use_smart:
            t_att_rel, t_dec_rel = detect_segments_smart_rel(y_trimmed, sr, cfg, min_sustain)
        else:
            t_att_rel, t_dec_rel = detect_segments_proportional(
                active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, min_sustain
            )

        expected_hz, file_meta = parse_note_from_filename(file_path)
        t_att_rel, t_dec_rel, pitch_info = refine_sustain_by_pitch(
            y_trimmed, sr, t_att_rel, t_dec_rel, cfg, expected_hz, file_path=file_path
        )
        if file_meta.get("note_name_wrap_spelling"):
            pitch_info["note_name_wrap_spelling"] = True
        t_att_rel, t_dec_rel = _clamp_segment_rel(t_att_rel, t_dec_rel, active_len, min_sustain)
        if pitch_info.get("used"):
            pitch_info["window_start"] = trim.t_start + pitch_info["window_start"]
            pitch_info["window_end"] = trim.t_start + pitch_info["window_end"]

        regime_info: Dict = {}
        source_att_rel, source_dec_rel = t_att_rel, t_dec_rel
        if cfg.use_regime_refine:
            if pitch_info.get("failed"):
                f0_for_regime = None
            else:
                f0_for_regime = pitch_info.get("median_f0_hz") or expected_hz
            new_att, new_dec, regime_info = refine_sustain_by_regime(
                y_trimmed,
                sr,
                t_att_rel,
                t_dec_rel,
                cfg,
                f0_for_regime,
                pitch_frame_length=pitch_info.get("pitch_frame_length"),
            )
            if cfg.regime_refine_mode == "trim":
                t_att_rel, t_dec_rel = new_att, new_dec
            if regime_info.get("window_start") is not None:
                regime_info["window_start"] = trim.t_start + float(regime_info["window_start"])
                regime_info["window_end"] = trim.t_start + float(regime_info["window_end"])
            regime_info["source_att"] = trim.t_start + source_att_rel
            regime_info["source_dec"] = trim.t_start + source_dec_rel

        min_decay = max(0.02, active_len * 0.05)
        t_att = trim.t_start + t_att_rel
        t_dec = min(trim.t_start + t_dec_rel, trim.t_end - min_decay)
        t_att = min(t_att, t_dec - min(min_sustain, active_len * 0.5))
        return SegmentResult(t_att, t_dec, trim.t_end, trim, pitch_info, regime_info)

    except Exception as exc:
        logger.warning("detect_segments fallback to proportional: %s", exc, exc_info=True)
        active_len = len(y) / sr
        t_att, t_dec = detect_segments_proportional(
            active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.min_sustain_duration
        )
        trim = TrimInfo(0, len(y), 0.0, active_len - 1e-3, active_len)
        return SegmentResult(t_att, t_dec, trim.t_end, trim, empty_pitch)


def validate_segments(
    t_att: float, t_dec: float, t_end: float, min_duration: float = 0.01
) -> bool:
    if t_att >= t_dec or t_dec >= t_end:
        return False
    if t_dec - t_att < min_duration or t_end - t_dec < min_duration:
        return False
    return True


def find_zero_crossing(
    y: np.ndarray, idx: int, sr: int, search_ms: float = DEFAULT_ZERO_CROSSING_SEARCH_MS
) -> int:
    idx = max(0, min(idx, len(y) - 1))
    search_samples = int(sr * (search_ms / 1000.0))
    start = max(0, idx - search_samples)
    end = min(len(y), idx + search_samples + 1)
    chunk = y[start:end]
    if len(chunk) < 2:
        return idx
    sign_changes = np.where(np.diff(np.signbit(chunk)))[0]
    if len(sign_changes) == 0:
        expanded = min(search_samples * 2, len(y) // 4)
        start = max(0, idx - expanded)
        end = min(len(y), idx + expanded)
        chunk = y[start:end]
        if len(chunk) < 2:
            return idx
        sign_changes = np.where(np.diff(np.signbit(chunk)))[0]
        if len(sign_changes) == 0:
            return idx
    target_offset = max(0, min(idx - start, len(chunk) - 1))
    crossing_idx = sign_changes[int(np.argmin(np.abs(sign_changes - target_offset)))]
    if crossing_idx < len(chunk) - 1:
        y1, y2 = chunk[crossing_idx], chunk[crossing_idx + 1]
        t = -y1 / (y2 - y1) if abs(y2 - y1) > 1e-10 else 0.0
        exact = crossing_idx + t
    else:
        exact = float(crossing_idx)
    return max(0, min(start + int(round(exact)), len(y) - 1))


def _fade_curve(n: int, fade_type: str, rising: bool) -> np.ndarray:
    if n < 1:
        return np.array([], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    if fade_type == "linear":
        curve = t if rising else 1.0 - t
    elif fade_type == "hann":
        curve = np.hanning(n * 2)[:n] if rising else np.hanning(n * 2)[n:]
    else:
        # cosine (raised cosine)
        curve = 0.5 * (1.0 - np.cos(np.pi * t)) if rising else 0.5 * (1.0 + np.cos(np.pi * t))
    return curve.astype(np.float64)


def apply_fades(audio: np.ndarray, sr: int, fade_ms: float, fade_type: str = "cosine") -> np.ndarray:
    if len(audio) == 0:
        return audio
    fade_samples = int(sr * (fade_ms / 1000.0))
    min_fade = int(sr / 20.0)
    fade_samples = max(fade_samples, min(min_fade, len(audio) // 4))
    fade_samples = min(fade_samples, len(audio) // 2)
    if fade_samples < 1:
        return audio
    fade_in = _fade_curve(fade_samples, fade_type, rising=True)
    fade_out = _fade_curve(fade_samples, fade_type, rising=False)
    out = audio.copy()
    out[:fade_samples] *= fade_in
    out[-fade_samples:] *= fade_out
    if abs(out[0]) < 0.001:
        out[0] = 0.0
    if abs(out[-1]) < 0.001:
        out[-1] = 0.0
    return out


def edge_click_severity(audio: np.ndarray, edge_samples: int = 32) -> float:
    """Boundary discontinuity vs typical interior sample-to-sample change (0 = clean)."""
    if len(audio) < 8:
        return 0.0
    edge = min(edge_samples, len(audio) // 4, len(audio) - 1)
    mid_lo = edge
    mid_hi = max(mid_lo + 4, len(audio) - edge)
    if mid_hi <= mid_lo + 4:
        ref_diff = float(np.median(np.abs(np.diff(audio)))) + 1e-10
    else:
        ref_diff = float(np.median(np.abs(np.diff(audio[mid_lo:mid_hi])))) + 1e-10

    start_amp = abs(float(audio[0]))
    end_amp = abs(float(audio[-1]))
    start_spike = float(np.max(np.abs(np.diff(audio[: max(2, edge)])))) if edge > 1 else start_amp
    end_spike = float(np.max(np.abs(np.diff(audio[-max(2, edge) :])))) if edge > 1 else end_amp
    return max(start_amp / ref_diff, end_amp / ref_diff, start_spike / (ref_diff * 2), end_spike / (ref_diff * 2))


def verify_no_clicks(audio: np.ndarray, tolerance: float = 0.01, max_severity: float = 4.0) -> bool:
    if len(audio) < 2:
        return True
    if abs(audio[0]) > tolerance or abs(audio[-1]) > tolerance:
        return False
    return edge_click_severity(audio) <= max_severity


def extract_and_fade_segments(
    y: np.ndarray,
    sr: int,
    t_att: float,
    t_dec: float,
    t_end: float,
    trim: TrimInfo,
    fade_ms: float,
    fade_type: str,
) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    max_idx = len(y) - 1
    idx_start = max(0, min(trim.idx_start, max_idx))
    idx_att_target = max(idx_start, min(int(t_att * sr), max_idx))
    idx_dec_target = max(idx_att_target + 1, min(int(t_dec * sr), max_idx))
    idx_end_target = max(idx_dec_target + 1, min(int(t_end * sr), max_idx))

    idx_att = find_zero_crossing(y, idx_att_target, sr)
    idx_dec = find_zero_crossing(y, max(idx_att + int(0.02 * sr), idx_dec_target), sr)
    idx_end = find_zero_crossing(y, idx_end_target, sr)

    idx_att = max(idx_start, min(idx_att, max_idx))
    idx_dec = max(idx_att + 1, min(idx_dec, max_idx))
    idx_end = max(idx_dec + 1, min(idx_end, len(y)))

    attack_seg = y[idx_start:idx_att].copy()
    sustain_seg = y[idx_att:idx_dec].copy()
    decay_seg = y[idx_dec:idx_end].copy()
    release_seg = y[idx_end:].copy()
    active_sound = y[idx_start:idx_end].copy()

    parts = {
        "_Attacks": apply_fades(attack_seg, sr, fade_ms, fade_type),
        "_Sustains": apply_fades(sustain_seg, sr, fade_ms, fade_type),
        "_Decays": apply_fades(decay_seg, sr, fade_ms, fade_type),
        "_Release_Silence": release_seg,
        "_Full_Active_Sound": apply_fades(active_sound, sr, fade_ms, fade_type),
    }
    for name, seg in parts.items():
        if len(seg) > 0 and name != "_Release_Silence" and not verify_no_clicks(seg):
            parts[name] = apply_fades(seg, sr, fade_ms * 1.5, fade_type)
    return parts, idx_att, idx_dec, idx_end


def _soundfile_format_for_extension(ext: str) -> Optional[str]:
    return {
        ".wav": "WAV",
        ".aif": "AIFF",
        ".aiff": "AIFF",
        ".flac": "FLAC",
        ".ogg": "OGG",
    }.get(ext.lower())


def write_audio(output_path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf_format = _soundfile_format_for_extension(output_path.suffix)
    if sf_format:
        sf.write(output_path, audio, sr, format=sf_format)
    else:
        sf.write(output_path, audio, sr)


def list_audio_files(folder: Path) -> List[Path]:
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        and not f.stem.lower().endswith("_backup")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def process_audio_file(
    input_path: Path,
    output_dir: Path,
    cfg: SegmentConfig,
    fade_ms: float = 50.0,
    fade_type: str = "cosine",
    write_flux_sidecar: bool = False,
) -> Dict:
    """
    Headless ADSR split for one file. Writes segment folders under output_dir.
    Returns metadata dict for the processed file.
    """
    import librosa

    y, sr = librosa.load(str(input_path), sr=None)
    result = detect_segments(y, sr, cfg, file_path=input_path)
    if not validate_segments(result.t_att, result.t_dec, result.t_end):
        raise ValueError(f"Invalid segment boundaries for {input_path.name}")

    rr = result.regime_refine or {}
    # Standard folders keep energy+pitch sustain; regime trim is an extra export.
    t_att_std = float(rr.get("source_att", result.t_att))
    t_dec_std = float(rr.get("source_dec", result.t_dec))
    if not validate_segments(t_att_std, t_dec_std, result.t_end):
        t_att_std, t_dec_std = result.t_att, result.t_dec

    parts, idx_att, idx_dec, idx_end = extract_and_fade_segments(
        y, sr, t_att_std, t_dec_std, result.t_end, result.trim, fade_ms, fade_type
    )

    for folder, audio in parts.items():
        target_dir = output_dir / folder
        target_dir.mkdir(exist_ok=True, parents=True)
        if len(audio) == 0:
            continue
        if folder == "_Full_Active_Sound":
            tag = "FullActive"
        elif folder == "_Release_Silence":
            tag = "Release"
        else:
            tag = folder.strip("_")
        write_audio(target_dir / f"{input_path.stem}_{tag}{input_path.suffix}", audio, sr)

    if cfg.use_regime_refine and cfg.regime_refine_mode == "trim":
        stable_dir = output_dir / "_Sustains_Stable"
        stable_dir.mkdir(exist_ok=True, parents=True)
        parts_stable, _, _, _ = extract_and_fade_segments(
            y, sr, result.t_att, result.t_dec, result.t_end, result.trim, fade_ms, fade_type
        )
        stable_audio = parts_stable.get("_Sustains", np.array([]))
        if len(stable_audio) > 0:
            write_audio(
                stable_dir / f"{input_path.stem}_SustainStable{input_path.suffix}",
                stable_audio,
                sr,
            )

    sidecar_path = None
    if write_flux_sidecar:
        pr = result.pitch_refine or {}
        f0_hz = None if pr.get("failed") else (pr.get("median_f0_hz") or pr.get("expected_note_hz"))
        payload = build_regime_flux_sidecar(
            y, sr, cfg, t_att_std, t_dec_std, f0_hz, pitch_frame_length=pr.get("pitch_frame_length")
        )
        sidecar_path = output_dir / f"{input_path.stem}.flux.json"
        sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trim = result.trim
    idx_start = trim.idx_start
    return {
        "file_path": str(input_path),
        "sr": sr,
        "t_start": idx_start / sr,
        "t_att": idx_att / sr,
        "t_dec": idx_dec / sr,
        "t_end": idx_end / sr,
        "dur_att": (idx_att - idx_start) / sr,
        "dur_sus": (idx_dec - idx_att) / sr,
        "dur_dec": (idx_end - idx_dec) / sr,
        "dur_rel": (len(y) - idx_end) / sr,
        "pitch_refine": result.pitch_refine,
        "regime_refine": rr,
        "regime_flux_sidecar": str(sidecar_path) if sidecar_path else None,
        "detection_mode": (
            "advanced" if cfg.use_advanced else ("smart" if cfg.use_smart else "proportional")
        ),
    }


def batch_process_folder(
    folder: Path,
    cfg: SegmentConfig,
    fade_ms: float = 50.0,
    fade_type: str = "cosine",
    output_dir: Optional[Path] = None,
    write_flux_sidecar: bool = False,
) -> List[Dict]:
    """Process all audio files in folder; returns list of per-file metadata."""
    out = output_dir or folder
    results: List[Dict] = []
    for f_path in list_audio_files(folder):
        try:
            results.append(
                process_audio_file(
                    f_path, out, cfg, fade_ms, fade_type, write_flux_sidecar=write_flux_sidecar
                )
            )
        except Exception as exc:
            logger.error("Failed %s: %s", f_path.name, exc)
            results.append({"file_path": str(f_path), "error": str(exc)})
    return results
