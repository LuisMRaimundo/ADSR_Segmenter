"""Register- and instrument-independent regime / pitch stage (v3.3)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_segment_core as core

IOWA_FIXTURE = ROOT / "tests" / "fixtures" / "IOWA_Trb_T_pp_C5_Sustains.aiff"


def _harmonics(sr, freq, n_or_dur, amps=(1.0, 0.4, 0.18, 0.08, 0.04)):
    n = int(n_or_dur) if n_or_dur > 50 else int(n_or_dur * sr)
    t = np.arange(n) / sr
    y = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        y += a * np.sin(2 * np.pi * k * freq * t)
    fade = min(int(0.02 * sr), len(y) // 10)
    if fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return 0.35 * y


def _adsr_tone(sr, freq, att=0.08, sus=2.0, dec=0.2, gap=0.1, amps=(1.0, 0.4, 0.18, 0.08)):
    n_att, n_sus, n_dec, n_gap = int(att * sr), int(sus * sr), int(dec * sr), int(gap * sr)
    env = np.concatenate(
        [np.linspace(0, 1, n_att, endpoint=False), np.ones(n_sus), np.linspace(1, 0, n_dec)]
    )
    body = _harmonics(sr, freq, len(env), amps) / 0.35
    return np.concatenate([np.zeros(n_gap), body * env, np.zeros(n_gap)])


def _add_hi(y, sr, freq, start_s, dur_s, level_db=-20.0, modulate=True):
    out = y.copy()
    i0 = int(start_s * sr)
    n = min(len(out) - i0, int(dur_s * sr))
    if n <= 0:
        return out
    t = np.arange(n) / sr
    amp = 10.0 ** (level_db / 20.0)
    am = (0.55 + 0.45 * np.sin(2 * np.pi * 14.0 * t)) if modulate else 1.0
    out[i0 : i0 + n] += amp * am * (
        np.sin(2 * np.pi * 1.5 * freq * t) + np.sin(2 * np.pi * 2.5 * freq * t)
    )
    return out


@pytest.fixture
def sr():
    return 22050


def test_hi_band_scales_and_resolution_gate(sr):
    cfg = core.SegmentConfig()
    # n_fft large enough that α·f0 ≥ 2·(sr/n_fft) at every listed f0, including 82 Hz.
    n_fft = 4096
    for f0 in (82.0, 262.0, 1046.0, 3520.0):
        y = _harmonics(sr, f0, 1.2)
        ratio, _times, status = core.compute_half_integer_ratio_db(y, sr, f0, cfg, n_fft=n_fft)
        assert status["half_integer_valid"] is True
        assert status["half_integer_bandwidth_hz"] == pytest.approx(0.15 * f0, rel=1e-9)
        mid = float(np.nanmedian(ratio[np.isfinite(ratio)]))
        assert mid <= -25.0, f"f0={f0} middle={mid}"

    y40 = _harmonics(sr, 40.0, 1.2)
    _ratio, _t, st = core.compute_half_integer_ratio_db(y40, sr, 40.0, cfg, n_fft=1024)
    assert st["half_integer_valid"] is False
    assert st["reason"] == "band_below_resolution"


def test_tail_regime_hi_walk_not_flux(sr):
    freq = 523.25
    y = _adsr_tone(sr, freq, att=0.06, sus=2.2, dec=0.2, gap=0.05)
    y_trim, _ = core.trim_active_region(y, sr)
    # last 0.3 s of the pitch-stable sustain (before t_dec), not the release tail
    t_att, t_dec = 0.06, 2.15
    y_trim = _add_hi(y_trim, sr, freq, start_s=t_dec - 0.30, dur_s=0.30, modulate=False)
    cfg = core.SegmentConfig(
        regime_refine_mode="trim",
        regime_min_duration=0.5,
        regime_min_windows=4,
        regime_use_half_integer=True,
        regime_hi_rise_db=10.0,
    )
    _att, _dec, info = core.refine_sustain_by_regime(y_trim, sr, t_att, t_dec, cfg, freq)
    assert info["boundary_source_end"] == "half_integer"
    assert info["hi_trimmed_end_s"] >= 0.2
    assert info["trimmed_end_s"] >= 0.2
    assert (info.get("flux_edge_ratio_end") or 1.0) < float(cfg.regime_flux_ratio)


def test_steady_multiphonic_not_trimmed(sr):
    freq = 440.0
    y = _adsr_tone(sr, freq, sus=2.0)
    y_trim, _ = core.trim_active_region(y, sr)
    # Steady HI through the sustain only — attack/release stay harmonic.
    y_trim = _add_hi(y_trim, sr, freq, start_s=0.08, dur_s=2.10, level_db=-20.0, modulate=False)
    cfg = core.SegmentConfig(regime_refine_mode="trim", regime_min_duration=0.5, regime_min_windows=4)
    new_att, new_dec, info = core.refine_sustain_by_regime(y_trim, sr, 0.10, 2.05, cfg, freq)
    assert info["refused"] is False
    assert info["trimmed_start_s"] < 0.08
    assert info["trimmed_end_s"] < 0.08
    assert abs(new_att - 0.10) < 0.08
    assert abs(new_dec - 2.05) < 0.08


def test_vibrato_tremolo_not_trimmed(sr):
    freq = 440.0
    dur = 2.2
    t = np.arange(int(dur * sr)) / sr
    inst = np.cumsum(freq * (2 ** ((20.0 / 1200.0) * np.sin(2 * np.pi * 6.0 * t))) / sr)
    env = 10 ** ((2.0 / 20.0) * np.sin(2 * np.pi * 6.0 * t) / 2.0)
    y = 0.35 * np.sin(2 * np.pi * inst) * env
    y[: int(0.04 * sr)] *= np.linspace(0, 1, int(0.04 * sr))
    cfg = core.SegmentConfig(
        regime_refine_mode="trim",
        regime_vibrato_robust=True,
        vibrato_robust=True,
        regime_min_duration=0.5,
        regime_min_windows=4,
    )
    _a, _d, info = core.refine_sustain_by_regime(y, sr, 0.08, 2.05, cfg, freq)
    assert info["trimmed_start_s"] < 0.08
    assert info["trimmed_end_s"] < 0.08
    assert info["flux_edge_ratio_start"] <= 1.3
    assert info["flux_edge_ratio_end"] <= 1.3


def test_level_independent_flux_reference(sr):
    y0 = _adsr_tone(sr, 440.0, sus=1.8)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12)
    y40 = y0 * (10 ** (-40.0 / 20.0))
    cfg = core.SegmentConfig(regime_refine_mode="annotate", regime_min_duration=0.5, regime_min_windows=4)
    _, _, a = core.refine_sustain_by_regime(y0, sr, 0.10, 1.85, cfg, 440.0)
    _, _, b = core.refine_sustain_by_regime(y40, sr, 0.10, 1.85, cfg, 440.0)
    assert a["flux_reference"] == pytest.approx(b["flux_reference"], rel=1e-3)
    assert a["window_start"] == pytest.approx(b["window_start"], abs=1e-6)
    assert a["window_end"] == pytest.approx(b["window_end"], abs=1e-6)


def test_pitch_range_from_filename_and_octave_error(sr):
    freq = 82.41
    t = np.arange(int(2.0 * sr)) / sr
    # 82 Hz series plus a subharmonic and a 200 Hz partial. Unconstrained
    # yin locks near 41 Hz; the filename ±1-octave gate keeps E2.
    y = 0.6 * (
        np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    )
    y += 0.16 * np.sin(2 * np.pi * 41.2 * t)
    y += 0.08 * np.sin(2 * np.pi * 200.0 * t)
    y[: int(0.03 * sr)] *= np.linspace(0, 1, int(0.03 * sr))
    cfg_ok = core.SegmentConfig(pitch_refine_mode="annotate", min_sustain_duration=0.2)
    r_ok = core.detect_segments(y, sr, cfg_ok, file_path=Path("Iowa_Trb_E2.wav"))
    assert r_ok.pitch_refine.get("failed") is not True
    assert r_ok.pitch_refine.get("std_cents") is not None
    assert r_ok.pitch_refine["std_cents"] < 5.0

    cfg_fail = core.SegmentConfig(
        pitch_refine_mode="annotate",
        min_sustain_duration=0.2,
        pitch_fail_cents=50.0,
        pitch_fmin=30.0,
        pitch_fmax=4200.0,
    )
    _a, _d, info = core.refine_sustain_by_pitch(
        y, sr, 0.05, 1.8, cfg_fail, expected_note_hz=82.41, search_range=(30.0, 4200.0)
    )
    assert info["failed"] is True
    assert info["failed_reason"] == "octave_error"
    assert info["kept_energy_boundaries"] is True
    cfg_fail.use_regime_refine = True
    _na, _nd, rinfo = core.refine_sustain_by_regime(y, sr, 0.05, 1.8, cfg_fail, f0_hz=None)
    assert rinfo["half_integer_valid"] is False


def test_unvoiced_noise_keeps_energy(sr):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, int(1.5 * sr))
    # band-limit
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / sr)
    spec[(freqs < 400) | (freqs > 4000)] = 0
    y = np.fft.irfft(spec, n=len(noise)).astype(np.float64)
    y *= np.hanning(len(y))
    cfg = core.SegmentConfig(min_sustain_duration=0.15, pitch_refine_mode="annotate")
    result = core.detect_segments(y, sr, cfg, file_path=Path("noise_burst.wav"))
    assert result.pitch_refine.get("failed") is True
    assert result.pitch_refine.get("failed_reason") == "unvoiced"
    assert result.regime_refine.get("used") is True


def test_filename_wrap_spelling_and_mismatch(sr):
    y = _adsr_tone(sr, 493.88, sus=1.4)  # B4
    cfg = core.SegmentConfig(min_sustain_duration=0.2, pitch_refine_mode="annotate")
    result = core.detect_segments(y, sr, cfg, file_path=Path("IOWA_Trb.T_pp.B#4.aif"))
    assert result.pitch_refine.get("note_name_wrap_spelling") is True
    assert result.pitch_refine.get("note_name_mismatch") is True
    assert result.pitch_refine.get("median_f0_hz") == pytest.approx(493.88, rel=0.03)


def test_floor_scales_with_register(sr):
    cfg = core.SegmentConfig(regime_analysis_n_fft=None)
    low = core.detect_segments(
        _adsr_tone(sr, 82.41, sus=2.0), sr, cfg, file_path=Path("Bass_E2.wav")
    )
    high = core.detect_segments(
        _adsr_tone(sr, 1046.5, sus=2.0), sr, cfg, file_path=Path("Picc_C6.wav")
    )
    n_low = low.regime_refine.get("analysis_n_fft")
    n_high = high.regime_refine.get("analysis_n_fft")
    assert n_low is not None and n_high is not None
    assert n_low >= 4 * n_high
    assert low.regime_refine.get("analysis_n_fft_source") == "pitch_frame"
    assert high.regime_refine.get("analysis_n_fft_source") == "pitch_frame"


def test_sidecar_has_new_keys(sr):
    y = _adsr_tone(sr, 440.0, sus=1.2)
    cfg = core.SegmentConfig()
    payload = core.build_regime_flux_sidecar(y, sr, cfg, 0.1, 1.3, 440.0, pitch_frame_length=2048)
    assert payload["flux_normalised"] is True
    assert "half_integer_valid" in payload
    assert "hi_bandwidth_hz" in payload
    assert payload["f0_hz"] == 440.0
    assert payload["pitch_frame_length"] == 2048
    assert payload["hi_n_fft"] == 2048
    assert len(payload["flux_smoothed"]) == len(payload["flux"])
    assert len(payload["half_integer_ratio_db_smoothed"]) == len(payload["flux"])


def test_floor_refuses_trim_but_fills_diagnostics(sr):
    # 82 Hz, ~1.2 s sustain: 8 × 4096 / 22050 ≈ 1.49 s floor.
    y = _adsr_tone(sr, 82.41, att=0.06, sus=1.15, dec=0.12, gap=0.05)
    y_trim, _ = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(
        regime_refine_mode="trim",
        regime_min_windows=8,
        regime_min_duration=1.0,
    )
    t_att, t_dec = 0.08, 1.18
    new_att, new_dec, info = core.refine_sustain_by_regime(
        y_trim, sr, t_att, t_dec, cfg, 82.41, pitch_frame_length=4096
    )
    assert info["refused"] is True
    assert info["refused_reason"] == "span_below_floor"
    assert new_att == t_att and new_dec == t_dec
    assert info["flux_reference"] is not None
    assert info["flux_edge_ratio_start"] is not None
    assert info["flux_edge_ratio_end"] is not None
    assert info["window_start"] is not None
    assert info["window_end"] is not None
    assert info["hi_reference_db"] is not None
    assert info["hi_edge_rise_db_start"] is not None
    assert info["hi_edge_rise_db_end"] is not None


def test_low_register_hi_uses_pitch_frame(sr):
    y = _harmonics(sr, 82.0, 1.4)
    cfg = core.SegmentConfig()
    _a, _d, info = core.refine_sustain_by_regime(y, sr, 0.05, 1.30, cfg, 82.0)
    assert info["hi_n_fft"] >= 4096
    assert info["half_integer_valid"] is True
    assert info["half_integer_invalid_reason"] is None


def test_normalised_threshold_hook(sr):
    y = _adsr_tone(sr, 440.0, att=0.08, sus=2.0, dec=0.2)
    y_trim, _ = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(regime_refine_mode="trim", regime_min_duration=0.5, regime_min_windows=4)
    _a0, _d0, norm = core.refine_sustain_by_regime(
        y_trim, sr, 0.10, 2.05, cfg, 440.0, flux_normalised=True
    )
    _a1, _d1, raw = core.refine_sustain_by_regime(
        y_trim, sr, 0.10, 2.05, cfg, 440.0, flux_normalised=False
    )
    assert norm["flux_normalised"] is True
    assert raw["flux_normalised"] is False
    assert norm["flux_ratio_applied"] == pytest.approx(1.5)
    assert raw["flux_ratio_applied"] == pytest.approx(2.0)
    assert norm["flux_ratio_applied"] != raw["flux_ratio_applied"]
    assert norm["trimmed_start_s"] < 0.08 and norm["trimmed_end_s"] < 0.08
    assert raw["trimmed_start_s"] < 0.08 and raw["trimmed_end_s"] < 0.08
    assert abs(_a0 - 0.10) < 0.08 and abs(_a1 - 0.10) < 0.08


@pytest.mark.skipif(not IOWA_FIXTURE.exists(), reason="IOWA trombone fixture not present")
def test_iowa_trombone_c5_flux_start_hi_end():
    y, sr = __import__("librosa").load(str(IOWA_FIXTURE), sr=None)
    cfg = core.SegmentConfig.from_preset("soft_high_brass", regime_refine_mode="trim")
    result = core.detect_segments(y, sr, cfg, file_path=IOWA_FIXTURE)
    info = result.regime_refine
    assert 0.1 <= info["trimmed_start_s"] <= 0.35
    assert info["trimmed_end_s"] >= 0.05
    assert info["boundary_source_end"] == "half_integer"
