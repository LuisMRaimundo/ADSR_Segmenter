"""Tests for the spectral-regime refinement stage."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_segment_core as core

IOWA_FIXTURE = ROOT / "tests" / "fixtures" / "IOWA_Trb_T_pp_C5_Sustains.aiff"


def _harmonic_adsr(
    sr: int,
    freq: float,
    attack_s: float,
    sustain_s: float,
    decay_s: float,
    gap_s: float = 0.15,
    harmonics=(1.0, 0.45, 0.22, 0.10),
):
    n_att = int(attack_s * sr)
    n_sus = int(sustain_s * sr)
    n_dec = int(decay_s * sr)
    n_gap = int(gap_s * sr)
    att = np.linspace(0, 1, n_att, endpoint=False) if n_att else np.array([])
    sus = np.ones(n_sus)
    dec = np.linspace(1, 0, n_dec, endpoint=True) if n_dec else np.array([])
    env = np.concatenate([x for x in (att, sus, dec) if len(x)])
    t = np.arange(len(env)) / sr
    tone = np.zeros_like(t)
    for k, amp in enumerate(harmonics, start=1):
        tone += amp * np.sin(2 * np.pi * k * freq * t)
    tone *= 0.35 * env
    return np.concatenate([np.zeros(n_gap), tone, np.zeros(n_gap)])


def _add_half_integer_regime(y: np.ndarray, sr: int, freq: float, start_s: float, dur_s: float, level_db: float = -20.0):
    """Add amplitude-modulated 1.5·f0 / 2.5·f0 so spectral flux stays high in [start, start+dur]."""
    out = y.copy()
    i0 = int(start_s * sr)
    n = int(dur_s * sr)
    i1 = min(len(out), i0 + n)
    n = i1 - i0
    if n <= 0:
        return out
    t = np.arange(n) / sr
    amp = 10.0 ** (level_db / 20.0)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 14.0 * t) + 0.25 * np.sin(2 * np.pi * 27.0 * t)
    extra = amp * am * (
        np.sin(2 * np.pi * 1.5 * freq * t) + np.sin(2 * np.pi * 2.5 * freq * t)
    )
    rng = np.random.default_rng(7)
    extra += 0.15 * amp * rng.normal(0.0, 1.0, n) * np.linspace(1.0, 0.2, n)
    out[i0:i1] += extra
    return out


def _add_burst(y: np.ndarray, sr: int, freq: float, at_s: float, dur_s: float = 0.10, level_db: float = -14.0):
    return _add_half_integer_regime(y, sr, freq, at_s, dur_s, level_db=level_db)


@pytest.fixture
def sr():
    return 22050


def test_steady_harmonic_no_trim(sr):
    y = _harmonic_adsr(sr, 440.0, 0.08, 2.0, 0.25)
    y_trim, _ = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(
        use_smart=True,
        use_pitch_refine=True,
        pitch_refine_mode="annotate",
        regime_refine_mode="trim",
        min_sustain_duration=0.2,
    )
    t_att, t_dec = 0.10, 2.05
    new_att, new_dec, info = core.refine_sustain_by_regime(y_trim, sr, t_att, t_dec, cfg, 440.0)
    assert info["used"] is True
    assert info["refused"] is False
    assert abs(new_att - t_att) < 0.05
    assert abs(new_dec - t_dec) < 0.05
    assert info["window_start"] == pytest.approx(new_att, abs=1e-9)
    assert info["window_end"] == pytest.approx(new_dec, abs=1e-9)
    assert info["trimmed_start_s"] < 0.08
    assert info["trimmed_end_s"] < 0.08


def test_unstable_onset_trimmed_burst_kept(sr):
    y = _harmonic_adsr(sr, 523.25, 0.06, 2.2, 0.20, gap_s=0.05)
    y = _add_half_integer_regime(y, sr, 523.25, 0.05, 0.40, level_db=-20.0)
    y = _add_burst(y, sr, 523.25, 1.15, dur_s=0.10)
    y_trim, trim = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(
        use_smart=True,
        regime_refine_mode="trim",
        regime_flux_ratio=2.0,
        min_sustain_duration=0.2,
        regime_min_duration=0.6,
        regime_min_windows=4,
    )
    t_att, t_dec = 0.06, 2.15
    new_att, new_dec, info = core.refine_sustain_by_regime(y_trim, sr, t_att, t_dec, cfg, 523.25)
    assert info["used"] is True
    assert info["refused"] is False
    assert info["trimmed_start_s"] >= 0.3
    # Interior burst must not pull the end inward past the late sustain.
    assert new_dec >= 1.6
    assert info["flux_edge_ratio_start"] > cfg.regime_flux_ratio


def test_short_span_refused_by_floor(sr):
    y = _harmonic_adsr(sr, 440.0, 0.04, 0.55, 0.12, gap_s=0.05)
    y = _add_half_integer_regime(y, sr, 440.0, 0.05, 0.40, level_db=-20.0)
    y_trim, _ = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(
        regime_refine_mode="trim",
        regime_analysis_n_fft=16384,
        regime_min_windows=20,
        regime_min_duration=1.0,
    )
    t_att, t_dec = 0.04, 0.72
    new_att, new_dec, info = core.refine_sustain_by_regime(y_trim, sr, t_att, t_dec, cfg, 440.0)
    assert info["refused"] is True
    assert info["refused_reason"] == "span_below_floor"
    assert new_att == t_att
    assert new_dec == t_dec


def test_annotate_keeps_boundaries_reports_window(sr):
    y = _harmonic_adsr(sr, 523.25, 0.06, 2.2, 0.20, gap_s=0.05)
    y = _add_half_integer_regime(y, sr, 523.25, 0.05, 0.40, level_db=-20.0)
    y_trim, _ = core.trim_active_region(y, sr)
    cfg = core.SegmentConfig(
        regime_refine_mode="annotate",
        regime_flux_ratio=2.0,
        regime_min_duration=0.6,
        regime_min_windows=4,
    )
    t_att, t_dec = 0.06, 2.15
    new_att, new_dec, info = core.refine_sustain_by_regime(y_trim, sr, t_att, t_dec, cfg, 523.25)
    assert new_att == t_att
    assert new_dec == t_dec
    assert info["window_start"] is not None and info["window_end"] is not None
    assert info["window_start"] > t_att
    assert info["trimmed_start_s"] >= 0.3


def test_regime_off_empty_and_byte_identical(sr):
    y = _harmonic_adsr(sr, 440.0, 0.08, 1.4, 0.25)
    cfg_off = core.SegmentConfig(use_smart=True, min_sustain_duration=0.1, use_regime_refine=False)
    cfg_ann = core.SegmentConfig(
        use_smart=True, min_sustain_duration=0.1, use_regime_refine=True, regime_refine_mode="annotate"
    )
    r_off = core.detect_segments(y, sr, cfg_off)
    r_ann = core.detect_segments(y, sr, cfg_ann)
    assert r_off.regime_refine == {}
    assert abs(r_off.t_att - r_ann.t_att) < 1e-9
    assert abs(r_off.t_dec - r_ann.t_dec) < 1e-9
    assert abs(r_off.t_end - r_ann.t_end) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "off"
        b = Path(tmp) / "ann"
        inp = Path(tmp) / "note_A4.wav"
        sf.write(inp, y, sr)
        a.mkdir()
        b.mkdir()
        core.process_audio_file(inp, a, cfg_off, fade_ms=30.0)
        core.process_audio_file(inp, b, cfg_ann, fade_ms=30.0)
        for folder in ("_Attacks", "_Sustains", "_Decays"):
            wa = next((a / folder).glob("*.wav"))
            wb = next((b / folder).glob("*.wav"))
            ya, sra = sf.read(wa)
            yb, srb = sf.read(wb)
            assert sra == srb
            assert ya.tobytes() == yb.tobytes()
        assert not (b / "_Sustains_Stable").exists()


def test_flux_sidecar_round_trip(sr):
    y = _harmonic_adsr(sr, 440.0, 0.08, 1.6, 0.25)
    cfg = core.SegmentConfig(use_smart=True, min_sustain_duration=0.1, regime_half_integer=True)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "note_A4.wav"
        sf.write(inp, y, sr)
        meta = core.process_audio_file(inp, Path(tmp), cfg, fade_ms=25.0, write_flux_sidecar=True)
        sidecar = Path(meta["regime_flux_sidecar"])
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert len(data["times"]) == len(data["flux"]) == len(data["half_integer_ratio_db"])
        assert data["hop_length"] == cfg.hop_length
        assert data["n_fft"] == cfg.frame_length
        assert data["sr"] == sr


@pytest.mark.skipif(not IOWA_FIXTURE.exists(), reason="IOWA trombone fixture not present")
def test_iowa_trombone_c5_regression():
    y, sr = __import__("librosa").load(str(IOWA_FIXTURE), sr=None)
    cfg_trim = core.SegmentConfig.from_preset("soft_high_brass", regime_refine_mode="trim")
    cfg_ann = core.SegmentConfig.from_preset("soft_high_brass", regime_refine_mode="annotate")
    r_trim = core.detect_segments(y, sr, cfg_trim, file_path=IOWA_FIXTURE)
    r_ann = core.detect_segments(y, sr, cfg_ann, file_path=IOWA_FIXTURE)
    info = r_trim.regime_refine
    assert 0.10 <= info["trimmed_start_s"] <= 0.50
    assert info["trimmed_end_s"] >= 0.05
    assert abs(r_ann.t_att - r_ann.regime_refine.get("source_att", r_ann.t_att)) < 1e-6
    assert abs(r_ann.t_dec - r_ann.regime_refine.get("source_dec", r_ann.t_dec)) < 1e-6
