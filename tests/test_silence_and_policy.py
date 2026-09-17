"""Silence rejection, batch continuation, and short-file sustain policy."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_segment_core as core
import split_audio_cli


def _adsr(sr, att=0.08, sus=0.8, dec=0.2, gap=0.1, freq=440.0):
    n_a, n_s, n_d, n_g = int(att * sr), int(sus * sr), int(dec * sr), int(gap * sr)
    env = np.concatenate(
        [np.linspace(0, 1, n_a, endpoint=False), np.ones(n_s), np.linspace(1, 0, n_d)]
    )
    tone = 0.45 * np.sin(2 * np.pi * freq * np.arange(len(env)) / sr) * env
    return np.concatenate([np.zeros(n_g), tone, np.zeros(n_g)])


def test_digital_silence_rejected_no_export(tmp_path, sr=22050):
    cfg = core.SegmentConfig.from_preset("Medium (1.5-3.0s)")
    y = np.zeros(sr, dtype=np.float32)
    result = core.detect_segments(y, sr, cfg)
    assert core.describe_rejection(y, result) == core.REJECTION_NO_ACTIVE_ENERGY
    inp = tmp_path / "silence.wav"
    sf.write(inp, y, sr)
    with pytest.raises(ValueError, match="no_active_energy"):
        core.process_audio_file(inp, tmp_path / "out", cfg, fade_ms=30.0)
    assert not list((tmp_path / "out").glob("*/*")) if (tmp_path / "out").exists() else True


def test_cli_mixed_silence_and_valid(tmp_path):
    sr = 22050
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    sf.write(src / "ok_A4.wav", _adsr(sr), sr)
    sf.write(src / "silence.wav", np.zeros(sr, dtype=np.float32), sr)
    code = split_audio_cli.main(["-f", str(src), "-o", str(out), "--export-metadata"])
    assert code == 2
    assert list((out / "_Attacks").glob("ok_A4*"))
    assert not list(out.rglob("silence_*"))
    meta = (out / "segmentation_metadata.json").read_text(encoding="utf-8")
    assert "no_active_energy" in meta
    assert "ok_A4" in meta


def test_cli_all_silent(tmp_path):
    sr = 22050
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    sf.write(src / "a.wav", np.zeros(sr, dtype=np.float32), sr)
    sf.write(src / "b.wav", np.zeros(sr, dtype=np.float32), sr)
    code = split_audio_cli.main(["-f", str(src), "-o", str(out), "--export-metadata"])
    assert code == 2
    assert not list(out.glob("_Attacks/*")) if (out / "_Attacks").exists() else True


def test_short_file_boundaries_ordered(sr=22050):
    hop_s = core.DEFAULT_HOP_LENGTH / sr
    floor_s = 40 * hop_s
    cfg = core.SegmentConfig.from_preset("Medium (1.5-3.0s)")
    for active in (floor_s * 0.6, floor_s, floor_s * 1.3):
        y = _adsr(sr, att=0.04, sus=max(0.05, active - 0.08), dec=0.04, gap=0.05)
        result = core.detect_segments(y, sr, cfg)
        assert result.t_att < result.t_dec < result.t_end
        assert result.trim.t_start <= result.t_att
        assert result.boundary_policy["decay_definition"] == "energy_threshold_after_peak"
        assert result.boundary_policy["min_sustain_frames_s"] == pytest.approx(floor_s, rel=1e-6)
        if result.trim.active_len < floor_s:
            assert result.boundary_policy["frame_floor_limits_short_file"] is True


def test_gui_snapshot_process_file_without_mainloop(tmp_path):
    """Batch worker must use a UI-thread snapshot, not live Tk .get()."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk display unavailable")

    import split_audio_segments as gui

    sr = 22050
    src = tmp_path / "in"
    src.mkdir()
    sf.write(src / "ok_A4.wav", _adsr(sr), sr)
    sf.write(src / "silence.wav", np.zeros(sr, dtype=np.float32), sr)
    app = gui.ADSRSegmenter(root)
    app.source_folder.set(str(src))
    app._snapshot_run_settings()
    root.destroy()

    out = tmp_path / "out"
    out.mkdir()
    ok_msg = app.process_file(src / "ok_A4.wav", out)
    sil_msg = app.process_file(src / "silence.wav", out)
    assert "Att:" in ok_msg
    assert "no_active_energy" in sil_msg
    assert list((out / "_Attacks").glob("ok_A4*"))
    assert not list(out.rglob("silence_*"))
    assert any("no_active_energy" in f.get("error", "") for f in app.batch_failures)


def test_no_sustain_still_gets_operational_interval(sr=22050):
    n_a, n_r, n_g = int(0.05 * sr), int(0.25 * sr), int(0.08 * sr)
    env = np.concatenate([np.linspace(0, 1, n_a, endpoint=False), np.linspace(1, 0, n_r)])
    y = np.concatenate(
        [np.zeros(n_g), 0.4 * np.sin(2 * np.pi * 523 * np.arange(len(env)) / sr) * env, np.zeros(n_g)]
    )
    cfg = core.SegmentConfig.from_preset("Very Short (< 0.5s)")
    result = core.detect_segments(y, sr, cfg)
    assert result.t_dec - result.t_att >= 0.02
    assert result.boundary_policy["sustain_assignment"] in {
        "operational_clamp",
        "operational_proportional",
        "detector",
        "pitch_refined",
    }
