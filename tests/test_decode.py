"""動画の連続復号(VideoFrameReader)と音声の一括変換(AudioTrack)のテスト。"""

from __future__ import annotations

from conftest import make_video, needs_ffmpeg

from mediasearch import media


@needs_ffmpeg
def test_reader_returns_frames_at_fixed_interval(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, "red", seconds=3)
    reader = media.VideoFrameReader(video, step_ms=500, max_side=160)
    frames = list(reader)
    assert [t for t, _ in frames][:3] == [0, 500, 1000] and 5 <= len(frames) <= 7
    assert frames[0][1].shape == (120, 160, 3)  # 長辺 160 に縮小、縦横比は保つ
    assert frames[0][1][..., 0].mean() > 200  # 赤
    assert reader.decoder == "cpu"


@needs_ffmpeg
def test_gpu_decode_falls_back_to_cpu(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, "blue", seconds=2)
    # GPU の無い環境で GPU 復号を指定しても、CPU でやり直して取り込みは止まらない
    reader = media.VideoFrameReader(video, step_ms=1000, max_side=64, hwaccel="vaapi", hwaccel_device="/dev/null")
    assert len(list(reader)) >= 1 and reader.decoder == "cpu"
    assert media.select_hwaccel("none") == (None, None)


@needs_ffmpeg
def test_audio_track_slices(tmp_path):
    video = tmp_path / "a.mp4"
    make_video(video, "green", seconds=3, audio="tone")
    track = media.AudioTrack(video, tmp_path)
    try:
        pcm = track.slice(1000, 2000)
        assert abs(pcm.size - 16000) <= 1 and float(abs(pcm).max()) > 0.1
        assert track.slice(10_000, 11_000).size == 0  # 範囲外は空
    finally:
        track.close()


@needs_ffmpeg
def test_skip_modes_still_return_frames(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, "red", seconds=2)
    for skip in ("none", "noref", "keyframes"):
        assert len(list(media.VideoFrameReader(video, step_ms=500, max_side=64, skip=skip))) >= 3
