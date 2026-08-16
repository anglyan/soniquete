import numpy as np
import pytest

from soniquete.wav import _PCM_MAX
from soniquete.wav import read_wav, write_wav


def test_read_wav_returns_raw_pcm_by_default(tmp_path):
    path = tmp_path / "out.wav"
    write_wav(path=str(path), array=np.array([0.5, -0.5, 0.25]), sample_rate=8000)

    samples, sample_rate = read_wav(str(path))
    assert sample_rate == 8000
    assert samples.dtype == np.int16
    assert samples == pytest.approx(
        [round(0.5 * _PCM_MAX), round(-0.5 * _PCM_MAX), round(0.25 * _PCM_MAX)]
    )


def test_read_wav_normalized_rescales_by_pcm_max(tmp_path):
    path = tmp_path / "out.wav"
    write_wav(path=str(path), array=np.array([0.5, -0.5, 0.25]), sample_rate=8000)

    samples, sample_rate = read_wav(str(path), normalized=True)
    assert sample_rate == 8000
    assert samples.dtype == np.float64
    assert samples == pytest.approx([0.5, -0.5, 0.25], abs=1e-4)
