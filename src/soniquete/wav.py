"""WAV file I/O helpers used by :class:`~soniquete.block.Block`."""

from __future__ import annotations

import wave

import numpy as np

from .config import _PCM_MAX, _PCM_MIN


def write_wav(
    path: str, array: np.ndarray, sample_rate: int, normalize: bool = True
) -> None:
    """Write samples to ``path`` as a mono 16-bit PCM WAV file.

    If ``normalize`` is True, ``array`` is treated as float samples in
    ``[-1, 1]`` and rescaled to 16-bit PCM integers. Otherwise, ``array`` is
    assumed to already be 16-bit PCM data and is written out as-is.
    """
    if normalize:
        pcm = np.clip(np.round(array * _PCM_MAX), _PCM_MIN, _PCM_MAX).astype(np.int16)
    else:
        pcm = array

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: str, normalized: bool = False) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV file at ``path``.

    Returns the samples and the file's sample rate. If ``normalized`` is
    True, the samples are returned as float64 rescaled to ``[-1, 1]`` by
    ``_PCM_MAX``. Otherwise, the raw 16-bit PCM samples are returned as-is.
    Raises :class:`ValueError` for anything other than mono, 16-bit PCM
    audio (stereo/multi-channel, or a bit depth other than 16), rather than
    silently mixing channels down or reinterpreting samples.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if channels != 1:
        raise ValueError(
            f"Block only supports mono WAV files, got {channels} channels"
        )
    if sampwidth != 2:
        raise ValueError(
            "Block only supports 16-bit PCM WAV files, got "
            f"{sampwidth * 8}-bit"
        )

    samples = np.frombuffer(raw, dtype=np.int16)
    if normalized:
        return samples.astype(np.float64) / _PCM_MAX, sample_rate
    return samples, sample_rate

