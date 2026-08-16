"""Core building block for soniquete: a single soundwave backed by numpy."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .wav import _DEFAULT_SAMPLE_RATE
from .player import play_wav
from .shapes import FadeEnvelope,  _DEFAULT_WINDOW_RISE_TIME
from .wav import read_wav, write_wav


def normalize_array(array: np.ndarray, normalize: bool) -> np.ndarray:
    """Convert arbitrary input samples to normalized float64 in [-1, 1]."""
    arr = np.array(array, dtype=np.float64)  # copy: never mutate the caller's array
    if arr.ndim != 1:
        raise ValueError("Block only supports single-channel (1D) arrays")

    if normalize:
        max_value = float(np.max(np.abs(arr))) if arr.size else 0.0
        if max_value > 1:
            arr /= max_value
        return arr
    else:
        return np.clip(arr, -1.0, 1.0)


class Block:
    """A single-channel soundwave stored as a float64 in the [-1, 1] range.

    Block takes a 1D array and a sample_rate and generate a sound waveform. Block
    automatically applies a fade at the beginning and the end of the array to
    prevent pops and clicks.

    Blocks can be exported as WAV objects or directly played.
    
    """

    def __init__(
        self,
        array: np.ndarray | None = None,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        duration: float | None = None,
        normalize: bool = True,
        apply_tapper: bool=True,
        rise_time: float = _DEFAULT_WINDOW_RISE_TIME
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self._normalize = normalize
        self._rise_time = rise_time
        
        if array is None:
            self._sample_rate = int(sample_rate)
            if duration is not None:
                if duration < 0:
                    raise ValueError("duration must be non-negative")
                length = int(round(duration * self._sample_rate))
                self._array = np.zeros(length, dtype=np.float64)
            else:
                self._array = np.array([], dtype=np.float64)
        else:
            if duration is not None:
                raise ValueError("cannot set duration when passing an array")
            self._sample_rate = int(sample_rate)
            self._array = normalize_array(array, self._normalize)
            if self.apply_taper:
                self.apply_taper()


    @property
    def sample_rate(self) -> int:
        """The sample rate (in Hz) this soundwave is stored at."""
        return self._sample_rate

    @property
    def array(self) -> np.ndarray:
        """The underlying float64 numpy array, with samples in [-1, 1]."""
        return self._array

    @property
    def duration(self) -> float:
        """Length of the soundwave, in seconds."""
        return len(self._array) / self._sample_rate

    @property
    def taper_rise_time(self) -> float:
        """The taper rise time """
        return self._rise_time


    def set_volume(self, value: float) -> "Block":
        """Rescale the waveform so its peak amplitude is ``value`` * full scale.

        ``value`` must be between 0 (silence) and 1 (loudest representable
        signal without clipping). Modifies the block in place and returns it
        for chaining.
        """
        if not 0 <= value <= 1:
            raise ValueError("volume must be between 0 and 1")
        if self._array.size == 0:
            return self

        peak = float(np.max(np.abs(self._array)))
        if peak == 0:
            return self  # silence: nothing to scale

        scale = value / peak
        self._array = np.clip(self._array * scale, -1.0, 1.0)
        return self

    def apply_taper(self) -> "Block":
        """Taper the start and end of the wave to zero to prevent clicks.

        A soundwave that starts or ends away from zero produces an audible
        click/pop on playback. This applies a raised-cosine (Hann) fade
        in/out envelope — 0 -> 1 over ``rise_time`` seconds at the start,
        and 1 -> 0 over ``rise_time`` seconds at the end — chosen because it
        is smooth (zero slope at both ends) unlike a linear ramp. The
        envelope itself is :class:`~soniquete.shapes.FadeEnvelope`.

        """

        n = len(self._array)

        if n > 0:
            envelope = FadeEnvelope(duration=self.duration, rise_time=self._rise_time)
            self._array = np.clip(envelope(self._array, self._sample_rate), -1.0, 1.0)
        return self


    def insert(self, block: "Block", start_time: float, normalize=True) -> "Block":
        """Mix ``block``'s waveform into this one, starting at ``start_time``.

        The two waveforms are added sample-by-sample (superimposed), not
        replaced — any overlapping audio already present is preserved and
        summed with the inserted block. If ``block`` extends past the current
        end of this waveform, this block is zero-padded to make room.
        Modifies the block in place and returns it for chaining. If normalize is True,
        the resulting array is normalized to -1 to 1. Otherwise it is clipped.
        """
        if not isinstance(block, Block):
            raise TypeError("block must be a Block instance")
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if block.sample_rate != self._sample_rate:
            raise ValueError(
                "Cannot insert a block with a different sample_rate "
                f"({block.sample_rate} != {self._sample_rate})"
            )
        if len(block) == 0:
            return self

        start_sample = int(round(start_time * self._sample_rate))
        end_sample = start_sample + len(block)

        if end_sample > len(self._array):
            extended = np.zeros(end_sample, dtype=np.float64)
            extended[: len(self._array)] = self._array
            self._array = extended

        self._array[start_sample:end_sample] += block.array
        if normalize:
            peak = float(np.max(np.abs(self._array)))
            if peak > 1.0:
                self._array /= peak
        else:
            self._array = np.clip(self._array, -1.0, 1.0)
        return self

    # -- export / playback ----------------------------------------------------


    def to_wav(self, path: str) -> None:
        """Write the soundwave to ``path`` as a mono 16-bit PCM WAV file."""
        write_wav(path, self._array, self._sample_rate, normalize=True)

    def from_wav(self, path: str) -> "Block":
        """Read a mono 16-bit PCM WAV file at ``path`` into a Block

        This method overwrites the content of the block with the file input
        """

        samples, sample_rate = read_wav(path, normalized=True)
        self._sample_rate = sample_rate
        self._array = np.clip(samples, -1.0, 1.0)
        return self


    def play(self) -> None:
        """Play the soundwave using the operating system's audio output.

        Works out of the box on macOS (via ``afplay``); also supports Linux
        (via ``aplay``/``paplay``) and Windows (via the ``winsound`` module).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.to_wav(tmp_path)
            play_wav(tmp_path)
        finally:
            os.remove(tmp_path)

    def __len__(self) -> int:
        return len(self._array)

    def __repr__(self) -> str:
        return (
            f"Block(samples={len(self._array)}, "
            f"sample_rate={self._sample_rate}, "
            f"duration={self.duration:.3f}s)"
        )


def load_wav(path: str) -> Block:
    """Read a mono 16-bit PCM WAV file at ``path`` into a new Block object.

    """
    return Block().from_wav(path)
