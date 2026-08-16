"""Waveform and WhiteNoise: Blocks synthesized from a higher-level spec."""

from __future__ import annotations

import numpy as np

from .wav import _DEFAULT_SAMPLE_RATE
from .freq import Frequency

class Waveform:
    """A soundwave synthesized from a list of (frequency, amplitude) tones.

    """

    def __init__(
        self,
        frequencies: list | None = None,
        intensity: float = 1.0,
        duration: float = 1.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        envelopes: list | None = None,
    ) -> None:
        if not 0 <= intensity <= 1:
            raise ValueError("intensity must be between 0 and 1")
        if duration < 0:
            raise ValueError("duration must be non-negative")

        self.frequencies = []
        for f, amplitude in frequencies or []:
            if amplitude <= 0:
                raise ValueError("amplitude must be positive")
            self.frequencies.append(
                (f if isinstance(f, Frequency) else Frequency(f), amplitude)
            )

        self.intensity = intensity
        self.duration = duration
        self.sample_rate = sample_rate

        if envelopes is None:
            self.envelopes = []
        else:
            self.envelopes = envelopes

        self._wform = self._build_wform()

    @property
    def array(self) -> np.ndarray:
        """The synthesized waveform"""
        return self._wform

    def _build_wform(self) -> np.array:

        n_samples = int(round(self.duration * self.sample_rate))
        t = np.arange(n_samples) / self.sample_rate

        signal = np.zeros(n_samples, dtype=np.float64)
        for f, amplitude in self.frequencies:
            signal += amplitude * np.sin(2 * np.pi * f.hz * t)

        total_amplitude = sum(amplitude for _, amplitude in self.frequencies)
        if total_amplitude > 0:
            signal = signal / total_amplitude * self.intensity

        for envelope in self.envelopes:
            signal = envelope(signal, self.sample_rate)
        return signal

