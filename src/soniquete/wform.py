"""Waveform and WhiteNoise: Blocks synthesized from a higher-level spec."""

from __future__ import annotations

import numpy as np

from .wav import _DEFAULT_SAMPLE_RATE
from .freq import Frequency


class Tone:
    """A single pure tone: a frequency, an amplitude, and a phase.

    ``frequency`` is either a :class:`~soniquete.freq.Frequency` or a plain
    number, taken to be a frequency in Hz and converted into one.
    """

    def __init__(
        self,
        frequency: "Frequency | str | float | int",
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")

        self._freq = (
            frequency if isinstance(frequency, Frequency) else Frequency(frequency)
        )
        self._amp = amplitude
        self._phase = phase

    @property
    def hz(self) -> float:
        """The frequency of the tone, in Hz."""
        return self._freq.hz

    @property
    def amp(self) -> float:
        """The amplitude of the tone."""
        return self._amp

    @property
    def phase(self) -> float:
        """The phase of the tone, in radians."""
        return self._phase


class Waveform:
    """A soundwave synthesized from a list of :class:`Tone` objects.

    """

    def __init__(
        self,
        frequencies: list | None = None,
        duration: float = 1.0,
        intensity: float = 1.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        envelopes: list | None = None,
    ) -> None:
        if not 0 <= intensity <= 1:
            raise ValueError("intensity must be between 0 and 1")
        if duration < 0:
            raise ValueError("duration must be non-negative")

        self.frequencies = []
        for tone in frequencies or []:
            if not isinstance(tone, Tone):
                raise TypeError(f"Expected a Tone, got {tone!r}")
            self.frequencies.append(tone)

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

    @property
    def t(self) -> np.array:
        """Time steps of the waveform"""
        return self._t

    def _build_wform(self) -> np.array:

        n_samples = int(round(self.duration * self.sample_rate))
        t = np.arange(n_samples) / self.sample_rate
        self._t = t

        signal = np.zeros(n_samples, dtype=np.float64)
        for tone in self.frequencies:
            signal += tone.amp * np.sin(2 * np.pi * tone.hz * t + tone.phase)

        for envelope in self.envelopes:
            signal = envelope(signal, self.sample_rate)

        max_signal = np.max(np.abs(signal))
        print(max_signal)
        if max_signal > 0:
            signal = signal / max_signal * self.intensity

        return signal

