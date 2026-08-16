"""Envelope shapes: time-varying intensity curves applied to a waveform."""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from .config import _DEFAULT_SAMPLE_RATE, _DEFAULT_WINDOW_RISE_TIME


# -- shape evaluation ---------------------------------------------------
#
# Each function computes one envelope shape's intensity curve from the
# number of points to evaluate (N) and the spacing between them in seconds
# (dt) — from which it builds its own t = dt * arange(N) — plus that
# shape's own parameters.


def gaussian_envelope(N: int, dt: float, center: float, sigma: float) -> np.ndarray:
    """Return a 0 to 1 normalized gaussian envelope"""
    t = dt * np.arange(N)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def exponential_envelope(N: int, dt: float, decay_time: float) -> np.ndarray:
    """Return an envelope with linear rise and exponential decay"""
    t = dt * np.arange(N)
    tau = decay_time
    return (t / tau) * np.exp(1 - t / tau)


def sine_modulated_envelope(
    N: int, dt: float, frequency: float, fraction: float
) -> np.ndarray:
    """Return an envelope with a sine modulated intensity"""
    t = dt * np.arange(N)
    half = fraction / 2
    return (1 - half) + half * np.cos(2 * np.pi * frequency * t)


def hamming_envelope(N: int, dt: float, duration: float) -> np.ndarray:
    """Return a hamming envelope"""
    t = dt * np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * t / duration)


def window_envelope(N: int, dt: float, duration: float, rise_time: float) -> np.ndarray:
    """Return a window envelop that implements cosine rise and fall at both ends"""
    t = dt * np.arange(N)
    rise_time = min(rise_time, duration / 2)
    envelope = np.ones_like(t)
    if rise_time <= 0:
        return envelope
    rising = t < rise_time
    falling = t > duration - rise_time
    envelope = np.where(
        rising,
        0.5 * (1 - np.cos(np.pi * np.clip(t, 0, rise_time) / rise_time)),
        envelope,
    )
    envelope = np.where(
        falling,
        0.5 * (1 - np.cos(np.pi * np.clip(duration - t, 0, rise_time) / rise_time)),
        envelope,
    )
    return envelope

# Populated by Envelope.__init_subclass__ below: maps each shape's "name" to
# the Envelope subclass that implements it. parse_envelope() uses this to
# pick the right subclass for a JSON blob's "name" field.
_ENVELOPE_TYPES: dict[str, type["Envelope"]] = {}


class Envelope:
    """Base class for a named intensity-vs-time curve applied to a soundwave.

    An envelope transforms an array by multiplying each sample by a
    modulated intensity value computed from its position in time. In that
    sense, the click-suppressing raised-cosine ramp used by
    :meth:`~soniquete.core.Block.window` is itself a kind of envelope (see
    :class:`WindowEnvelope`).

    This is an abstract base — instantiate one of its subclasses directly
    (:class:`GaussianEnvelope`, :class:`ExponentialEnvelope`,
    :class:`SineModulatedEnvelope`, :class:`HammingEnvelope`,
    :class:`WindowEnvelope`), or, when the shape is only known at runtime
    (e.g. loaded from a JSON string), build one with :func:`parse_envelope`.

    ``duration`` (seconds) sets every shape's overall scale; it's the only
    parameter every envelope has in common. Each subclass documents its own
    extra, shape-specific parameters.
    """

    #: Overridden by each subclass: the shape name used in JSON
    #: serialization and looked up by :func:`parse_envelope`.
    name: str = ""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            _ENVELOPE_TYPES[cls.name] = cls

    def __init__(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = duration

    # -- shape evaluation ---------------------------------------------------

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        """Subclasses compute their shape's intensity curve here."""
        raise NotImplementedError

    def intensity(
        self, n: int, sample_rate: float = _DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        """The envelope's intensity over ``n`` samples at ``sample_rate`` Hz.

        Samples are evaluated at ``t = np.arange(n) / sample_rate`` seconds
        from the pulse start (each subclass's ``_intensity`` builds that
        ``t`` array itself from ``n`` and ``dt = 1 / sample_rate``).
        ``sample_rate`` defaults to
        :data:`~soniquete.config._DEFAULT_SAMPLE_RATE`, the rate used for
        WAV export.
        """
        dt = 1.0 / sample_rate
        return self._intensity(n, dt)

    # -- application ------------------------------------------------------

    def apply(
        self, array: np.ndarray, sample_rate: float = _DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        """Return ``array`` multiplied by this envelope's intensity curve.

        ``sample_rate`` (Hz) maps each array index to the time (seconds)
        the envelope shape is evaluated at; it defaults to
        :data:`~soniquete.config._DEFAULT_SAMPLE_RATE`, the rate used for
        WAV export.
        """
        arr = np.asarray(array, dtype=np.float64)
        return arr * self.intensity(len(arr), sample_rate)

    def __call__(
        self, array: np.ndarray, sample_rate: float = _DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        return self.apply(array, sample_rate)

    # -- serialization ------------------------------------------------------

    def _params(self) -> dict:
        """Shape-specific constructor kwargs beyond ``duration``.

        Overridden by subclasses that take extra parameters; used by
        :meth:`to_dict`/:meth:`to_json` to round-trip them.
        """
        return {}

    def to_dict(self) -> dict:
        """This envelope's configuration (shape and parameters) as a dict."""
        return {"name": self.name, "duration": self.duration, **self._params()}

    def to_json(self) -> str:
        """Serialize this envelope's configuration (shape and parameters) to JSON."""
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(duration={self.duration})"


class GaussianEnvelope(Envelope):
    """A bell curve centered on the pulse.

    ``duration`` spans six standard deviations (``sigma = duration / 6``).
    """

    name = "gaussian"

    def __init__(self, duration: float) -> None:
        super().__init__(duration)
        self.center = duration / 2
        self.sigma = duration / 6

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        return gaussian_envelope(N, dt, self.center, self.sigma)


class ExponentialEnvelope(Envelope):
    """A causal rise-and-decay pulse, ``t * exp(-t/tau)``.

    Normalized so its maximum intensity is exactly 1 (reached at
    ``t = tau``). ``tau``, the characteristic decay time, is given by
    ``decay_time`` (default ``duration / 5``).
    """

    name = "exponential"

    def __init__(self, duration: float, decay_time: float | None = None) -> None:
        super().__init__(duration)
        self.decay_time = decay_time if decay_time is not None else duration / 5
        if self.decay_time <= 0:
            raise ValueError("decay_time must be positive")

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        return exponential_envelope(N, dt, self.decay_time)

    def _params(self) -> dict:
        return {"decay_time": self.decay_time}


class SineModulatedEnvelope(Envelope):
    """A cosine tremolo oscillating between ``1 - fraction`` and ``1``.

    ``frequency`` (Hz) defaults to ``5 / duration``, i.e. five cycles over
    the pulse.
    """

    name = "sine_modulated"

    def __init__(
        self,
        duration: float,
        frequency: float | None = None,
        fraction: float = 0.5,
    ) -> None:
        super().__init__(duration)
        self.frequency = frequency if frequency is not None else 5.0 / duration
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")
        self.fraction = fraction

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        return sine_modulated_envelope(N, dt, self.frequency, self.fraction)

    def _params(self) -> dict:
        return {"frequency": self.frequency, "fraction": self.fraction}


class HammingEnvelope(Envelope):
    """The classic Hamming window: a raised cosine centered on the pulse.

    It peaks at 1 but — unlike :class:`GaussianEnvelope` — never quite
    reaches 0 at the edges (it bottoms out at 0.08), which is what gives
    the Hamming window its particularly low nearest side lobe. Takes no
    parameters beyond ``duration``.
    """

    name = "hamming"

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        return hamming_envelope(N, dt, self.duration)


class WindowEnvelope(Envelope):
    """The click-suppressing raised-cosine fade in/out.

    Used by :meth:`~soniquete.core.Block.window` — 0 -> 1 over
    ``rise_time`` seconds at the start, 1 -> 0 over ``rise_time`` seconds
    at the end, and 1 in between. ``rise_time`` defaults to
    :data:`~soniquete.config._DEFAULT_WINDOW_RISE_TIME`.
    """

    name = "window"

    def __init__(self, duration: float, rise_time: float | None = None) -> None:
        super().__init__(duration)
        self.rise_time = (
            rise_time if rise_time is not None else _DEFAULT_WINDOW_RISE_TIME
        )
        if self.rise_time < 0:
            raise ValueError("rise_time must be non-negative")

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        return window_envelope(N, dt, self.duration, self.rise_time)

    def _params(self) -> dict:
        return {"rise_time": self.rise_time}

    def apply(
        self, array: np.ndarray, sample_rate: float = _DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        """Return ``array`` multiplied by this envelope's intensity curve.

        Evaluated directly in sample-space (matching the exact algorithm
        :meth:`~soniquete.core.Block.window` used before it was factored
        out into this class), so its ramp length is an integer number of
        samples clamped to half the array's length, rather than an
        approximation via :meth:`Envelope.intensity`.
        """
        arr = np.asarray(array, dtype=np.float64)
        return arr * self._window_samples(len(arr), sample_rate)

    def _window_samples(self, n: int, sample_rate: float) -> np.ndarray:
        """The exact sample-domain raised-cosine ramp for this shape."""
        envelope = np.ones(n, dtype=np.float64)
        if n == 0:
            return envelope

        rise_samples = min(int(round(self.rise_time * sample_rate)), n // 2)
        if rise_samples <= 0:
            return envelope

        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(rise_samples) / rise_samples))
        envelope[:rise_samples] = ramp
        envelope[n - rise_samples :] = ramp[::-1]
        return envelope


class CustomEnvelope(Envelope):
    """A user-supplied intensity curve, given as a Python callable.

    ``func`` receives the array of sample times ``t`` (in seconds, starting
    at ``0``) and must return an array of intensities of the same shape —
    the same convention every built-in shape function in this module
    follows (see e.g. :func:`gaussian_envelope`), just without the ``N``/
    ``dt`` split, since a caller building ``func`` by hand can derive ``t``
    itself.

    Unlike every other :class:`Envelope` subclass, this one deliberately
    has no ``name`` and is never registered with :func:`parse_envelope` —
    so a JSON blob can never cause one to be constructed. That's not an
    oversight: ``func`` is an arbitrary Python callable, and
    :func:`parse_envelope` exists specifically to build envelopes from
    *untrusted* data (e.g. a JSON string received over the network); there
    is no safe way to decode a callable back out of JSON without effectively
    running attacker-supplied code. For the same reason, :meth:`to_dict`
    (and thus :meth:`~Envelope.to_json`, which calls it) raises rather than
    silently dropping ``func`` or serializing something that can't be
    rebuilt. Construct a ``CustomEnvelope`` directly in trusted Python code
    and pass it around as an object (e.g. to :class:`~soniquete.wform.Waveform`)
    instead.
    """

    name = ""  # Deliberately unregistered — see class docstring.

    def __init__(
        self, duration: float, func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        super().__init__(duration)
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func

    def _intensity(self, N: int, dt: float) -> np.ndarray:
        t = dt * np.arange(N)
        result = np.asarray(self.func(t), dtype=np.float64)
        if result.shape != t.shape:
            raise ValueError(
                "CustomEnvelope's func must return an array of the same "
                f"shape as its input ({t.shape}), got {result.shape}"
            )
        return result

    def to_dict(self) -> dict:
        """Not supported — see the class docstring."""
        raise TypeError(
            "CustomEnvelope wraps an arbitrary Python function and can't be "
            "serialized to JSON, so it can never be rebuilt by "
            "parse_envelope()/Waveform.from_json() either — construct it "
            "directly in code and pass it in as an object instead."
        )

    def __repr__(self) -> str:
        return f"CustomEnvelope(duration={self.duration}, func={self.func!r})"


def parse_envelope(data: str) -> Envelope:
    """Build the right :class:`Envelope` subclass from a JSON string.

    ``data`` is a JSON object shaped like the output of
    :meth:`Envelope.to_json` — a ``"name"`` field selecting the shape (one
    of the ``Envelope`` subclasses in this module), plus ``"duration"`` and
    any shape-specific parameters, all forwarded as keyword arguments to
    that shape's constructor.
    """
    config = json.loads(data)
    if not isinstance(config, dict):
        raise ValueError(f"Envelope JSON must decode to an object, got {config!r}")

    config = dict(config)
    raw_name = config.pop("name", None)
    name = raw_name.lower() if isinstance(raw_name, str) else raw_name
    try:
        envelope_cls = _ENVELOPE_TYPES[name]
    except KeyError:
        raise ValueError(
            f"Unknown envelope shape: {raw_name!r} "
            f"(expected one of {sorted(_ENVELOPE_TYPES)})"
        ) from None
    return envelope_cls(**config)
