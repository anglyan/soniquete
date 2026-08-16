"""Implement a simple class for note to frequency conversion

Considers an equal temperament to convert scientific pitch notation to frequency,
with an A4 reference value of 440.0 Hz.

"""

from __future__ import annotations

import re

# Concert pitch reference: A4 = 440 Hz = MIDI note 69.
_A4_HZ = 440.0
_A4_MIDI = 69

# Semitone offset of each natural note from C, within one octave.
_NOTE_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Scientific pitch notation, e.g. "C4", "F#3", "Bb5". Accidentals use the
# common, unambiguous ASCII notation: '#' for sharp, 'b' for flat.
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note_to_frequency(note: str, offset: float) -> float:
    if not -1 <= offset <= 1:
        raise ValueError("offset must be between -1 and 1")

    match = _NOTE_RE.match(note.strip())
    if not match:
        raise ValueError(f"Invalid note name: {note!r}")
    letter, accidental, octave_str = match.groups()

    semitone = _NOTE_SEMITONES[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1

    midi = (int(octave_str) + 1) * 12 + semitone
    base_hz = midi_to_hz(midi)

    if offset == 0:
        return base_hz

    neighbor_midi = midi + (1 if offset > 0 else -1)
    neighbor_hz = midi_to_hz(neighbor_midi)
    return base_hz + abs(offset) * (neighbor_hz - base_hz)


def midi_to_hz(midi: int) -> float:
    return _A4_HZ * 2 ** ((midi - _A4_MIDI) / 12)


class Frequency:
    """A single tone's frequency, optionally derived from a musical note.

    ``value`` is either:

    - a musical note name in scientific pitch notation, e.g. ``"C4"``
      (middle C), ``"F#3"`` (F sharp), ``"Bb5"`` (B flat); or
    - a plain number, taken to be a frequency in Hz.

    ``offset`` only applies when ``value`` is a note name. It nudges the
    pitch towards the neighboring semitone as a fraction of the distance
    to that neighbor: a positive offset moves towards the note a semitone
    *above*, a negative offset towards the note a semitone *below* (e.g.
    ``Frequency("C4", 0.5)`` is exactly halfway between C4 and C#4).
    ``offset`` must be between -1 and 1.
    """

    def __init__(self, value: "str | float | int | Frequency", offset: float = 0.0) -> None:
        self._note: str | None = None
        self._offset: float = 0.0

        if isinstance(value, Frequency):
            self._hz = value.hz
            self._note = value.note
            self._offset = value.offset
        elif isinstance(value, str):
            self._note = value
            self._offset = float(offset)
            self._hz = note_to_frequency(value, self._offset)
        elif isinstance(value, (int, float)):
            if offset:
                raise ValueError("offset only applies when a note name is supplied")
            self._hz = float(value)
        else:
            raise TypeError(f"Unsupported frequency value: {value!r}")

    # -- properties -----------------------------------------------------------

    @property
    def hz(self) -> float:
        """The frequency in Hz."""
        return self._hz

    @property
    def note(self) -> str | None:
        """The original note name, if this frequency was built from one."""
        return self._note

    @property
    def offset(self) -> float:
        """The offset (fraction of a semitone) applied to ``note``, if any."""
        return self._offset

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        """A JSON-serializable representation of this frequency."""
        data: dict = {"hz": self._hz}
        if self._note is not None:
            data["note"] = self._note
            data["offset"] = self._offset
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Frequency":
        """Build a :class:`Frequency` from a dict produced by :meth:`to_dict`."""
        if "note" in data:
            return cls(data["note"], data.get("offset", 0.0))
        return cls(data["hz"])

    def __float__(self) -> float:
        return self._hz

    def __repr__(self) -> str:
        if self._note is not None:
            return f"Frequency({self._note!r}, offset={self._offset}) -> {self._hz:.3f} Hz"
        return f"Frequency({self._hz:.3f} Hz)"
