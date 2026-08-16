import pytest

from soniquete.freq import Frequency


def test_plain_number_is_used_as_is():
    f = Frequency(261.63)
    assert f.hz == pytest.approx(261.63)
    assert f.note is None


def test_a4_is_440hz():
    f = Frequency("A4")
    assert f.hz == pytest.approx(440.0)


def test_c4_is_middle_c():
    f = Frequency("C4")
    assert f.hz == pytest.approx(261.6256, rel=1e-3)


def test_sharp_and_flat_notation():
    sharp = Frequency("C#4")
    flat = Frequency("Db4")
    assert sharp.hz == pytest.approx(flat.hz)
    assert sharp.hz > Frequency("C4").hz
    assert sharp.hz < Frequency("D4").hz


def test_lowercase_note_is_accepted():
    assert Frequency("a4").hz == pytest.approx(Frequency("A4").hz)


def test_octave_changes_frequency_by_factor_of_two():
    assert Frequency("A5").hz == pytest.approx(Frequency("A4").hz * 2)


def test_positive_offset_moves_towards_higher_neighbor():
    base = Frequency("C4").hz
    upper = Frequency("C#4").hz
    half = Frequency("C4", 0.5)
    assert half.hz == pytest.approx(base + 0.5 * (upper - base))


def test_negative_offset_moves_towards_lower_neighbor():
    base = Frequency("C4").hz
    lower = Frequency("B3").hz
    half = Frequency("C4", -0.5)
    assert half.hz == pytest.approx(base + 0.5 * (lower - base))
    assert half.hz < base


def test_full_offset_reaches_neighbor_exactly():
    assert Frequency("C4", 1.0).hz == pytest.approx(Frequency("C#4").hz)
    assert Frequency("C4", -1.0).hz == pytest.approx(Frequency("B3").hz)


def test_offset_out_of_range_raises():
    with pytest.raises(ValueError):
        Frequency("C4", 1.5)


def test_offset_without_note_raises():
    with pytest.raises(ValueError):
        Frequency(440.0, 0.5)


def test_invalid_note_raises():
    with pytest.raises(ValueError):
        Frequency("H4")


def test_wrapping_existing_frequency():
    original = Frequency("C4", 0.25)
    wrapped = Frequency(original)
    assert wrapped.hz == pytest.approx(original.hz)
    assert wrapped.note == "C4"
    assert wrapped.offset == 0.25


def test_to_dict_for_plain_number():
    f = Frequency(300.0)
    assert f.to_dict() == {"hz": 300.0}


def test_to_dict_for_note():
    f = Frequency("A4")
    d = f.to_dict()
    assert d["hz"] == pytest.approx(440.0)
    assert d["note"] == "A4"
    assert d["offset"] == 0.0


def test_invalid_type_raises():
    with pytest.raises(TypeError):
        Frequency(object())


def test_from_dict_round_trips_plain_number():
    f = Frequency.from_dict({"hz": 300.0})
    assert f.hz == pytest.approx(300.0)
    assert f.note is None


def test_from_dict_round_trips_note():
    original = Frequency("C#3", offset=0.25)
    f = Frequency.from_dict(original.to_dict())
    assert f.hz == pytest.approx(original.hz)
    assert f.note == "C#3"
    assert f.offset == pytest.approx(0.25)
