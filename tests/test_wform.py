import numpy as np
import pytest

from soniquete.freq import Frequency
from soniquete.shapes import FadeEnvelope
from soniquete.wform import Waveform


def test_default_waveform_is_silent():
    wf = Waveform()
    assert wf._wform.shape == (44100,)
    assert np.all(wf._wform == 0.0)


def test_sample_count_matches_duration_and_sample_rate():
    wf = Waveform(frequencies=[(440, 1.0)], duration=0.5, sample_rate=1000)
    assert wf._wform.shape == (500,)


def test_single_tone_matches_sine_wave():
    wf = Waveform(frequencies=[(10, 1.0)], intensity=1.0, duration=1.0, sample_rate=100)
    t = np.arange(100) / 100
    expected = np.sin(2 * np.pi * 10 * t)
    assert wf._wform == pytest.approx(expected)


def test_intensity_scales_peak_amplitude():
    # 25 Hz at 100 samples/s lands a sample exactly on the sine's peak
    # (t=0.01s -> sin(2*pi*25*0.01) = sin(pi/2) = 1).
    wf = Waveform(frequencies=[(25, 1.0)], intensity=0.5, duration=1.0, sample_rate=100)
    assert np.max(np.abs(wf._wform)) == pytest.approx(0.5)


def test_multiple_tones_are_amplitude_weighted_and_normalized():
    wf = Waveform(
        frequencies=[(10, 1.0), (10, 3.0)],
        intensity=1.0,
        duration=1.0,
        sample_rate=100,
    )
    t = np.arange(100) / 100
    # Same frequency twice at weights 1 and 3 out of a total of 4 collapses
    # back to a plain sine at full intensity.
    expected = np.sin(2 * np.pi * 10 * t)
    assert wf._wform == pytest.approx(expected)


def test_accepts_frequency_instances_and_note_names():
    wf = Waveform(frequencies=[(Frequency("A4"), 1.0)], duration=1.0, sample_rate=100)
    assert wf.frequencies[0][0].hz == pytest.approx(440.0)


def test_envelopes_are_applied_in_order():
    envelope = FadeEnvelope(duration=1.0, rise_time=0.5)
    wf = Waveform(
        frequencies=[(10, 1.0)],
        duration=1.0,
        sample_rate=100,
        envelopes=[envelope],
    )
    assert wf._wform[0] == pytest.approx(0.0)


def test_zero_duration_gives_empty_waveform():
    wf = Waveform(duration=0.0)
    assert wf._wform.shape == (0,)


def test_array_property_returns_wform():
    wf = Waveform(frequencies=[(440, 1.0)], duration=0.1)
    assert wf.array is wf._wform


# -- validation -----------------------------------------------------------


def test_negative_duration_raises():
    with pytest.raises(ValueError):
        Waveform(duration=-1.0)


@pytest.mark.parametrize("intensity", [-0.1, 1.1])
def test_intensity_out_of_range_raises(intensity):
    with pytest.raises(ValueError):
        Waveform(intensity=intensity)


def test_non_positive_amplitude_raises():
    with pytest.raises(ValueError):
        Waveform(frequencies=[(440, 0.0)])
