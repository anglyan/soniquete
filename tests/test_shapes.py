import json

import numpy as np
import pytest

from soniquete.shapes import (
    CustomEnvelope,
    Envelope,
    ExponentialEnvelope,
    FadeEnvelope,
    GaussianEnvelope,
    SineModulatedEnvelope,
    _ENVELOPE_TYPES,
    parse_envelope,
)


def _linspace_intensity(env: Envelope, n: int, duration: float) -> np.ndarray:
    """Evaluate ``env`` at ``n`` points evenly spaced over ``[0, duration]``.

    ``intensity`` evaluates at ``t = arange(n) / sample_rate``, so choosing
    ``sample_rate = (n - 1) / duration`` reproduces what
    ``np.linspace(0, duration, n)`` used to give directly.
    """
    sample_rate = (n - 1) / duration
    return env.intensity(n, sample_rate)


def test_parse_envelope_rejects_unknown_shape():
    with pytest.raises(ValueError):
        parse_envelope(json.dumps({"name": "triangle", "duration": 1.0}))


def test_non_positive_duration_raises():
    with pytest.raises(ValueError):
        GaussianEnvelope(0.0)


# -- gaussian -----------------------------------------------------------


def test_gaussian_peaks_at_center_with_intensity_one():
    env = GaussianEnvelope(duration=0.6)
    assert env.intensity(4, sample_rate=10)[-1] == pytest.approx(1.0)  # t=0.3


def test_gaussian_duration_spans_six_sigma():
    duration = 0.6
    env = GaussianEnvelope(duration=duration)
    assert env.sigma == pytest.approx(duration / 6)
    # +/- 3 sigma (the pulse edges) should sit at the standard 3-sigma value.
    edge = env.intensity(7, sample_rate=10)[[0, -1]]  # t=0.0, 0.6
    expected = np.exp(-0.5 * 3**2)
    assert edge == pytest.approx([expected, expected])


def test_gaussian_symmetric_around_center():
    env = GaussianEnvelope(duration=1.0)
    values = env.intensity(9, sample_rate=10)  # t=0.0 .. 0.8
    left, right = values[2], values[8]  # t=0.2, 0.8
    assert left == pytest.approx(right)


# -- exponential --------------------------------------------------------


def test_exponential_starts_at_zero():
    env = ExponentialEnvelope(duration=1.0, decay_time=0.2)
    assert env.intensity(1, sample_rate=10)[0] == pytest.approx(0.0)  # t=0.0


def test_exponential_peaks_at_decay_time_with_intensity_one():
    env = ExponentialEnvelope(duration=1.0, decay_time=0.2)
    assert env.intensity(3, sample_rate=10)[-1] == pytest.approx(1.0)  # t=0.2


def test_exponential_default_decay_time_is_duration_over_five():
    env = ExponentialEnvelope(duration=1.0)
    assert env.decay_time == pytest.approx(0.2)


def test_exponential_decays_after_peak():
    env = ExponentialEnvelope(duration=1.0, decay_time=0.2)
    values = env.intensity(9, sample_rate=10)  # t=0.0 .. 0.8
    peak, later = values[2], values[8]  # t=0.2, 0.8
    assert later < peak


def test_exponential_rejects_non_positive_decay_time():
    with pytest.raises(ValueError):
        ExponentialEnvelope(duration=1.0, decay_time=0.0)


# -- sine_modulated -------------------------------------------------------


def test_sine_modulated_starts_at_max():
    env = SineModulatedEnvelope(duration=1.0, frequency=5.0, fraction=0.4)
    assert env.intensity(1, sample_rate=10)[0] == pytest.approx(1.0)  # t=0.0


def test_sine_modulated_oscillates_between_expected_bounds():
    env = SineModulatedEnvelope(duration=1.0, frequency=5.0, fraction=0.4)
    values = _linspace_intensity(env, 10000, 1.0)
    assert values.max() == pytest.approx(1.0, abs=1e-3)
    assert values.min() == pytest.approx(0.6, abs=1e-3)


def test_sine_modulated_rejects_fraction_out_of_range():
    with pytest.raises(ValueError):
        SineModulatedEnvelope(duration=1.0, fraction=1.5)


def test_sine_modulated_default_frequency_gives_five_cycles():
    env = SineModulatedEnvelope(duration=2.0)
    assert env.frequency == pytest.approx(2.5)  # 5 cycles over 2 seconds


# -- __call__ ---------------------------------------------------------------


def test_call_uses_default_sample_rate_case():
    from soniquete import DSR

    env = SineModulatedEnvelope(duration=1.0, frequency=5.0, fraction=0.4)
    arr = np.ones(100)
    assert np.allclose(env(arr), env(arr, DSR))


def test_call_multiplies_array_by_intensity():
    env = SineModulatedEnvelope(duration=1.0, frequency=5.0, fraction=0.4)
    arr = np.ones(100)
    sample_rate = 100
    result = env(arr, sample_rate)
    assert np.allclose(result, env.intensity(len(arr), sample_rate))


def test_call_does_not_mutate_input():
    env = GaussianEnvelope(duration=1.0)
    arr = np.ones(10)
    original = arr.copy()
    env(arr, 10)
    assert np.array_equal(arr, original)


# -- fade ---------------------------------------------------------------


def test_fade_default_rise_time_matches_config():
    from soniquete.shapes import _DEFAULT_WINDOW_RISE_TIME

    env = FadeEnvelope(duration=1.0)
    assert env.rise_time == _DEFAULT_WINDOW_RISE_TIME


def test_fade_rejects_negative_rise_time():
    with pytest.raises(ValueError):
        FadeEnvelope(duration=1.0, rise_time=-0.1)


def test_fade_ramps_edges_to_zero_and_one_in_middle():
    env = FadeEnvelope(duration=1.0, rise_time=0.1)
    arr = np.ones(100)
    result = env(arr, sample_rate=100)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(0.0)
    assert result[50] == pytest.approx(1.0)


def test_fade_rise_time_clamped_to_half_the_array():
    # rise_time far exceeds the array's length: the ramp should still meet
    # cleanly in the middle rather than overshoot past it.
    env = FadeEnvelope(duration=1.0, rise_time=10.0)
    arr = np.ones(10)
    result = env(arr, sample_rate=10)
    assert len(result) == 10
    assert np.all(result >= 0)


def test_fade_matches_original_block_algorithm():
    # Bit-for-bit equivalence with the raised-cosine ramp Block.window()
    # used to compute inline before it was factored out into Envelope.
    n, sample_rate, rise_time = 1000, 44100, 0.01
    rise_samples = min(int(round(rise_time * sample_rate)), n // 2)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(rise_samples) / rise_samples))
    expected = np.ones(n, dtype=np.float64)
    expected[:rise_samples] = ramp
    expected[n - rise_samples:] = ramp[::-1]

    env = FadeEnvelope(duration=n / sample_rate, rise_time=rise_time)
    result = env(np.ones(n), sample_rate)
    assert np.array_equal(result, expected)


# -- custom -----------------------------------------------------------------


def test_custom_evaluates_user_function_of_t():
    env = CustomEnvelope(duration=1.0, func=lambda t: 2.0 * t)
    values = env.intensity(3, sample_rate=10)  # t=0.0, 0.1, 0.2
    assert values == pytest.approx([0.0, 0.2, 0.4])


def test_custom_rejects_non_callable_func():
    with pytest.raises(TypeError):
        CustomEnvelope(duration=1.0, func="not callable")


def test_custom_rejects_func_returning_wrong_shape():
    env = CustomEnvelope(duration=1.0, func=lambda t: np.ones(len(t) + 1))
    with pytest.raises(ValueError):
        env.intensity(5, sample_rate=10)


def test_custom_call_multiplies_array_by_func_output():
    env = CustomEnvelope(duration=1.0, func=lambda t: np.full_like(t, 0.5))
    arr = np.ones(10)
    assert np.allclose(env(arr, sample_rate=10), 0.5)


def test_custom_has_no_name_and_is_not_registered():
    # Deliberate: see CustomEnvelope's class docstring — it must never be
    # reachable from parse_envelope(), since func is arbitrary Python code.
    assert CustomEnvelope.name == ""
    assert CustomEnvelope not in _ENVELOPE_TYPES.values()
    assert "" not in _ENVELOPE_TYPES
    assert "custom" not in _ENVELOPE_TYPES


def test_custom_to_dict_raises():
    env = CustomEnvelope(duration=1.0, func=lambda t: t)
    with pytest.raises(TypeError):
        env.to_dict()


def test_custom_to_json_raises():
    env = CustomEnvelope(duration=1.0, func=lambda t: t)
    with pytest.raises(TypeError):
        env.to_json()


def test_parse_envelope_cannot_build_a_custom_envelope():
    with pytest.raises(ValueError):
        parse_envelope(json.dumps({"name": "custom", "duration": 1.0}))


# -- serialization / parse_envelope ----------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        GaussianEnvelope(duration=0.6),
        ExponentialEnvelope(duration=1.0, decay_time=0.3),
        SineModulatedEnvelope(duration=1.0, frequency=3.0, fraction=0.25),
        FadeEnvelope(duration=1.0, rise_time=0.02),
    ],
)
def test_parse_envelope_round_trips_to_json(env):
    rebuilt = parse_envelope(env.to_json())
    assert type(rebuilt) is type(env)
    assert rebuilt.to_dict() == env.to_dict()
    arr = np.ones(50)
    assert np.array_equal(rebuilt(arr, 50), env(arr, 50))


def test_parse_envelope_is_case_insensitive():
    rebuilt = parse_envelope(json.dumps({"name": "GAUSSIAN", "duration": 1.0}))
    assert isinstance(rebuilt, GaussianEnvelope)


def test_to_dict_includes_shape_specific_params():
    env = ExponentialEnvelope(duration=1.0, decay_time=0.3)
    assert env.to_dict() == {
        "name": "exponential",
        "duration": 1.0,
        "decay_time": 0.3,
    }
