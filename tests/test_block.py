import os

import numpy as np
import pytest

from soniquete.block import Block, load_wav, normalize_array
from soniquete.config import _DEFAULT_SAMPLE_RATE

# -- normalize_array ----------------------------------------------------------

def test_normalize_array_scales_peak_to_one():
    result = normalize_array(np.array([2.0, -4.0, 1.0]), normalize=True)
    assert result == pytest.approx([0.5, -1.0, 0.25])


def test_normalize_array_leaves_values_already_within_range_untouched():
    result = normalize_array(np.array([0.5, -0.5]), normalize=True)
    assert result == pytest.approx([0.5, -0.5])


def test_normalize_array_handles_empty_array():
    result = normalize_array(np.array([]), normalize=True)
    assert len(result) == 0


def test_normalize_array_clips_when_not_normalizing():
    result = normalize_array(np.array([2.0, -2.0, 0.5]), normalize=False)
    assert result == pytest.approx([1.0, -1.0, 0.5])


def test_normalize_array_does_not_mutate_input():
    original = np.array([2.0, -4.0, 1.0])
    untouched = original.copy()
    normalize_array(original, normalize=True)
    assert np.array_equal(original, untouched)


def test_normalize_array_rejects_multi_dimensional_input():
    with pytest.raises(ValueError):
        normalize_array(np.zeros((2, 2)), normalize=True)


def test_normalize_array_returns_float64():
    result = normalize_array(np.array([1, 2, 3], dtype=np.int16), normalize=False)
    assert result.dtype == np.float64


# -- Block construction ----------------------------------------------------------


def test_block_defaults_to_empty_array_and_default_sample_rate():
    block = Block()
    assert len(block) == 0
    assert block.sample_rate == _DEFAULT_SAMPLE_RATE


def test_block_from_duration_is_zero_filled_with_expected_length():
    block = Block(duration=0.1, sample_rate=100)
    assert len(block) == 10
    assert np.all(block.array == 0.0)


def test_block_rejects_negative_duration():
    with pytest.raises(ValueError):
        Block(duration=-1.0)


def test_block_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        Block(sample_rate=0)


def test_block_rejects_array_and_duration_together():
    with pytest.raises(ValueError):
        Block(array=np.array([1.0]), duration=1.0)


def test_block_from_array_sets_sample_rate():
    block = Block(np.array([0.1, 0.2]), sample_rate=8000)
    assert block.sample_rate == 8000


def test_block_from_array_normalizes_by_default():
    block = Block(np.array([2.0, -4.0]))
    assert block.array == pytest.approx([0.5, -1.0])


def test_block_from_array_clips_when_normalize_is_false():
    block = Block(np.array([2.0, -2.0]), normalize=False)
    assert block.array == pytest.approx([1.0, -1.0])


def test_block_from_array_does_not_mutate_caller_array():
    original = np.array([2.0, -4.0])
    untouched = original.copy()
    Block(original)
    assert np.array_equal(original, untouched)


# -- properties ----------------------------------------------------------


def test_duration_is_samples_over_sample_rate():
    block = Block(np.zeros(100), sample_rate=200)
    assert block.duration == pytest.approx(0.5)


def test_taper_rise_time_defaults_from_config():
    from soniquete.config import _DEFAULT_WINDOW_RISE_TIME

    block = Block()
    assert block.taper_rise_time == _DEFAULT_WINDOW_RISE_TIME


def test_taper_rise_time_can_be_overridden():
    block = Block(rise_time=0.05)
    assert block.taper_rise_time == 0.05


def test_len_matches_array_length():
    block = Block(np.zeros(7))
    assert len(block) == 7


def test_repr_includes_samples_rate_and_duration():
    block = Block(np.zeros(100), sample_rate=100)
    text = repr(block)
    assert "samples=100" in text
    assert "sample_rate=100" in text
    assert "duration=1.000s" in text


# -- set_volume ----------------------------------------------------------


def test_set_volume_rescales_peak_amplitude():
    block = Block(np.array([0.5, -0.25]), normalize=False)
    block.set_volume(0.5)
    assert float(np.max(np.abs(block.array))) == pytest.approx(0.5)


def test_set_volume_returns_self_for_chaining():
    block = Block(np.array([0.5]))
    assert block.set_volume(0.2) is block


def test_set_volume_rejects_out_of_range_value():
    block = Block(np.array([0.5]))
    with pytest.raises(ValueError):
        block.set_volume(1.5)
    with pytest.raises(ValueError):
        block.set_volume(-0.1)


def test_set_volume_on_silence_is_a_no_op():
    block = Block(np.zeros(4))
    block.set_volume(0.8)
    assert np.all(block.array == 0.0)


def test_set_volume_on_empty_block_is_a_no_op():
    block = Block()
    result = block.set_volume(0.8)
    assert result is block
    assert len(block) == 0


# -- apply_taper ----------------------------------------------------------


def test_apply_taper_fades_edges_to_zero():
    block = Block(np.ones(100), sample_rate=100, normalize=False, rise_time=0.1)
    block.apply_taper()
    assert block.array[0] == pytest.approx(0.0)
    assert block.array[-1] == pytest.approx(0.0)


def test_apply_taper_leaves_middle_untouched():
    block = Block(np.ones(100), sample_rate=100, normalize=False, rise_time=0.1)
    block.apply_taper()
    assert block.array[50] == pytest.approx(1.0)


def test_apply_taper_returns_self_for_chaining():
    block = Block(np.ones(10), sample_rate=10, normalize=False)
    assert block.apply_taper() is block


def test_apply_taper_on_empty_block_is_a_no_op():
    block = Block()
    result = block.apply_taper()
    assert result is block
    assert len(block) == 0


# -- insert ----------------------------------------------------------


def test_insert_mixes_at_start_time():
    base = Block(np.zeros(10), sample_rate=10, normalize=False)
    other = Block(np.array([1.0, 1.0]), sample_rate=10, normalize=False)
    base.insert(other, start_time=0.3)
    assert base.array[3] == pytest.approx(1.0)
    assert base.array[4] == pytest.approx(1.0)
    assert base.array[0] == pytest.approx(0.0)


def test_insert_sums_overlapping_audio():
    base = Block(np.array([0.2, 0.2, 0.2]), sample_rate=10, normalize=False)
    other = Block(np.array([0.3, 0.3]), sample_rate=10, normalize=False)
    base.insert(other, start_time=0.0, normalize=False)
    assert base.array == pytest.approx([0.5, 0.5, 0.2])


def test_insert_extends_block_when_it_runs_past_the_end():
    base = Block(np.array([0.1, 0.1]), sample_rate=10, normalize=False)
    other = Block(np.array([0.2, 0.2]), sample_rate=10, normalize=False)
    base.insert(other, start_time=0.2, normalize=False)
    assert len(base) == 4
    assert base.array == pytest.approx([0.1, 0.1, 0.2, 0.2])


def test_insert_returns_self_for_chaining():
    base = Block(np.zeros(4), sample_rate=10)
    other = Block(np.zeros(2), sample_rate=10)
    assert base.insert(other, 0.0) is base


def test_insert_with_empty_block_is_a_no_op():
    base = Block(np.array([0.1, 0.2]), sample_rate=10, normalize=False)
    empty = Block(sample_rate=10)
    base.insert(empty, start_time=0.0)
    assert base.array == pytest.approx([0.1, 0.2])


def test_insert_rejects_non_block_argument():
    base = Block(np.zeros(4), sample_rate=10)
    with pytest.raises(TypeError):
        base.insert("not a block", 0.0)


def test_insert_rejects_negative_start_time():
    base = Block(np.zeros(4), sample_rate=10)
    other = Block(np.zeros(2), sample_rate=10)
    with pytest.raises(ValueError):
        base.insert(other, -0.1)


def test_insert_rejects_mismatched_sample_rate():
    base = Block(np.zeros(4), sample_rate=10)
    other = Block(np.zeros(2), sample_rate=20)
    with pytest.raises(ValueError):
        base.insert(other, 0.0)


def test_insert_normalizes_when_it_clips():
    base = Block(np.array([0.8, 0.8]), sample_rate=10, normalize=False)
    other = Block(np.array([0.8, 0.8]), sample_rate=10, normalize=False)
    base.insert(other, start_time=0.0, normalize=True)
    assert float(np.max(np.abs(base.array))) == pytest.approx(1.0)


def test_insert_clips_instead_of_normalizing_when_normalize_is_false():
    base = Block(np.array([0.8, 0.8]), sample_rate=10, normalize=False)
    other = Block(np.array([0.8, 0.8]), sample_rate=10, normalize=False)
    base.insert(other, start_time=0.0, normalize=False)
    assert base.array == pytest.approx([1.0, 1.0])


# -- WAV export / import ----------------------------------------------------------


def test_to_wav_and_load_wav_round_trip(tmp_path):
    path = tmp_path / "out.wav"
    original = Block(np.array([0.5, -0.5, 0.25, -0.25]), sample_rate=8000, normalize=False)
    original.to_wav(str(path))

    loaded = load_wav(str(path))
    assert loaded.sample_rate == 8000
    assert len(loaded) == len(original)
    assert loaded.array == pytest.approx(original.array, abs=1e-4)


def test_from_wav_overwrites_existing_block_contents(tmp_path):
    path = tmp_path / "out.wav"
    Block(np.array([0.1, 0.2, 0.3]), sample_rate=4000, normalize=False).to_wav(str(path))

    block = Block(np.ones(50), sample_rate=44100)
    result = block.from_wav(str(path))
    assert result is block
    assert block.sample_rate == 4000
    assert len(block) == 3


# -- listen ----------------------------------------------------------


def test_listen_writes_a_temp_wav_and_plays_it_then_cleans_up(monkeypatch):
    played_paths = []

    def fake_play_wav(path):
        played_paths.append(path)
        assert os.path.exists(path)

    monkeypatch.setattr("soniquete.block.play_wav", fake_play_wav)

    block = Block(np.array([0.1, 0.2]), sample_rate=100, normalize=False)
    block.listen()

    assert len(played_paths) == 1
    assert not os.path.exists(played_paths[0])
