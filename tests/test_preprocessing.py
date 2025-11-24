"""Tests for preprocessing module."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from preprocessing import (
    moving_average,
    savitzky_golay_filter,
    normalize,
    preprocess_waveform,
    remove_baseline
)


def test_moving_average_basic():
    """Test basic moving average functionality."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(y, window_size=3)

    assert len(result) == len(y)
    # Middle value should be average of surrounding
    assert abs(result[2] - 3.0) < 0.1


def test_moving_average_single_window():
    """Test moving average with window size 1 (no smoothing)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(y, window_size=1)

    np.testing.assert_array_almost_equal(result, y)


def test_moving_average_invalid_window():
    """Test that invalid window size raises error."""
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        moving_average(y, window_size=0)


def test_moving_average_large_window():
    """Test moving average with window larger than array."""
    y = np.array([1.0, 2.0, 3.0])
    result = moving_average(y, window_size=10)

    # Should handle gracefully
    assert len(result) == len(y)


def test_normalize_basic():
    """Test normalization to [0, 1] range."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = normalize(y)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert len(result) == len(y)


def test_normalize_already_normalized():
    """Test normalization of already normalized data."""
    y = np.array([0.0, 0.5, 1.0])
    result = normalize(y)

    np.testing.assert_array_almost_equal(result, y)


def test_normalize_constant():
    """Test normalization of constant array."""
    y = np.array([5.0, 5.0, 5.0])
    result = normalize(y)

    # Should return zeros when all values are same
    np.testing.assert_array_equal(result, np.zeros_like(y))


def test_normalize_negative_values():
    """Test normalization with negative values."""
    y = np.array([-10.0, 0.0, 10.0])
    result = normalize(y)

    assert result[0] == 0.0
    assert result[1] == 0.5
    assert result[2] == 1.0


def test_preprocess_waveform_savgol():
    """Test full preprocessing pipeline with Savitzky-Golay filter."""
    x = np.linspace(0, 100, 50)
    # Create noisy peak
    y = np.exp(-((x - 50)**2) / 100) + np.random.normal(0, 0.05, len(x))

    x_proc, y_proc = preprocess_waveform(x, y, smooth_method='savgol')

    assert len(x_proc) == len(x)
    assert len(y_proc) == len(y)
    assert y_proc.min() >= 0.0
    assert y_proc.max() <= 1.0


def test_preprocess_waveform_moving_avg():
    """Test preprocessing with moving average."""
    x = np.linspace(0, 100, 50)
    y = np.exp(-((x - 50)**2) / 100)

    x_proc, y_proc = preprocess_waveform(x, y, smooth_method='moving_avg')

    assert len(x_proc) == len(x)
    assert y_proc.min() >= 0.0
    assert y_proc.max() <= 1.0


def test_preprocess_waveform_no_smoothing():
    """Test preprocessing without smoothing."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 20.0, 10.0])

    x_proc, y_proc = preprocess_waveform(x, y, smooth_method='none')

    np.testing.assert_array_equal(x_proc, x)
    # Should still be normalized
    assert y_proc.max() == 1.0
    assert y_proc.min() == 0.0


def test_remove_baseline():
    """Test baseline removal."""
    y = np.array([10.0, 11.0, 12.0, 50.0, 12.0, 11.0, 10.0])
    result = remove_baseline(y, percentile=10)

    # All values should be >= 0
    assert np.all(result >= 0)
    # Peak should still be prominent
    assert np.argmax(result) == 3


def test_remove_baseline_no_noise():
    """Test baseline removal on clean signal."""
    y = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    result = remove_baseline(y, percentile=10)

    # Should preserve peak
    assert result[3] > 0
    assert np.argmax(result) == 3
