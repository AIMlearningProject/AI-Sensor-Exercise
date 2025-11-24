"""Tests for distance estimation module."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from distance_estimation import (
    peak_detection,
    weighted_centroid,
    gaussian_fit,
    estimate_distance,
    estimate_confidence
)


def test_peak_detection_single_peak():
    """Test peak detection with single clear peak."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.3, 0.9, 0.4, 0.1])

    result = peak_detection(x, y)
    assert result == 200.0


def test_peak_detection_edge_peak():
    """Test peak detection when peak is at edge."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.9, 0.3, 0.2, 0.1, 0.1])

    result = peak_detection(x, y)
    assert result == 100.0


def test_peak_detection_multiple_equal_peaks():
    """Test peak detection with multiple equal peaks."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.9, 0.2, 0.9, 0.1])

    # Should return first peak
    result = peak_detection(x, y)
    assert result == 150.0


def test_weighted_centroid_single_peak():
    """Test weighted centroid with single peak."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.3, 0.9, 0.4, 0.1])

    result = weighted_centroid(x, y, threshold=0.1)

    # Should be close to 200, but weighted by neighbors
    assert 180.0 < result < 220.0


def test_weighted_centroid_symmetric_peak():
    """Test weighted centroid with symmetric peak."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])

    result = weighted_centroid(x, y, threshold=0.1)

    # Symmetric peak should give exact center
    assert abs(result - 200.0) < 1.0


def test_weighted_centroid_high_threshold():
    """Test weighted centroid with high threshold."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.3, 0.9, 0.4, 0.1])

    result = weighted_centroid(x, y, threshold=0.8)

    # Only peak should count
    assert result == 200.0


def test_weighted_centroid_all_below_threshold():
    """Test weighted centroid when all values below threshold."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.01, 0.02, 0.05, 0.02, 0.01])

    # Should fall back to peak detection
    result = weighted_centroid(x, y, threshold=0.1)
    assert result == 200.0


def test_gaussian_fit_perfect_gaussian():
    """Test Gaussian fit on perfect Gaussian data."""
    x = np.linspace(0, 100, 100)
    center_true = 50.0
    amplitude_true = 1.0
    sigma_true = 10.0

    y = amplitude_true * np.exp(-((x - center_true)**2) / (2 * sigma_true**2))

    center, amplitude, sigma = gaussian_fit(x, y)

    assert abs(center - center_true) < 1.0
    assert abs(amplitude - amplitude_true) < 0.1
    assert abs(sigma - sigma_true) < 2.0


def test_gaussian_fit_noisy_gaussian():
    """Test Gaussian fit on noisy data."""
    np.random.seed(42)
    x = np.linspace(0, 100, 100)
    center_true = 60.0

    y = np.exp(-((x - center_true)**2) / 200) + np.random.normal(0, 0.05, len(x))
    y = np.maximum(y, 0)  # Ensure no negative values

    center, _, _ = gaussian_fit(x, y)

    # Should be within 5mm of true center
    assert abs(center - center_true) < 5.0


def test_estimate_distance_peak_method():
    """Test estimate_distance with peak method."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.3, 0.9, 0.4, 0.1])

    result = estimate_distance(x, y, method='peak')
    assert result == 200.0


def test_estimate_distance_weighted_centroid_method():
    """Test estimate_distance with weighted centroid method."""
    x = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([0.1, 0.3, 0.9, 0.4, 0.1])

    result = estimate_distance(x, y, method='weighted_centroid')
    assert 180.0 < result < 220.0


def test_estimate_distance_gaussian_method():
    """Test estimate_distance with Gaussian method."""
    x = np.linspace(0, 100, 100)
    y = np.exp(-((x - 50)**2) / 200)

    result = estimate_distance(x, y, method='gaussian')
    assert 45.0 < result < 55.0


def test_estimate_distance_invalid_method():
    """Test estimate_distance with invalid method raises error."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.1, 0.5, 0.1])

    with pytest.raises(ValueError):
        estimate_distance(x, y, method='invalid')


def test_estimate_confidence_high_snr():
    """Test confidence estimation with high SNR."""
    y = np.array([0.01, 0.02, 0.01, 1.0, 0.01, 0.02, 0.01])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    confidence = estimate_confidence(x, y)

    # High peak, low noise should give high confidence
    assert confidence > 0.5


def test_estimate_confidence_low_snr():
    """Test confidence estimation with low SNR."""
    y = np.array([0.3, 0.35, 0.32, 0.4, 0.33, 0.35, 0.31])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    confidence = estimate_confidence(x, y)

    # Low peak height, high noise should give lower confidence
    assert 0.0 <= confidence <= 1.0


def test_estimate_confidence_range():
    """Test that confidence is always in [0, 1] range."""
    for _ in range(10):
        y = np.random.rand(50)
        x = np.linspace(0, 100, 50)

        confidence = estimate_confidence(x, y)

        assert 0.0 <= confidence <= 1.0
