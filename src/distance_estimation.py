"""
Distance estimation module.
Extracts distance from radar waveform using various methods.
"""
import numpy as np
from typing import Tuple, Optional


def peak_detection(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate distance using peak detection (maximum intensity).

    Args:
        x: Distance array (mm)
        y: Intensity array (should be preprocessed)

    Returns:
        Estimated distance in mm
    """
    idx = np.argmax(y)
    return float(x[idx])


def weighted_centroid(x: np.ndarray, y: np.ndarray, threshold: float = 0.1) -> float:
    """
    Estimate distance using weighted centroid method.
    More robust to noise than simple peak detection.

    Args:
        x: Distance array (mm)
        y: Intensity array (should be normalized to [0,1])
        threshold: Minimum intensity to include in calculation

    Returns:
        Estimated distance in mm
    """
    # Apply threshold to reduce noise influence
    y_thresh = np.where(y >= threshold, y, 0)

    total_weight = np.sum(y_thresh)
    if total_weight == 0:
        # Fallback to peak if no values above threshold
        return peak_detection(x, y)

    weighted_sum = np.sum(x * y_thresh)
    return float(weighted_sum / total_weight)


def gaussian_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Gaussian curve to waveform and extract peak location.

    Args:
        x: Distance array (mm)
        y: Intensity array

    Returns:
        Tuple of (center, amplitude, sigma)
    """
    try:
        from scipy.optimize import curve_fit

        def gaussian(x, amplitude, center, sigma):
            return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

        # Initial guess based on data
        idx_max = np.argmax(y)
        p0 = [y[idx_max], x[idx_max], (x[-1] - x[0]) / 10]

        # Fit curve
        popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=5000)

        return float(popt[1]), float(popt[0]), float(popt[2])

    except Exception as e:
        print(f"Warning: Gaussian fit failed ({e}), using weighted centroid")
        center = weighted_centroid(x, y)
        return center, float(np.max(y)), 10.0


def estimate_distance(x: np.ndarray, y: np.ndarray,
                      method: str = 'weighted_centroid') -> float:
    """
    Main function to estimate distance from waveform.

    Args:
        x: Distance array (mm)
        y: Intensity array (should be preprocessed)
        method: 'peak', 'weighted_centroid', or 'gaussian'

    Returns:
        Estimated distance in mm
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if method == 'peak':
        return peak_detection(x, y)
    elif method == 'weighted_centroid':
        return weighted_centroid(x, y)
    elif method == 'gaussian':
        center, _, _ = gaussian_fit(x, y)
        return center
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_confidence(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate confidence in distance measurement based on signal quality.

    Args:
        x: Distance array
        y: Intensity array (normalized)

    Returns:
        Confidence score [0, 1]
    """
    # Factors: peak height, signal-to-noise ratio, peak sharpness
    peak_height = np.max(y)

    # SNR estimate: peak vs mean of lower 50%
    sorted_y = np.sort(y)
    noise_level = np.mean(sorted_y[:len(sorted_y)//2])
    snr = peak_height / (noise_level + 1e-6)

    # Combine factors
    confidence = min(1.0, (peak_height * 0.5 + min(snr / 10, 0.5)))

    return float(confidence)
