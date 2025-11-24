"""
Preprocessing module for radar waveform data.
Handles smoothing, denoising, and normalization.
"""
import numpy as np
from typing import Tuple


def moving_average(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to signal.

    Args:
        y: Input signal array
        window_size: Size of moving average window

    Returns:
        Smoothed signal
    """
    if window_size < 1:
        raise ValueError("Window size must be positive")
    if window_size > len(y):
        window_size = len(y)

    kernel = np.ones(window_size) / window_size
    # Use 'same' mode to maintain array length
    smoothed = np.convolve(y, kernel, mode='same')
    return smoothed


def savitzky_golay_filter(y: np.ndarray, window_size: int = 11, poly_order: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing while preserving peaks.

    Args:
        y: Input signal array
        window_size: Size of filter window (must be odd)
        poly_order: Order of polynomial to fit

    Returns:
        Smoothed signal
    """
    try:
        from scipy.signal import savgol_filter

        # Ensure window size is odd and valid
        if window_size % 2 == 0:
            window_size += 1
        if window_size > len(y):
            window_size = len(y) if len(y) % 2 == 1 else len(y) - 1
        if poly_order >= window_size:
            poly_order = window_size - 1

        return savgol_filter(y, window_size, poly_order)
    except ImportError:
        # Fallback to moving average if scipy not available
        print("Warning: scipy not available, using moving average instead")
        return moving_average(y, window_size)


def normalize(y: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [0, 1] range.

    Args:
        y: Input signal array

    Returns:
        Normalized signal
    """
    y_min = np.min(y)
    y_max = np.max(y)

    if y_max == y_min:
        return np.zeros_like(y)

    return (y - y_min) / (y_max - y_min)


def preprocess_waveform(x: np.ndarray, y: np.ndarray,
                         smooth_method: str = 'savgol',
                         window_size: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single waveform.

    Args:
        x: Distance array (mm)
        y: Intensity array
        smooth_method: 'savgol', 'moving_avg', or 'none'
        window_size: Window size for smoothing

    Returns:
        Tuple of (x, processed_y)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Apply smoothing
    if smooth_method == 'savgol':
        y_smooth = savitzky_golay_filter(y, window_size)
    elif smooth_method == 'moving_avg':
        y_smooth = moving_average(y, window_size)
    else:
        y_smooth = y

    # Normalize to [0, 1]
    y_normalized = normalize(y_smooth)

    return x, y_normalized


def remove_baseline(y: np.ndarray, percentile: float = 10) -> np.ndarray:
    """
    Remove baseline noise from signal.

    Args:
        y: Input signal
        percentile: Percentile to use as baseline estimate

    Returns:
        Signal with baseline removed
    """
    baseline = np.percentile(y, percentile)
    return np.maximum(y - baseline, 0)
