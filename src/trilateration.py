"""
Trilateration module.
Pure math module for calculating object position from 3 distances.
No I/O operations - only mathematical computations.
"""
import numpy as np
from typing import Tuple, Dict, Optional


# Sensor positions for circular configuration
# Sensors positioned at 120° intervals on a circle
CIRCLE_RADIUS = 600  # mm

# Default sensor positions (on circle, 120° apart, centered at origin)
# Using angles: 90°, 210°, 330° (or equivalently: top, bottom-left, bottom-right)
DEFAULT_SENSORS = {
    'A': np.array([CIRCLE_RADIUS * np.cos(np.radians(90)),
                   CIRCLE_RADIUS * np.sin(np.radians(90))]),   # Top
    'B': np.array([CIRCLE_RADIUS * np.cos(np.radians(210)),
                   CIRCLE_RADIUS * np.sin(np.radians(210))]),  # Bottom-left
    'C': np.array([CIRCLE_RADIUS * np.cos(np.radians(330)),
                   CIRCLE_RADIUS * np.sin(np.radians(330))])   # Bottom-right
}


def get_sensor_positions(radius: float = 600, angles: list = None) -> Dict[str, np.ndarray]:
    """
    Calculate sensor positions for circular configuration.
    Sensors are placed at equal angular intervals on a circle.

    Args:
        radius: Distance from center to each sensor (mm)
        angles: List of angles in degrees for sensor positions (default: [90, 210, 330])

    Returns:
        Dictionary of sensor positions
    """
    if angles is None:
        angles = [90, 210, 330]  # 120° apart

    return {
        'A': np.array([radius * np.cos(np.radians(angles[0])),
                       radius * np.sin(np.radians(angles[0]))]),
        'B': np.array([radius * np.cos(np.radians(angles[1])),
                       radius * np.sin(np.radians(angles[1]))]),
        'C': np.array([radius * np.cos(np.radians(angles[2])),
                       radius * np.sin(np.radians(angles[2]))])
    }


def trilaterate_closed_form(d_a: float, d_b: float, d_c: float,
                             sensors: Optional[Dict[str, np.ndarray]] = None) -> Tuple[float, float]:
    """
    Solve trilateration using closed-form algebraic solution.
    Works for arbitrary sensor positions.

    Args:
        d_a: Distance from sensor A
        d_b: Distance from sensor B
        d_c: Distance from sensor C
        sensors: Dictionary of sensor positions (uses default if None)

    Returns:
        Tuple of (x, y) coordinates
    """
    if sensors is None:
        sensors = DEFAULT_SENSORS

    # Extract positions
    x1, y1 = sensors['A']
    x2, y2 = sensors['B']
    x3, y3 = sensors['C']

    r1, r2, r3 = d_a, d_b, d_c

    # Solve system of equations:
    # (x - x1)^2 + (y - y1)^2 = r1^2
    # (x - x2)^2 + (y - y2)^2 = r2^2
    # (x - x3)^2 + (y - y3)^2 = r3^2

    # Subtract first equation from second and third to linearize:
    # 2(x2-x1)x + 2(y2-y1)y = r1^2 - r2^2 - x1^2 + x2^2 - y1^2 + y2^2
    # 2(x3-x1)x + 2(y3-y1)y = r1^2 - r3^2 - x1^2 + x3^2 - y1^2 + y3^2

    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2

    D = 2 * (x3 - x1)
    E = 2 * (y3 - y1)
    F = r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2

    # Solve 2x2 linear system: Ax + By = C, Dx + Ey = F
    det = A * E - B * D

    if abs(det) < 1e-10:
        # Sensors are collinear, can't solve uniquely
        raise ValueError("Sensors are collinear, cannot perform trilateration")

    x = (C * E - F * B) / det
    y = (A * F - D * C) / det

    return float(x), float(y)


def trilaterate_least_squares(d_a: float, d_b: float, d_c: float,
                               sensors: Optional[Dict[str, np.ndarray]] = None) -> Tuple[float, float]:
    """
    Solve trilateration using least-squares optimization.
    More robust to measurement noise.

    Args:
        d_a: Distance from sensor A
        d_b: Distance from sensor B
        d_c: Distance from sensor C
        sensors: Dictionary of sensor positions

    Returns:
        Tuple of (x, y) coordinates
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("Warning: scipy not available, using closed-form solution")
        return trilaterate_closed_form(d_a, d_b, d_c, sensors)

    if sensors is None:
        sensors = DEFAULT_SENSORS

    # Use closed-form solution as initial guess (better than centroid)
    try:
        initial_guess = np.array(trilaterate_closed_form(d_a, d_b, d_c, sensors))
    except:
        # Fallback to centroid if closed-form fails
        initial_guess = np.mean([sensors['A'], sensors['B'], sensors['C']], axis=0)

    def residuals(point):
        """Calculate residuals for each sensor."""
        x, y = point
        r_a = np.sqrt((x - sensors['A'][0])**2 + (y - sensors['A'][1])**2) - d_a
        r_b = np.sqrt((x - sensors['B'][0])**2 + (y - sensors['B'][1])**2) - d_b
        r_c = np.sqrt((x - sensors['C'][0])**2 + (y - sensors['C'][1])**2) - d_c
        return [r_a, r_b, r_c]

    result = least_squares(residuals, initial_guess)

    return float(result.x[0]), float(result.x[1])


def trilaterate(distances: Dict[str, float],
                sensors: Optional[Dict[str, np.ndarray]] = None,
                method: str = 'least_squares') -> Tuple[float, float]:
    """
    Main trilateration function.

    Args:
        distances: Dictionary with sensor IDs as keys and distances as values
        sensors: Dictionary of sensor positions
        method: 'closed_form' or 'least_squares'

    Returns:
        Tuple of (x, y) estimated object position
    """
    if sensors is None:
        sensors = DEFAULT_SENSORS

    # Get distances in order
    d_a = distances.get('A', distances.get('1', distances.get(1)))
    d_b = distances.get('B', distances.get('2', distances.get(2)))
    d_c = distances.get('C', distances.get('3', distances.get(3)))

    if None in (d_a, d_b, d_c):
        raise ValueError(f"Missing distance measurements. Got: {distances}")

    if method == 'closed_form':
        return trilaterate_closed_form(d_a, d_b, d_c, sensors)
    elif method == 'least_squares':
        return trilaterate_least_squares(d_a, d_b, d_c, sensors)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_error(estimated: Tuple[float, float],
                    actual: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance error between estimated and actual position.

    Args:
        estimated: Estimated (x, y) position
        actual: Actual (x, y) position

    Returns:
        Error in mm
    """
    return float(np.sqrt((estimated[0] - actual[0])**2 +
                         (estimated[1] - actual[1])**2))


def get_triangle_center(sensors: Optional[Dict[str, np.ndarray]] = None) -> Tuple[float, float]:
    """
    Get the centroid of the sensor triangle.

    Args:
        sensors: Dictionary of sensor positions

    Returns:
        (x, y) coordinates of triangle center
    """
    if sensors is None:
        sensors = DEFAULT_SENSORS

    center = np.mean([sensors['A'], sensors['B'], sensors['C']], axis=0)
    return float(center[0]), float(center[1])
