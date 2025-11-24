"""Tests for trilateration module."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from trilateration import (
    get_sensor_positions,
    trilaterate_closed_form,
    trilaterate_least_squares,
    trilaterate,
    calculate_error,
    get_triangle_center,
    DEFAULT_SENSORS,
    CIRCLE_RADIUS
)


def test_get_sensor_positions_default():
    """Test sensor position calculation with default radius."""
    sensors = get_sensor_positions(600)

    assert 'A' in sensors
    assert 'B' in sensors
    assert 'C' in sensors

    # All sensors should be 600mm from origin (center)
    for sensor_id, pos in sensors.items():
        dist_from_center = np.linalg.norm(pos)
        assert abs(dist_from_center - 600.0) < 0.01, f"Sensor {sensor_id} not at correct radius"


def test_sensor_triangle_is_equilateral():
    """Test that sensors form equilateral triangle."""
    sensors = get_sensor_positions(600)

    # Calculate distances between sensors
    ab = np.linalg.norm(sensors['B'] - sensors['A'])
    bc = np.linalg.norm(sensors['C'] - sensors['B'])
    ca = np.linalg.norm(sensors['A'] - sensors['C'])

    # All sides should be equal
    assert abs(ab - bc) < 0.01
    assert abs(bc - ca) < 0.01
    assert abs(ca - ab) < 0.01


def test_sensors_on_circle():
    """Test that sensors are positioned on a circle centered at origin."""
    sensors = get_sensor_positions(600)
    center = np.array([0.0, 0.0])

    for sensor_id, pos in sensors.items():
        dist = np.linalg.norm(pos - center)
        assert abs(dist - 600.0) < 0.01


def test_trilaterate_at_center():
    """Test trilateration when object is at triangle center."""
    sensors = DEFAULT_SENSORS
    center = get_triangle_center(sensors)

    # Calculate distances from center to each sensor
    d_a = np.linalg.norm(np.array(center) - sensors['A'])
    d_b = np.linalg.norm(np.array(center) - sensors['B'])
    d_c = np.linalg.norm(np.array(center) - sensors['C'])

    # Trilaterate
    x, y = trilaterate_least_squares(d_a, d_b, d_c, sensors)

    # Should return center position
    assert abs(x - center[0]) < 1.0
    assert abs(y - center[1]) < 1.0


def test_trilaterate_at_sensor_a():
    """Test trilateration when object is near sensor A."""
    sensors = DEFAULT_SENSORS

    # Object near sensor A
    obj_pos = np.array([50.0, 50.0])

    d_a = np.linalg.norm(obj_pos - sensors['A'])
    d_b = np.linalg.norm(obj_pos - sensors['B'])
    d_c = np.linalg.norm(obj_pos - sensors['C'])

    x, y = trilaterate_least_squares(d_a, d_b, d_c, sensors)

    error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
    assert error < 1.0


def test_trilaterate_various_positions():
    """Test trilateration at various positions in triangle."""
    sensors = DEFAULT_SENSORS

    # Test multiple positions
    test_positions = [
        [200.0, 200.0],
        [500.0, 300.0],
        [300.0, 600.0],
        [700.0, 400.0],
        [400.0, 100.0]
    ]

    for pos in test_positions:
        obj_pos = np.array(pos)

        d_a = np.linalg.norm(obj_pos - sensors['A'])
        d_b = np.linalg.norm(obj_pos - sensors['B'])
        d_c = np.linalg.norm(obj_pos - sensors['C'])

        x, y = trilaterate_least_squares(d_a, d_b, d_c, sensors)

        error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
        assert error < 1.0, f"Position {pos} had error {error}"


def test_trilaterate_with_noise():
    """Test trilateration with noisy distance measurements."""
    sensors = DEFAULT_SENSORS
    center = get_triangle_center(sensors)
    obj_pos = np.array(center)

    # True distances
    d_a_true = np.linalg.norm(obj_pos - sensors['A'])
    d_b_true = np.linalg.norm(obj_pos - sensors['B'])
    d_c_true = np.linalg.norm(obj_pos - sensors['C'])

    # Add noise (5mm standard deviation)
    np.random.seed(42)
    noise = np.random.normal(0, 5, 3)

    d_a = d_a_true + noise[0]
    d_b = d_b_true + noise[1]
    d_c = d_c_true + noise[2]

    x, y = trilaterate_least_squares(d_a, d_b, d_c, sensors)

    # Error should be reasonable even with noise
    error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
    assert error < 20.0  # Allow more error due to noise


def test_trilaterate_closed_form():
    """Test closed-form trilateration solution."""
    sensors = DEFAULT_SENSORS
    obj_pos = np.array([300.0, 300.0])

    d_a = np.linalg.norm(obj_pos - sensors['A'])
    d_b = np.linalg.norm(obj_pos - sensors['B'])
    d_c = np.linalg.norm(obj_pos - sensors['C'])

    x, y = trilaterate_closed_form(d_a, d_b, d_c, sensors)

    error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
    assert error < 5.0


def test_trilaterate_dict_interface():
    """Test trilateration with dictionary input."""
    sensors = DEFAULT_SENSORS
    obj_pos = np.array([400.0, 350.0])

    distances = {
        'A': float(np.linalg.norm(obj_pos - sensors['A'])),
        'B': float(np.linalg.norm(obj_pos - sensors['B'])),
        'C': float(np.linalg.norm(obj_pos - sensors['C']))
    }

    x, y = trilaterate(distances, sensors, method='least_squares')

    error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
    assert error < 1.0


def test_trilaterate_numeric_keys():
    """Test trilateration with numeric sensor keys."""
    sensors = DEFAULT_SENSORS
    obj_pos = np.array([400.0, 350.0])

    distances = {
        1: float(np.linalg.norm(obj_pos - sensors['A'])),
        2: float(np.linalg.norm(obj_pos - sensors['B'])),
        3: float(np.linalg.norm(obj_pos - sensors['C']))
    }

    x, y = trilaterate(distances, sensors, method='least_squares')

    error = np.sqrt((x - obj_pos[0])**2 + (y - obj_pos[1])**2)
    assert error < 1.0


def test_trilaterate_missing_distance():
    """Test trilateration with missing distance raises error."""
    distances = {
        'A': 100.0,
        'B': 150.0,
        # Missing 'C'
    }

    with pytest.raises(ValueError):
        trilaterate(distances)


def test_calculate_error_exact():
    """Test error calculation with exact match."""
    estimated = (100.0, 200.0)
    actual = (100.0, 200.0)

    error = calculate_error(estimated, actual)
    assert error == 0.0


def test_calculate_error_horizontal():
    """Test error calculation with horizontal displacement."""
    estimated = (110.0, 200.0)
    actual = (100.0, 200.0)

    error = calculate_error(estimated, actual)
    assert error == 10.0


def test_calculate_error_vertical():
    """Test error calculation with vertical displacement."""
    estimated = (100.0, 215.0)
    actual = (100.0, 200.0)

    error = calculate_error(estimated, actual)
    assert error == 15.0


def test_calculate_error_diagonal():
    """Test error calculation with diagonal displacement."""
    estimated = (103.0, 204.0)
    actual = (100.0, 200.0)

    error = calculate_error(estimated, actual)
    assert abs(error - 5.0) < 0.01  # 3-4-5 triangle


def test_get_triangle_center():
    """Test triangle center calculation."""
    sensors = DEFAULT_SENSORS
    center = get_triangle_center(sensors)

    # Center should be equidistant from all sensors
    d_a = np.linalg.norm(np.array(center) - sensors['A'])
    d_b = np.linalg.norm(np.array(center) - sensors['B'])
    d_c = np.linalg.norm(np.array(center) - sensors['C'])

    # All distances should be equal (centroid)
    assert abs(d_a - d_b) < 0.01
    assert abs(d_b - d_c) < 0.01


def test_triangle_center_custom_sensors():
    """Test center calculation with custom sensor positions."""
    sensors = {
        'A': np.array([0.0, 0.0]),
        'B': np.array([100.0, 0.0]),
        'C': np.array([50.0, 100.0])
    }

    cx, cy = get_triangle_center(sensors)

    # Should be average of coordinates
    assert abs(cx - 50.0) < 0.01
    assert abs(cy - 100.0/3) < 0.1
