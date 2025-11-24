"""
Generate synthetic sample data for testing the radar position estimation system.
Creates realistic waveform data with known ground truth positions.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from trilateration import get_sensor_positions, CIRCLE_RADIUS


def generate_waveform(true_distance: float, noise_level: float = 0.1,
                       n_points: int = 512, max_range: float = 2000) -> tuple:
    """
    Generate a synthetic radar waveform with peak at true_distance.

    Args:
        true_distance: True distance to object (mm)
        noise_level: Noise standard deviation
        n_points: Number of points in waveform
        max_range: Maximum range of sensor (mm)

    Returns:
        Tuple of (x_array, y_array)
    """
    # Create distance array
    x = np.linspace(0, max_range, n_points)

    # Create Gaussian peak at true distance
    sigma = 30  # Peak width in mm
    amplitude = 0.8 + np.random.uniform(-0.1, 0.1)

    # Main peak
    y = amplitude * np.exp(-((x - true_distance) ** 2) / (2 * sigma ** 2))

    # Add some baseline noise
    y += np.random.normal(0, noise_level, n_points)

    # Add minor secondary reflections (more realistic)
    if np.random.random() > 0.5:
        secondary_dist = true_distance + np.random.uniform(50, 150)
        if secondary_dist < max_range:
            y += 0.2 * amplitude * np.exp(-((x - secondary_dist) ** 2) / (2 * sigma * 1.5 ** 2))

    # Ensure non-negative
    y = np.maximum(y, 0)

    return x.tolist(), y.tolist()


def generate_measurement_set(object_position: tuple, sensor_positions: dict,
                               noise_level: float = 0.05) -> dict:
    """
    Generate a complete measurement set for one object position.

    Args:
        object_position: (x, y) position of object
        sensor_positions: Dictionary of sensor positions
        noise_level: Waveform noise level

    Returns:
        Dictionary with sensor IDs as keys
    """
    measurements = {}
    obj_x, obj_y = object_position

    for sensor_id, sensor_pos in sensor_positions.items():
        # Calculate true distance
        true_dist = np.sqrt((obj_x - sensor_pos[0])**2 +
                            (obj_y - sensor_pos[1])**2)

        # Generate waveform
        x, y = generate_waveform(true_dist, noise_level)

        measurements[sensor_id] = {
            'a': sensor_id,
            'x': x,
            'y': y,
            'd': float(true_dist)
        }

    return measurements


def generate_dataset(n_measurements: int = 10, output_dir: str = '../data',
                     noise_level: float = 0.05):
    """
    Generate complete dataset with multiple measurements.

    Args:
        n_measurements: Number of measurement sets to generate
        output_dir: Directory to save JSON files
        noise_level: Noise level for waveforms
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sensors = get_sensor_positions()

    # Generate object positions (within circle, near center)
    # Circle is centered at origin (0, 0)
    center_x = 0.0
    center_y = 0.0

    print(f"Circle center: ({center_x:.2f}, {center_y:.2f})")
    print(f"Circle radius: {CIRCLE_RADIUS:.2f} mm")
    print(f"Generating {n_measurements} measurement sets...")

    # Store ground truth
    ground_truth = []

    for i in range(n_measurements):
        # Random position near center (within ~400mm radius, staying inside circle)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 400)  # Up to 400mm from center
        obj_x = center_x + radius * np.cos(angle)
        obj_y = center_y + radius * np.sin(angle)

        ground_truth.append({
            'measurement_id': i,
            'true_position': [float(obj_x), float(obj_y)],
            'distance_from_center': float(radius)
        })

        # Generate measurements for this position
        measurements = generate_measurement_set((obj_x, obj_y), sensors, noise_level)

        # Save each sensor's data as separate file
        for sensor_id, data in measurements.items():
            filename = f"measurement_{i:03d}_sensor_{sensor_id}.json"
            filepath = output_path / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"  Generated measurement {i}: object at ({obj_x:.2f}, {obj_y:.2f})")

    # Save ground truth
    gt_path = output_path / "ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nDataset saved to {output_path}")
    print(f"Ground truth saved to {gt_path}")


if __name__ == '__main__':
    generate_dataset(n_measurements=10, noise_level=0.05)
