"""
Data loader module for radar sensor waveform data.
Handles reading JSON files and organizing data by sensor ID.
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def load_sensor_file(filepath: str) -> Dict[str, Any]:
    """
    Load a single sensor data file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary containing sensor data with keys: 'a' (sensor_id), 'x', 'y', 'd'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Validate structure
    required_keys = {'a', 'x', 'y', 'd'}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise ValueError(f"Missing required keys: {missing}")

    return data


def load_all_data(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all sensor data files from directory.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary grouped by sensor ID (a), each containing list of measurements
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all JSON files
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")

    # Group by sensor ID
    sensor_data = {}

    for file_path in json_files:
        try:
            data = load_sensor_file(str(file_path))
            sensor_id = data['a']

            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = []

            sensor_data[sensor_id].append({
                'x': data['x'],
                'y': data['y'],
                'd': data['d'],
                'file': file_path.name
            })
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return sensor_data


def load_measurement_set(data_dir: str, measurement_id: int = 0) -> Dict[str, Dict]:
    """
    Load a single measurement set (one reading from each of 3 sensors).

    Args:
        data_dir: Path to data directory
        measurement_id: Index of measurement to retrieve

    Returns:
        Dictionary with sensor IDs as keys, each containing x, y, d arrays
    """
    all_data = load_all_data(data_dir)

    if len(all_data) != 3:
        raise ValueError(f"Expected 3 sensors, found {len(all_data)}")

    result = {}
    for sensor_id, measurements in all_data.items():
        if measurement_id >= len(measurements):
            raise IndexError(f"Measurement {measurement_id} not available for sensor {sensor_id}")
        result[sensor_id] = measurements[measurement_id]

    return result
