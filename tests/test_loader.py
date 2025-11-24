"""Tests for data loader module."""
import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from loader import load_sensor_file, load_all_data


def test_load_sensor_file_valid():
    """Test loading a valid sensor file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {
            'a': 'A',
            'x': [1.0, 2.0, 3.0],
            'y': [0.1, 0.5, 0.2],
            'd': 150.0
        }
        json.dump(data, f)
        f.flush()

        result = load_sensor_file(f.name)

        assert result['a'] == 'A'
        assert result['x'] == [1.0, 2.0, 3.0]
        assert result['y'] == [0.1, 0.5, 0.2]
        assert result['d'] == 150.0

    Path(f.name).unlink()


def test_load_sensor_file_missing_keys():
    """Test that missing keys raise an error."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {
            'a': 'A',
            'x': [1.0, 2.0, 3.0],
            # Missing 'y' and 'd'
        }
        json.dump(data, f)
        f.flush()

        with pytest.raises(ValueError) as exc_info:
            load_sensor_file(f.name)

        assert 'Missing required keys' in str(exc_info.value)

    Path(f.name).unlink()


def test_load_all_data():
    """Test loading multiple sensor files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files for 3 sensors
        for sensor_id in ['A', 'B', 'C']:
            data = {
                'a': sensor_id,
                'x': [1.0, 2.0, 3.0],
                'y': [0.1, 0.5, 0.2],
                'd': 150.0
            }
            filepath = Path(tmpdir) / f'sensor_{sensor_id}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f)

        result = load_all_data(tmpdir)

        assert len(result) == 3
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert len(result['A']) == 1


def test_load_all_data_empty_dir():
    """Test that empty directory raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError) as exc_info:
            load_all_data(tmpdir)

        assert 'No JSON files found' in str(exc_info.value)


def test_load_all_data_nonexistent_dir():
    """Test that nonexistent directory raises error."""
    with pytest.raises(FileNotFoundError):
        load_all_data('/nonexistent/path')
