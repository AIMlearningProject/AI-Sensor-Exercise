"""
Main orchestration module for 3-Radar Position Estimation.
Combines all modules into a complete pipeline.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from loader import load_all_data, load_measurement_set
from preprocessing import preprocess_waveform
from distance_estimation import estimate_distance, estimate_confidence
from trilateration import trilaterate, calculate_error, get_triangle_center, get_sensor_positions


def process_single_measurement(measurement: Dict[str, Dict],
                                 method: str = 'weighted_centroid',
                                 smooth_method: str = 'savgol') -> Dict:
    """
    Process a single measurement set (one reading per sensor).

    Args:
        measurement: Dictionary with sensor data
        method: Distance estimation method
        smooth_method: Waveform preprocessing method

    Returns:
        Dictionary with results
    """
    distances = {}
    confidences = {}
    reference_distances = {}

    # Map sensor keys to standard A, B, C
    sensor_mapping = {}
    for i, key in enumerate(sorted(measurement.keys())):
        sensor_mapping[key] = ['A', 'B', 'C'][i]

    for sensor_key, data in measurement.items():
        sensor_id = sensor_mapping[sensor_key]

        # Preprocess waveform
        x_proc, y_proc = preprocess_waveform(
            np.array(data['x']),
            np.array(data['y']),
            smooth_method=smooth_method
        )

        # Estimate distance
        dist = estimate_distance(x_proc, y_proc, method=method)
        distances[sensor_id] = dist

        # Estimate confidence
        conf = estimate_confidence(x_proc, y_proc)
        confidences[sensor_id] = conf

        # Store reference distance
        reference_distances[sensor_id] = data['d']

    # Perform trilateration
    estimated_pos = trilaterate(distances, method='least_squares')

    # Calculate center (reference point)
    center = get_triangle_center()

    # Calculate error if reference available
    error = calculate_error(estimated_pos, center)

    return {
        'estimated_position': estimated_pos,
        'distances': distances,
        'reference_distances': reference_distances,
        'confidences': confidences,
        'error_from_center': error,
        'center': center
    }


def run_pipeline(data_dir: str,
                  method: str = 'weighted_centroid',
                  smooth_method: str = 'savgol',
                  verbose: bool = True) -> List[Dict]:
    """
    Run full processing pipeline on all measurements.

    Args:
        data_dir: Path to data directory
        method: Distance estimation method
        smooth_method: Preprocessing method
        verbose: Print progress

    Returns:
        List of results for each measurement
    """
    if verbose:
        print(f"Loading data from {data_dir}...")

    all_data = load_all_data(data_dir)

    if verbose:
        print(f"Found {len(all_data)} sensors")
        for sensor_id, measurements in all_data.items():
            print(f"  Sensor {sensor_id}: {len(measurements)} measurements")

    # Load ground truth if available
    ground_truth = None
    gt_path = Path(data_dir) / "ground_truth.json"
    if gt_path.exists():
        try:
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
            if verbose:
                print(f"Ground truth loaded: {len(ground_truth)} positions")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load ground truth: {e}")

    # Determine number of measurements
    n_measurements = min(len(m) for m in all_data.values())

    if verbose:
        print(f"\nProcessing {n_measurements} measurement sets...")

    results = []
    errors = []
    estimation_errors = []

    for i in range(n_measurements):
        # Build measurement dict
        measurement = {}
        for sensor_id, measurements in all_data.items():
            measurement[sensor_id] = measurements[i]

        # Process
        result = process_single_measurement(measurement, method, smooth_method)
        result['measurement_id'] = i

        # Calculate actual estimation error if ground truth available
        if ground_truth and i < len(ground_truth):
            true_pos = ground_truth[i]['true_position']
            est_error = calculate_error(result['estimated_position'], tuple(true_pos))
            result['estimation_error'] = est_error
            result['true_position'] = true_pos
            estimation_errors.append(est_error)
        else:
            result['estimation_error'] = None
            result['true_position'] = None

        results.append(result)
        errors.append(result['error_from_center'])

        if verbose:
            est_pos = result['estimated_position']
            if result['estimation_error'] is not None:
                print(f"  Measurement {i}: Est=({est_pos[0]:7.2f}, {est_pos[1]:7.2f}) mm, "
                      f"True=({true_pos[0]:7.2f}, {true_pos[1]:7.2f}) mm, "
                      f"Error={result['estimation_error']:.2f} mm")
            else:
                print(f"  Measurement {i}: Position = ({est_pos[0]:.2f}, "
                      f"{est_pos[1]:.2f}) mm, "
                      f"Distance from center = {result['error_from_center']:.2f} mm")

    # Summary statistics
    if verbose:
        print(f"\n=== Summary ===")

        if estimation_errors:
            print(f"Estimation Accuracy (vs Ground Truth):")
            print(f"  Mean error: {np.mean(estimation_errors):.2f} mm")
            print(f"  Max error: {np.max(estimation_errors):.2f} mm")
            print(f"  Min error: {np.min(estimation_errors):.2f} mm")
            print(f"  Std error: {np.std(estimation_errors):.2f} mm")

            if np.mean(estimation_errors) < 100:
                print(f"  [PASS] Target achieved (<100 mm)")
            else:
                print(f"  [FAIL] Warning: Target not met (>100 mm)")
        else:
            print(f"Distance from Center:")
            print(f"  Mean: {np.mean(errors):.2f} mm")
            print(f"  Max: {np.max(errors):.2f} mm")
            print(f"  Min: {np.min(errors):.2f} mm")

    return results


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='3-Radar Position Estimation')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--method', type=str, default='weighted_centroid',
                        choices=['peak', 'weighted_centroid', 'gaussian'],
                        help='Distance estimation method')
    parser.add_argument('--smooth', type=str, default='savgol',
                        choices=['savgol', 'moving_avg', 'none'],
                        help='Smoothing method')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Run pipeline
    results = run_pipeline(
        args.data_dir,
        method=args.method,
        smooth_method=args.smooth,
        verbose=not args.quiet
    )

    # Save results if output specified
    if args.output:
        save_results(results, args.output)

    return results


if __name__ == '__main__':
    main()
