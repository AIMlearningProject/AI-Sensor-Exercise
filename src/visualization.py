"""
Visualization module for radar position estimation.
Provides plotting functions for waveforms, sensor positions, and estimated locations.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def plot_waveform(x: np.ndarray, y: np.ndarray,
                  title: str = "Sensor Waveform",
                  detected_distance: Optional[float] = None,
                  save_path: Optional[str] = None):
    """
    Plot a single sensor waveform.

    Args:
        x: Distance array (mm)
        y: Intensity array
        title: Plot title
        detected_distance: If provided, mark this distance on plot
        save_path: If provided, save plot to this path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, 'b-', linewidth=1.5, label='Intensity')

    if detected_distance is not None:
        ax.axvline(x=detected_distance, color='r', linestyle='--',
                   linewidth=2, label=f'Detected: {detected_distance:.1f}mm')

        # Mark peak point
        idx = np.argmin(np.abs(x - detected_distance))
        ax.plot(detected_distance, y[idx], 'ro', markersize=10)

    ax.set_xlabel('Distance (mm)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.tight_layout()
    return fig, ax


def plot_all_waveforms(waveforms: Dict[str, Dict],
                       detected_distances: Optional[Dict[str, float]] = None,
                       save_path: Optional[str] = None):
    """
    Plot waveforms from all sensors in subplots.

    Args:
        waveforms: Dictionary with sensor IDs as keys, containing 'x' and 'y' arrays
        detected_distances: Dictionary of detected distances per sensor
        save_path: If provided, save plot to this path
    """
    n_sensors = len(waveforms)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 4*n_sensors))

    if n_sensors == 1:
        axes = [axes]

    for idx, (sensor_id, data) in enumerate(sorted(waveforms.items())):
        ax = axes[idx]
        x = np.array(data['x'])
        y = np.array(data['y'])

        ax.plot(x, y, 'b-', linewidth=1.5)

        if detected_distances and sensor_id in detected_distances:
            dist = detected_distances[sensor_id]
            ax.axvline(x=dist, color='r', linestyle='--', linewidth=2,
                      label=f'Detected: {dist:.1f}mm')

            # Mark reference distance if available
            if 'd' in data:
                ref_dist = data['d']
                ax.axvline(x=ref_dist, color='g', linestyle=':', linewidth=2,
                          label=f'Reference: {ref_dist:.1f}mm')

            ax.legend(loc='upper right')

        ax.set_xlabel('Distance (mm)', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.set_title(f'Sensor {sensor_id}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sensor Waveforms', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_sensor_configuration(sensors: Dict[str, np.ndarray],
                               estimated_pos: Optional[Tuple[float, float]] = None,
                               actual_pos: Optional[Tuple[float, float]] = None,
                               circle_radius: float = 600,
                               distances: Optional[Dict[str, float]] = None,
                               save_path: Optional[str] = None):
    """
    Plot sensor positions and estimated object location.

    Args:
        sensors: Dictionary of sensor positions
        estimated_pos: Estimated object position (x, y)
        actual_pos: Actual object position if known
        circle_radius: Radius of the sensor circle
        distances: Measured distances from each sensor
        save_path: If provided, save plot to this path
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the circle boundary
    circle = Circle((0, 0), circle_radius, fill=False,
                    linestyle='--', color='gray', linewidth=2, alpha=0.5)
    ax.add_patch(circle)

    # Plot sensors
    colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
    for sensor_id, pos in sensors.items():
        color = colors.get(sensor_id, 'black')
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=15,
                label=f'Sensor {sensor_id}')
        ax.text(pos[0], pos[1] + 40, f'{sensor_id}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Draw distance circles if distances provided
        if distances and sensor_id in distances:
            dist_circle = Circle((pos[0], pos[1]), distances[sensor_id],
                                 fill=False, linestyle=':', color=color,
                                 linewidth=1, alpha=0.3)
            ax.add_patch(dist_circle)

    # Plot center
    ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Center')

    # Plot estimated position
    if estimated_pos:
        ax.plot(estimated_pos[0], estimated_pos[1], 'x',
                color='purple', markersize=20, markeredgewidth=3,
                label=f'Estimated: ({estimated_pos[0]:.1f}, {estimated_pos[1]:.1f})')

    # Plot actual position
    if actual_pos:
        ax.plot(actual_pos[0], actual_pos[1], 's',
                color='orange', markersize=12,
                label=f'Actual: ({actual_pos[0]:.1f}, {actual_pos[1]:.1f})')

        # Draw error line
        if estimated_pos:
            error = np.sqrt((estimated_pos[0] - actual_pos[0])**2 +
                           (estimated_pos[1] - actual_pos[1])**2)
            ax.plot([estimated_pos[0], actual_pos[0]],
                   [estimated_pos[1], actual_pos[1]],
                   'k--', linewidth=1, alpha=0.5)
            mid_x = (estimated_pos[0] + actual_pos[0]) / 2
            mid_y = (estimated_pos[1] + actual_pos[1]) / 2
            ax.text(mid_x, mid_y, f'Error: {error:.1f}mm',
                   ha='center', va='bottom', fontsize=10)

    # Set axis properties
    ax.set_xlim(-circle_radius * 1.3, circle_radius * 1.3)
    ax.set_ylim(-circle_radius * 1.3, circle_radius * 1.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title('Sensor Configuration and Object Position', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.tight_layout()
    return fig, ax


def plot_error_distribution(errors: List[float],
                             target_error: float = 100,
                             save_path: Optional[str] = None):
    """
    Plot distribution of estimation errors.

    Args:
        errors: List of error values in mm
        target_error: Target error threshold
        save_path: If provided, save plot to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=target_error, color='r', linestyle='--', linewidth=2,
                label=f'Target: {target_error}mm')
    ax1.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(errors):.1f}mm')
    ax1.set_xlabel('Error (mm)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time series / measurement index
    ax2.plot(range(len(errors)), errors, 'b-o', markersize=6)
    ax2.axhline(y=target_error, color='r', linestyle='--', linewidth=2,
                label=f'Target: {target_error}mm')
    ax2.axhline(y=np.mean(errors), color='g', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(errors):.1f}mm')
    ax2.set_xlabel('Measurement Index', fontsize=12)
    ax2.set_ylabel('Error (mm)', fontsize=12)
    ax2.set_title('Error per Measurement', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Position Estimation Error Analysis', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_comparison(x_orig: np.ndarray, y_orig: np.ndarray,
                    x_proc: np.ndarray, y_proc: np.ndarray,
                    title: str = "Waveform Processing Comparison",
                    save_path: Optional[str] = None):
    """
    Compare original and processed waveforms.

    Args:
        x_orig: Original distance array
        y_orig: Original intensity array
        x_proc: Processed distance array
        y_proc: Processed intensity array
        title: Plot title
        save_path: If provided, save plot to this path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Original
    ax1.plot(x_orig, y_orig, 'b-', linewidth=1, label='Original')
    ax1.set_xlabel('Distance (mm)', fontsize=10)
    ax1.set_ylabel('Intensity', fontsize=10)
    ax1.set_title('Original Waveform', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Processed
    ax2.plot(x_proc, y_proc, 'r-', linewidth=1.5, label='Processed')
    ax2.set_xlabel('Distance (mm)', fontsize=10)
    ax2.set_ylabel('Normalized Intensity', fontsize=10)
    ax2.set_title('Processed Waveform (Smoothed & Normalized)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)


def create_summary_plot(result: Dict, sensors: Dict[str, np.ndarray],
                        waveforms: Dict[str, Dict],
                        save_path: Optional[str] = None):
    """
    Create a comprehensive summary plot with all diagnostic information.

    Args:
        result: Result dictionary from process_single_measurement
        sensors: Sensor positions
        waveforms: Raw waveform data
        save_path: If provided, save plot to this path
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Sensor configuration (large plot)
    ax_config = fig.add_subplot(gs[0:2, 0:2])

    # Draw circle
    circle = Circle((0, 0), 600, fill=False, linestyle='--',
                    color='gray', linewidth=2, alpha=0.5)
    ax_config.add_patch(circle)

    # Plot sensors
    colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
    for sensor_id, pos in sensors.items():
        color = colors.get(sensor_id, 'black')
        ax_config.plot(pos[0], pos[1], 'o', color=color, markersize=12)
        ax_config.text(pos[0], pos[1] + 35, f'{sensor_id}',
                      ha='center', fontsize=11, fontweight='bold')

        # Distance circles
        if sensor_id in result['distances']:
            dist_circle = Circle((pos[0], pos[1]), result['distances'][sensor_id],
                                fill=False, linestyle=':', color=color,
                                linewidth=1, alpha=0.3)
            ax_config.add_patch(dist_circle)

    # Plot estimated position
    est_x, est_y = result['estimated_position']
    ax_config.plot(est_x, est_y, 'x', color='purple', markersize=20,
                   markeredgewidth=3, label=f'Estimated: ({est_x:.1f}, {est_y:.1f})')

    # Plot center
    ax_config.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Center')

    ax_config.set_xlim(-800, 800)
    ax_config.set_ylim(-800, 800)
    ax_config.set_aspect('equal')
    ax_config.grid(True, alpha=0.3)
    ax_config.set_xlabel('X (mm)', fontsize=11)
    ax_config.set_ylabel('Y (mm)', fontsize=11)
    ax_config.set_title('Object Position Estimation', fontsize=13)
    ax_config.legend(loc='upper right')

    # Waveform plots (right side)
    for idx, sensor_id in enumerate(['A', 'B', 'C']):
        ax = fig.add_subplot(gs[idx, 2])

        if sensor_id in waveforms:
            data = waveforms[sensor_id]
            x = np.array(data['x'])
            y = np.array(data['y'])

            ax.plot(x, y, 'b-', linewidth=1)

            if sensor_id in result['distances']:
                dist = result['distances'][sensor_id]
                ax.axvline(x=dist, color='r', linestyle='--', linewidth=1.5)

        ax.set_xlabel('Distance (mm)', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.set_title(f'Sensor {sensor_id}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Statistics text box
    ax_stats = fig.add_subplot(gs[2, 0:2])
    ax_stats.axis('off')

    stats_text = f"""
    === Estimation Results ===

    Estimated Position: ({est_x:.2f}, {est_y:.2f}) mm
    Error from Center: {result['error_from_center']:.2f} mm

    Distances:
      Sensor A: {result['distances'].get('A', 'N/A'):.2f} mm (Ref: {result['reference_distances'].get('A', 'N/A'):.2f} mm)
      Sensor B: {result['distances'].get('B', 'N/A'):.2f} mm (Ref: {result['reference_distances'].get('B', 'N/A'):.2f} mm)
      Sensor C: {result['distances'].get('C', 'N/A'):.2f} mm (Ref: {result['reference_distances'].get('C', 'N/A'):.2f} mm)

    Confidence Scores:
      Sensor A: {result['confidences'].get('A', 0):.2f}
      Sensor B: {result['confidences'].get('B', 0):.2f}
      Sensor C: {result['confidences'].get('C', 0):.2f}
    """

    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='center',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('3-Radar Position Estimation Summary', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
