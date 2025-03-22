#!/usr/bin/env python3
"""
Comprehensive visualization generator for WiFi sensing data.
This script generates all types of plots available in the codebase using processed data.
"""

import os
import argparse
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from plots_utility import (
    # Signal Processing visualizations
    plot_r_angle, plot_r_abs, plot_abs_comparison, plot_angle_comparison,
    plot_gridspec_abs, plot_gridspec_angle, plt_amplitude, plt_phase,
    plt_amplitude_phase, plt_amplitude_phase_vert, plt_amplitude_phase_horiz,
    
    # Doppler visualizations
    plt_antennas, plt_doppler_antennas, plt_doppler_activities,
    plt_doppler_activities_compact, plt_doppler_activity_single,
    plt_doppler_comparison, plt_fft_doppler_activities,
    plt_fft_doppler_activities_compact, plt_fft_doppler_activities_compact_2,
    plt_fft_doppler_activity_single,
    
    # Performance visualizations
    plt_confusion_matrix
)


def safe_mkdir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def find_latest_file(directory, pattern):
    """Find the latest file matching a pattern in a directory."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def generate_confusion_matrix_plots(args):
    """Generate confusion matrix visualizations from metrics files."""
    print("\n== Generating Confusion Matrix Plots ==")
    
    metrics_file = find_latest_file(args.outputs_dir, "complete_different_*.txt")
    if not metrics_file:
        print("No metrics file found. Run 'make test' and 'make metrics' first.")
        return False
    
    print(f"Using metrics file: {metrics_file}")
    
    try:
        with open(metrics_file, "rb") as fp:
            conf_matrix_dict = pickle.load(fp)
        
        activities = np.array(['E', 'W', 'R', 'J', 'L', 'S', 'C', 'G'])
        conf_matrix = conf_matrix_dict['conf_matrix']
        conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']
        
        # Plot confusion matrices
        plt_confusion_matrix(activities.shape[0], conf_matrix, activities=activities, name="all_confusion_matrix")
        plt_confusion_matrix(activities.shape[0], conf_matrix_max_merge, activities=activities, name="all_max_merge_confusion_matrix")
        
        # Create accuracy bar charts
        plt.figure(figsize=(10, 6))
        acc_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
        plt.bar(activities, acc_per_class)
        plt.xlabel('Activity')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, 'all_per_class_accuracy.png'))
        plt.savefig(os.path.join(args.plots_dir, 'all_per_class_accuracy.pdf'))
        plt.close()
        
        print("Successfully generated confusion matrix visualizations")
        return True
    except Exception as e:
        print(f"Error generating confusion matrix plots: {e}")
        return False


def generate_phase_amplitude_plots(args):
    """Generate signal processing visualizations from phase data."""
    print("\n== Generating Phase and Amplitude Plots ==")
    
    # Check if processed phase directory exists
    if not os.path.isdir(args.phase_dir):
        print(f"Processed phase directory not found: {args.phase_dir}")
        print("Run 'make data_processing' first.")
        return False
    
    # Get a list of all setups
    setups = [d for d in os.listdir(args.phase_dir) 
              if os.path.isdir(os.path.join(args.phase_dir, d))]
    
    if not setups:
        print(f"No setup directories found in {args.phase_dir}")
        return False
    
    print(f"Found setups: {', '.join(setups)}")
    
    # Choose a representative setup
    setup = setups[0]
    setup_dir = os.path.join(args.phase_dir, setup)
    
    # Find phase data files
    phase_files = glob.glob(os.path.join(setup_dir, "*.txt"))
    
    if not phase_files:
        print(f"No phase data files found in {setup_dir}")
        return False
    
    # Get a sample phase file
    sample_file = phase_files[0]
    print(f"Using sample file: {sample_file}")
    
    try:
        # Load the sample data
        with open(sample_file, "rb") as fp:
            data = pickle.load(fp)
        
        # Check data structure to determine what kind of visualizations we can create
        if isinstance(data, dict) and 'amplitude' in data and 'phase_raw' in data and 'phase_sanitized' in data:
            # We have amplitude and phase data
            amplitude = data['amplitude']
            phase_raw = data['phase_raw']
            phase_sanitized = data['phase_sanitized']
            
            # Generate amplitude and phase plots
            plt_amplitude(amplitude, os.path.join(args.plots_dir, f"all_amplitude_{setup}"))
            plt_phase(phase_raw, phase_sanitized, os.path.join(args.plots_dir, f"all_phase_{setup}"))
            plt_amplitude_phase(amplitude, phase_raw, phase_sanitized, 
                             os.path.join(args.plots_dir, f"all_amplitude_phase_{setup}"))
            plt_amplitude_phase_vert(amplitude, phase_raw, phase_sanitized, 
                                  os.path.join(args.plots_dir, f"all_amplitude_phase_vert_{setup}"))
            plt_amplitude_phase_horiz(amplitude, phase_raw, phase_sanitized, 
                                   os.path.join(args.plots_dir, f"all_amplitude_phase_horiz_{setup}"))
            
            print("Successfully generated amplitude and phase visualizations")
            return True
        
        # If the data isn't in the expected format, try other potential formats
        if isinstance(data, dict) and 'H_true' in data and 'H_estimated' in data:
            # We have channel estimation data
            H_true = data['H_true']
            H_estimated = data['H_estimated']
            H_sanitized = data.get('H_sanitized', H_estimated)
            
            # Generate channel estimation plots
            plot_abs_comparison(H_true, H_estimated, os.path.join(args.plots_dir, f"all_abs_comparison_{setup}"))
            plot_angle_comparison(H_true, H_estimated, os.path.join(args.plots_dir, f"all_angle_comparison_{setup}"))
            plot_gridspec_abs(H_true, H_estimated, H_sanitized, 
                           os.path.join(args.plots_dir, f"all_gridspec_abs_{setup}"))
            plot_gridspec_angle(H_true, H_estimated, H_sanitized, 
                             os.path.join(args.plots_dir, f"all_gridspec_angle_{setup}"))
            
            print("Successfully generated channel estimation visualizations")
            return True
        
        print("Data format not recognized for phase/amplitude visualizations")
        return False
    
    except Exception as e:
        print(f"Error generating phase and amplitude plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_doppler_plots(args):
    """Generate Doppler spectrum visualizations."""
    print("\n== Generating Doppler Spectrum Plots ==")
    
    # Check if Doppler traces directory exists
    if not os.path.isdir(args.doppler_dir):
        print(f"Doppler traces directory not found: {args.doppler_dir}")
        print("Run 'make data_processing' first.")
        return False
    
    # Get a list of all setups
    setups = [d for d in os.listdir(args.doppler_dir) 
              if os.path.isdir(os.path.join(args.doppler_dir, d)) and d not in ['.', '..']]
    
    if not setups:
        print(f"No setup directories found in {args.doppler_dir}")
        return False
    
    print(f"Found setups: {', '.join(setups[:5])}...")
    
    # Choose a representative setup
    setup = setups[0]
    setup_dir = os.path.join(args.doppler_dir, setup)
    
    # Find different activity types
    activities = ['empty', 'sitting', 'walking', 'running', 'jumping']
    found_activities = []
    
    for activity in activities:
        activity_files = []
        all_files = os.listdir(setup_dir)
        for file in all_files:
            try:
                if len(file) > 4 and file[4:4+len(activity)] == activity and file.endswith('.txt'):
                    activity_files.append(os.path.join(setup_dir, file))
            except:
                continue
        
        if activity_files:
            found_activities.append({
                'name': activity,
                'files': activity_files
            })
    
    if not found_activities:
        print(f"No activity data files found in {setup_dir}")
        return False
    
    print(f"Found activities: {', '.join(act['name'] for act in found_activities)}")
    
    try:
        # Load sample data for each activity
        doppler_list = []
        for activity in found_activities:
            try:
                with open(activity['files'][0], "rb") as fp:
                    data = pickle.load(fp)
                data_log = 10 * np.log10(data)
                activity['data'] = data_log
                doppler_list.append(data_log)
            except Exception as e:
                print(f"Error loading data for activity {activity['name']}: {e}")
        
        if not doppler_list:
            print("No valid Doppler data could be loaded")
            return False
        
        # We need to organize the data for multi-antenna plots
        antenna_data = []
        for i in range(min(4, len(doppler_list))):
            antenna_data.append(doppler_list[i])
        
        # Calculate parameters needed for plotting
        Tc = 6e-3
        fc = 5e9
        v_light = 3e8
        feature_length = args.feature_length
        sliding = args.sliding
        delta_v = round(v_light / (Tc * fc * feature_length), 3)
        
        # Generate Doppler antenna plots
        if len(antenna_data) > 0:
            plt_antennas(antenna_data, os.path.join(args.plots_dir, f"all_antennas_{setup}"))
            plt_doppler_antennas(antenna_data, sliding, delta_v, 
                              os.path.join(args.plots_dir, f"all_doppler_antennas_{setup}"))
        
        # Generate Doppler activities plots if we have multiple activities
        if len(found_activities) >= 2:
            # Prepare data for activity plots
            activities_data = []
            activity_names = []
            
            for activity in found_activities[:5]:  # Limit to 5 activities max
                activities_data.append([activity['data']])  # Wrap in list for antenna dimension
                activity_names.append(activity['name'])
            
            # Generate Doppler activity plots
            plt_doppler_activities(activities_data, 0, activity_names, sliding, delta_v,
                                os.path.join(args.plots_dir, f"all_doppler_activities_{setup}"))
            
            # Generate compact versions if we have at least 3 activities
            if len(activities_data) >= 3:
                plt_doppler_activities_compact(activities_data, 0, activity_names, sliding, delta_v,
                                           os.path.join(args.plots_dir, f"all_doppler_activities_compact_{setup}"))
                
                # Create reduced version without walking
                reduced_data = list(activities_data)
                if len(reduced_data) > 2:
                    reduced_data.pop(2)  # Remove the third activity (typically walking)
                    reduced_names = activity_names.copy()
                    if len(reduced_names) > 2:
                        reduced_names.pop(2)
                    
                    plt_doppler_activities_compact_2(reduced_data, 0, reduced_names, sliding, delta_v,
                                                 os.path.join(args.plots_dir, f"all_doppler_activities_compact2_{setup}"))
        
        # Generate single activity plot (last activity)
        if found_activities:
            plt_doppler_activity_single(activities_data[-1], 0, sliding, delta_v,
                                     os.path.join(args.plots_dir, f"all_doppler_single_{setup}"))
        
        print("Successfully generated Doppler spectrum visualizations")
        return True
    
    except Exception as e:
        print(f"Error generating Doppler plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-dir', default='./plots',
                      help='Directory to save generated plots (default: ./plots)')
    parser.add_argument('--outputs-dir', default='./outputs',
                      help='Directory containing model outputs (default: ./outputs)')
    parser.add_argument('--phase-dir', default='./processed_phase',
                      help='Directory containing processed phase data (default: ./processed_phase)')
    parser.add_argument('--doppler-dir', default='./doppler_traces',
                      help='Directory containing Doppler traces (default: ./doppler_traces)')
    parser.add_argument('--feature-length', type=int, default=100,
                      help='Feature length for Doppler calculations (default: 100)')
    parser.add_argument('--sliding', type=int, default=1,
                      help='Sliding window for Doppler calculations (default: 1)')
    
    args = parser.parse_args()
    
    # Create plots directory
    safe_mkdir(args.plots_dir)
    
    print("===== WiFi Sensing Visualization Generator =====")
    print(f"Plots will be saved to: {os.path.abspath(args.plots_dir)}")
    
    # Generate all types of plots
    confusion_success = generate_confusion_matrix_plots(args)
    phase_success = generate_phase_amplitude_plots(args)
    doppler_success = generate_doppler_plots(args)
    
    # Summarize results
    print("\n===== Visualization Generation Summary =====")
    if confusion_success:
        print("✓ Confusion matrix plots successfully generated")
    else:
        print("✗ Confusion matrix plots could not be generated")
    
    if phase_success:
        print("✓ Phase and amplitude plots successfully generated")
    else:
        print("✗ Phase and amplitude plots could not be generated")
    
    if doppler_success:
        print("✓ Doppler spectrum plots successfully generated")
    else:
        print("✗ Doppler spectrum plots could not be generated")
    
    # List all generated plots
    print("\nAll generated visualizations:")
    for plot_file in sorted(os.listdir(args.plots_dir)):
        if plot_file.endswith('.pdf') or plot_file.endswith('.png'):
            print(f"  - {plot_file}")


if __name__ == '__main__':
    main() 