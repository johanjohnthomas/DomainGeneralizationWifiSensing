#!/usr/bin/env python3
"""
Script to generate Doppler comparison visualizations across different activities.
This script generates more visualization types than the standard CSI_doppler_plot_activities.py.
"""

import argparse
import numpy as np
import pickle
import os
import glob
from os import listdir
from plots_utility import (
    plt_doppler_activities,
    plt_doppler_activities_compact,
    plt_doppler_comparison,
    plt_antennas,
    plt_doppler_antennas
)


def safe_mkdir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def generate_comparison_plots(doppler_dir, setup, feature_length, sliding, labels_activities):
    """Generate comparison plots across activities."""
    print(f"Generating comparison plots for setup: {setup}")
    
    # Create plots directory
    safe_mkdir('./plots')
    
    # Physical constants
    Tc = 6e-3  # Time constant
    fc = 5e9   # Carrier frequency
    v_light = 3e8  # Speed of light
    
    # Calculate the Doppler velocity
    delta_v = round(v_light / (Tc * fc * feature_length), 3)
    
    # Default activities if none specified
    default_activities = np.asarray(['empty', 'sitting', 'walking', 'running', 'jumping'])
    
    # Parse the input activities
    activity_labels = []
    for lab_act in labels_activities.split(','):
        activity_labels.append(lab_act)
    
    # Make sure we have 5 activities for plotting
    if len(activity_labels) < 5:
        print(f"Warning: Only {len(activity_labels)} activities provided, filling with defaults")
        activity_labels.extend(default_activities[len(activity_labels):5])
    
    # Get directory
    exp_dir = os.path.join(doppler_dir, setup)
    
    if not os.path.isdir(exp_dir):
        print(f"Error: Directory {exp_dir} does not exist")
        return False
    
    # Process each activity
    all_activity_data = []
    for activity in activity_labels:
        print(f"Processing activity: {activity}")
        
        # Find activity files
        activity_files = []
        try:
            # Check if we have a direct activity directory
            activity_dir = os.path.join(exp_dir, f"{setup}_{activity}")
            if os.path.isdir(activity_dir):
                # Look for stream files
                for i in range(4):  # Try up to 4 antennas
                    stream_file = os.path.join(activity_dir, f"{setup}_{activity}_stream_{i}.txt")
                    if os.path.isfile(stream_file):
                        activity_files.append(stream_file)
            
            # If no files found, try to match in main directory
            if not activity_files:
                all_files = listdir(exp_dir)
                start_l = 4  # Position in filename where activity label starts
                for file in all_files:
                    try:
                        if file[start_l:start_l+len(activity)] == activity and file.endswith('.txt') and not file.endswith('.txt.p'):
                            activity_files.append(os.path.join(exp_dir, file))
                    except IndexError:
                        # Filename too short
                        continue
        except Exception as e:
            print(f"Error listing files for activity {activity}: {e}")
            all_activity_data.append([])
            continue
        
        if not activity_files:
            print(f"Warning: No files found for activity {activity}")
            all_activity_data.append([])
            continue
        
        # Load data from each file
        activity_data = []
        for file in activity_files:
            try:
                with open(file, "rb") as fp:
                    stft_data = pickle.load(fp)
                
                # Convert to log scale
                stft_data_log = 10 * np.log10(stft_data)
                activity_data.append(stft_data_log)
                print(f"  Loaded file: {os.path.basename(file)} - Shape: {stft_data_log.shape}")
            except Exception as e:
                print(f"Error loading file {file}: {e}")
        
        if activity_data:
            all_activity_data.append(activity_data)
        else:
            all_activity_data.append([])
    
    # If we don't have any data, exit
    if not any(all_activity_data):
        print("No valid data found for any activity")
        return False
    
    # Generate the comparison plots
    try:
        # 1. Basic antenna comparison (if we have multiple antennas for one activity)
        for i, activity_data in enumerate(all_activity_data):
            if activity_data and len(activity_data) > 1:
                activity_name = activity_labels[i] if i < len(activity_labels) else f"Activity {i}"
                name_p = f'./plots/doppler_antennas_comparison_{setup}_{activity_name}.pdf'
                plt_antennas(activity_data, name_p)
                print(f"Generated antenna comparison plot: {name_p}")
                
                # Also generate Doppler-specific antenna plot
                name_p = f'./plots/doppler_velocity_antennas_{setup}_{activity_name}.pdf'
                plt_doppler_antennas(activity_data, sliding, delta_v, name_p)
                print(f"Generated Doppler velocity antenna plot: {name_p}")
        
        # 2. Activity comparison plots (using the first antenna)
        first_antenna_data = []
        valid_labels = []
        
        for i, activity_data in enumerate(all_activity_data):
            if activity_data:
                first_antenna_data.append(activity_data[0])
                valid_labels.append(activity_labels[i] if i < len(activity_labels) else f"Activity {i}")
        
        if len(first_antenna_data) > 1:
            # Direct comparison plot
            name_p = f'./plots/doppler_direct_comparison_{setup}.pdf'
            plt_doppler_comparison(first_antenna_data, valid_labels, sliding, delta_v, name_p)
            print(f"Generated direct comparison plot: {name_p}")
            
            # Format data for activities plot which expects antennas in a different format
            formatted_data = []
            for data in first_antenna_data:
                formatted_data.append([data])
            
            # Traditional activities plot
            name_p = f'./plots/doppler_activities_compare_{setup}.pdf'
            plt_doppler_activities(formatted_data, 0, valid_labels, sliding, delta_v, name_p)
            print(f"Generated activities comparison plot: {name_p}")
            
            # Compact activities plot
            name_p = f'./plots/doppler_activities_compact_compare_{setup}.pdf'
            plt_doppler_activities_compact(formatted_data, 0, valid_labels, sliding, delta_v, name_p)
            print(f"Generated compact activities comparison plot: {name_p}")
        
        print(f"Successfully generated comparison plots for {setup}")
        return True
    
    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('doppler_dir', help='Directory containing Doppler traces')
    parser.add_argument('setup', help='Setup identifier (e.g., AR1a)')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('labels_activities', help='Comma-separated list of activity labels')
    
    args = parser.parse_args()
    
    generate_comparison_plots(args.doppler_dir, args.setup, args.feature_length, args.sliding, args.labels_activities)


if __name__ == '__main__':
    main() 