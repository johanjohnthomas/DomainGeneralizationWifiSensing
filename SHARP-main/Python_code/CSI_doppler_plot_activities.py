#!/usr/bin/env python3
"""
Script to plot Doppler activities from CSI data.
This version has been enhanced with better error handling and robustness.
"""

import argparse
import numpy as np
import pickle
import os
import re
from os import listdir
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from plots_utility import plt_fft_doppler_activities, plt_fft_doppler_activities_compact, \
    plt_fft_doppler_activity_single, plt_fft_doppler_activities_compact_2


def safe_delete_activity(activities_list, index):
    """Safely delete an activity from the list if it exists."""
    if activities_list and 0 <= index < len(activities_list):
        del activities_list[index]
    return activities_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('sub_dir', help='Sub directory of data')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('labels_activities', help='Comma-separated list of activity labels')
    parser.add_argument('start_plt', help='Start index to plot', type=int)
    parser.add_argument('end_plt', help='End index to plot', type=int)

    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)

    # Default activities if none specified
    default_activities = np.asarray(['empty', 'sitting', 'walking', 'running', 'jumping'])
    
    # Parse the input activities
    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)
    
    # Use default activities if we don't have all 5
    if len(csi_label_dict) < 5:
        print(f"Warning: Only {len(csi_label_dict)} activities provided, using default labels for missing ones")
        activities = default_activities
    else:
        activities = np.asarray(csi_label_dict[:5])  # Limit to first 5 for plotting

    feature_length = args.feature_length
    sliding = args.sliding
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8

    exp_dir = os.path.join(args.dir, args.sub_dir) + '/'
    
    # Verify directory exists
    if not os.path.isdir(exp_dir):
        print(f"Error: Directory {exp_dir} does not exist")
        exit(1)

    # First, let's scan for all available activities in the directory
    available_activities = set()
    
    try:
        all_files = listdir(exp_dir)
        # Check subdirectories (which might contain activity data)
        subdirs = [d for d in all_files if os.path.isdir(os.path.join(exp_dir, d))]
        
        # Look for activity subdirectories (format: AR6a_J1, AR6a_W2, etc.)
        activity_pattern = re.compile(r'{}_(.*?)(?:/|$)'.format(args.sub_dir))
        for subdir in subdirs:
            match = activity_pattern.match(subdir)
            if match:
                act_with_num = match.group(1)
                # Extract the base letter (activity type)
                base_activity = act_with_num[0]
                available_activities.add(base_activity)
                print(f"Found activity directory: {subdir} - Base activity: {base_activity}")
    except Exception as e:
        print(f"Error scanning activities in directory {exp_dir}: {e}")
        exit(1)
    
    print(f"Available base activities in this directory: {', '.join(sorted(available_activities))}")

    traces_activities = []
    for ilab in range(len(csi_label_dict)):
        names = []
        try:
            all_files = listdir(exp_dir)
        except Exception as e:
            print(f"Error listing directory {exp_dir}: {e}")
            exit(1)
            
        activity = csi_label_dict[ilab]
        print(f"Processing activity: {activity}")
        
        # Skip this activity if its base letter is not found in available_activities
        if activity not in available_activities:
            print(f"Warning: Activity {activity} not available in this domain, skipping")
            traces_activities.append([])
            continue

        # Look for files with activity name in them, including numbered variations
        start_l = 4  # Position in filename where activity label starts
        
        # First, find all subdirectories for this activity
        activity_dirs = []
        for subdir in all_files:
            if os.path.isdir(os.path.join(exp_dir, subdir)) and subdir.startswith(f"{args.sub_dir}_{activity}"):
                activity_dirs.append(subdir)
        
        # Check each activity directory
        for activity_dir in activity_dirs:
            dir_path = os.path.join(exp_dir, activity_dir)
            try:
                dir_files = listdir(dir_path)
                for file in dir_files:
                    if file.endswith('.txt') and not file.endswith('.txt.p'):
                        names.append(os.path.join(activity_dir, file[:-4]))
            except Exception as e:
                print(f"Error listing directory {dir_path}: {e}")
                continue
        
        # Also check for direct files in the main directory
        for i in range(len(all_files)):
            try:
                # Check if file starts with base activity letter at position 4
                if len(all_files[i]) > 4 and all_files[i][start_l] == activity and all_files[i][-5] != 'p':
                    # Verify it's an activity file (might have numbers after the letter)
                    next_char = all_files[i][start_l + 1] if len(all_files[i]) > start_l + 1 else ""
                    if next_char.isdigit() or next_char == "_" or next_char == ".":
                        names.append(all_files[i][:-4])
            except IndexError:
                # Filename too short, skip
                continue

        names.sort()
        
        if not names:
            print(f"Warning: No files found for activity {activity}")
            # Add an empty list so indexes still align
            traces_activities.append([])
            continue
        else:
            print(f"Found {len(names)} files for activity {activity}: {', '.join(os.path.basename(n) for n in names[:3])}" + 
                  (f"... and {len(names)-3} more" if len(names) > 3 else ""))

        stft_antennas = []
        for name in names:
            name_file = exp_dir + name + '.txt' if not name.startswith(args.sub_dir) else os.path.join(exp_dir, name + '.txt')
            print(f"Processing file: {name_file}")

            try:
                with open(name_file, "rb") as fp:
                    stft_sum_1 = pickle.load(fp)

                stft_sum_1_log = 10*np.log10(stft_sum_1)
                # Ensure start_plt is within bounds
                start_idx = min(args.start_plt, stft_sum_1_log.shape[0]-1) if stft_sum_1_log.shape[0] > 0 else 0
                # Ensure end_plt doesn't exceed array bounds
                end_idx = min(stft_sum_1_log.shape[0], args.end_plt) if args.end_plt > 0 else stft_sum_1_log.shape[0]
                
                stft_sum_1_log = stft_sum_1_log[start_idx:end_idx, :]
                stft_antennas.append(stft_sum_1_log)
            except Exception as e:
                print(f"Error processing file {name_file}: {e}")
                continue

        if stft_antennas:
            traces_activities.append(stft_antennas)
            print(f"Successfully processed {len(stft_antennas)} data files for activity {activity}")
        else:
            print(f"Warning: No valid data processed for activity {activity}")
            # Add an empty list to maintain index alignment
            traces_activities.append([])

    # Calculate the Doppler velocity
    delta_v = round(v_light / (Tc * fc * feature_length), 3)
    
    # Default to antenna 0
    antenna = 0
    
    try:
        # Generate the main Doppler activities plot
        name_p = './plots/csi_doppler_activities_' + args.sub_dir + '.pdf'
        plt_fft_doppler_activities(traces_activities, antenna, activities, sliding, delta_v, name_p)
        print(f"Generated plot: {name_p}")
    except Exception as e:
        print(f"Error generating Doppler activities plot: {e}")

    try:
        # Generate compact version
        name_p = './plots/csi_doppler_activities_' + '_' + args.sub_dir + '_compact.pdf'
        plt_fft_doppler_activities_compact(traces_activities, antenna, activities, sliding, delta_v, name_p)
        print(f"Generated plot: {name_p}")
    except Exception as e:
        print(f"Error generating compact Doppler activities plot: {e}")

    try:
        # Create a copy for the reduced plot (removing walking)
        traces_activities_reduced = list(traces_activities)  # Create a copy
        
        # If we have at least 3 activities, remove the third one (walking)
        if len(traces_activities_reduced) >= 3:
            traces_activities_reduced = safe_delete_activity(traces_activities_reduced, 2)
            reduced_activities = np.asarray(['empty', 'sitting', 'running', 'jumping'])
            
            # Generate compact 2 version
            name_p = './plots/csi_doppler_activities_' + '_' + args.sub_dir + '_compact_2.pdf'
            plt_fft_doppler_activities_compact_2(traces_activities_reduced, antenna, reduced_activities, sliding, delta_v, name_p)
            print(f"Generated plot: {name_p}")
    except Exception as e:
        print(f"Error generating compact 2 Doppler activities plot: {e}")

    try:
        # Check if we have jumping activity data (5th activity)
        if len(traces_activities) >= 5 and len(traces_activities[4]) > 0:
            # Try antenna 1 for single activity plot
            antenna = min(1, len(traces_activities[4][0])-1) if traces_activities[4] and len(traces_activities[4][0]) > 0 else 0
            name_p = './plots/csi_doppler_single_act_' + args.sub_dir + '.pdf'
            plt_fft_doppler_activity_single(traces_activities[4], antenna, sliding, delta_v, name_p)
            print(f"Generated plot: {name_p}")
    except Exception as e:
        print(f"Error generating single activity Doppler plot: {e}")

    print("Doppler plotting completed successfully.")
    plt.close('all')  # Close all remaining figure windows
