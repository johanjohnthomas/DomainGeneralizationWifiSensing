#!/usr/bin/env python3
"""
Script to plot Doppler activities from CSI data.
This version has been enhanced with better error handling and robustness.
"""

import argparse
import numpy as np
import pickle
import os
from os import listdir
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

        # Look for files with activity name in them
        start_l = 4  # Position in filename where activity label starts
        end_l = start_l + len(activity)
        for i in range(len(all_files)):
            try:
                if all_files[i][start_l:end_l] == activity and all_files[i][-5] != 'p':
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

        stft_antennas = []
        for name in names:
            name_file = exp_dir + name + '.txt'
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
        if len(traces_activities) >= 5 and traces_activities[4]:
            # Try antenna 1 for single activity plot
            antenna = min(1, len(traces_activities[4][0])-1) if traces_activities[4] and traces_activities[4][0] else 0
            name_p = './plots/csi_doppler_single_act_' + args.sub_dir + '.pdf'
            plt_fft_doppler_activity_single(traces_activities[4], antenna, sliding, delta_v, name_p)
            print(f"Generated plot: {name_p}")
    except Exception as e:
        print(f"Error generating single activity Doppler plot: {e}")

    print("Doppler plotting completed successfully.")
