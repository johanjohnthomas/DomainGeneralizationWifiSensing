"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import glob
import os
import numpy as np
import pickle
import math as mt
import shutil
from dataset_utility import create_windows_antennas, convert_to_number, convert_to_grouped_number
from sklearn.model_selection import train_test_split, GroupShuffleSplit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Sub-directories')
    parser.add_argument('sample_lengths', help='Number of packets in a sample', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('windows_length', help='Number of samples per window', type=int)
    parser.add_argument('stride_lengths', help='Number of samples to stride', type=int)
    parser.add_argument('labels_activities', help='Labels of the activities to be considered')
    parser.add_argument('n_tot', help='Number of streams * number of antennas', type=int)
    args = parser.parse_args()

    labels_activities = args.labels_activities
    csi_label_dict = []
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)

    activities = np.asarray(labels_activities)

    n_tot = args.n_tot
    num_packets = args.sample_lengths  # 51
    middle = int(np.floor(num_packets / 2))
    list_subdir = args.subdirs  # string

    for subdir in list_subdir.split(','):
        # Check if we're working with a setup+activity subdirectory format (like "AR9b_J1")
        # or just a setup format (like "AR9b")
        if '_' in subdir:
            # Extract setup ID from the subdirectory name (e.g., "AR9b" from "AR9b_J1")
            setup_id = subdir.split('_')[0]
            exp_dir = args.dir + subdir + '/'
            print(f"Processing subdirectory: {subdir} (setup ID: {setup_id})")
        else:
            # If no underscore, use the whole subdirectory name as the setup_id
            setup_id = subdir
            exp_dir = args.dir + subdir + '/'
            print(f"Processing subdirectory: {subdir}")

        # Ensure the exp_dir exists
        if not os.path.exists(exp_dir):
            print(f"Creating directory: {exp_dir}")
            os.makedirs(exp_dir, exist_ok=True)

        path_train = exp_dir + 'train_antennas_' + str(activities)
        path_val = exp_dir + 'val_antennas_' + str(activities)
        path_test = exp_dir + 'test_antennas_' + str(activities)
        paths = [path_train, path_val, path_test]
        for pat in paths:
            if os.path.exists(pat):
                remove_files = glob.glob(pat + '/*')
                for f in remove_files:
                    os.remove(f)
            else:
                os.makedirs(pat, exist_ok=True)
                print(f"Created directory: {pat}")

        path_complete = exp_dir + 'complete_antennas_' + str(activities)
        if os.path.exists(path_complete):
            shutil.rmtree(path_complete)

        names = []
        all_files = os.listdir(exp_dir)
        for filename in all_files:
            if filename.endswith('.txt') and not filename.startswith('.') and '_stream_' in filename:
                names.append(filename[:-4])
        names.sort()

        csi_matrices = []
        labels = []
        lengths = []
        label = 'null'
        prev_label = label
        csi_matrix = []
        processed = False
        for i_name, name in enumerate(names):
            if i_name % n_tot == 0 and i_name != 0 and processed:
                ll = csi_matrix[0].shape[1]

                for i_ant in range(1, n_tot):
                    if ll != csi_matrix[i_ant].shape[1]:
                        break
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                csi_matrix = []

            # Extract activity from filename instead of subdirectory
            # Handle multiple potential filename formats
            try:
                filename_parts = name.split('_')
                
                # Format specified by user: "ARxy_Z..." where activity is in the third part
                if len(filename_parts) >= 3:
                    # Extract activity from the third part as specified (index 2)
                    activity_code = filename_parts[2]
                elif len(filename_parts) >= 2:
                    # Fallback to second part if third doesn't exist
                    activity_code = filename_parts[1]
                else:
                    # Last resort fallback to the original method
                    activity_code = subdir.split("_")[-1]
                
                # Handle cases where activity code might have number suffix (e.g., "J2")
                # Extract just the letter part if needed
                if activity_code and not activity_code[0].isdigit():
                    base_activity = activity_code[0]
                    if base_activity in csi_label_dict:
                        activity_code = base_activity
            except:
                # Fallback in case of parsing errors
                activity_code = subdir.split("_")[-1]
                print(f"Warning: Using fallback label extraction for {name}")
            
            if activity_code not in csi_label_dict:
                processed = False
                continue
            processed = True

            print(f"Processing {name}, activity code: {activity_code}")

            # Convert the activity code to a label number
            label = convert_to_grouped_number(activity_code, csi_label_dict)
            if i_name % n_tot == 0:
                prev_label = label
            elif label != prev_label:
                print(f'ERROR: Label mismatch in {str(name)}, got {label} vs previous {prev_label}')
                break

            print(f"File: {name}, Activity: {activity_code}, Label: {label}")

            name_file = exp_dir + name + '.txt'
            with open(name_file, "rb") as fp:
                stft_sum_1 = pickle.load(fp)
                
            # Convert lists to numpy arrays
            if isinstance(stft_sum_1, list):
                stft_sum_1 = np.array(stft_sum_1, dtype=np.float32)

            # Validate array type
            if stft_sum_1.dtype != np.float32 and stft_sum_1.dtype != np.float64:
                stft_sum_1 = stft_sum_1.astype(np.float32)
            stft_sum_1_mean = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_mean.T)

        error = False
        if processed:
            # for the last block
            if len(csi_matrix) < n_tot:
                print('error in ' + str(name))
            ll = csi_matrix[0].shape[1]

            for i_ant in range(1, n_tot):
                if ll != csi_matrix[i_ant].shape[1]:
                    print('error in ' + str(name))
                    error = True
            if not error:
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)

        if not error:
            lengths = np.asarray(lengths)
            if len(lengths) == 0:
                print("Error: No valid data was processed. The lengths array is empty.")
                continue  # Skip to the next subdir
            length_min = np.min(lengths)

            # Print distribution of extracted labels for verification
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            print("\nExtracted label distribution:")
            for lab, count in zip(unique_labels, label_counts):
                matching_activities = [act for act in csi_label_dict if convert_to_grouped_number(act, csi_label_dict) == lab]
                print(f"  Label {lab} ({', '.join(matching_activities)}): {count} samples")

            # Convert data to format suitable for sklearn's train_test_split
            csi_matrices_for_split = []
            
            # Prepare data for splitting
            for i in range(len(labels)):
                # Extract features from each CSI matrix up to the minimum length
                csi_matrix = csi_matrices[i][:, :, :length_min]
                # Store as a sample for splitting
                csi_matrices_for_split.append(csi_matrix)
            
            # For stratification to work correctly, we need one label per sample
            labels_for_split = np.array(labels)
            
            # Check if we have enough samples to do stratified splitting
            # We need at least 3 samples: 1 for train, 1 for val, 1 for test (minimum viable split)
            total_samples = len(csi_matrices_for_split)
            
            print(f"\nPreparing to split {total_samples} samples across train, validation, and test sets")
            
            # Get counts for each class to check if stratification is possible
            unique_labels, label_counts = np.unique(labels_for_split, return_counts=True)
            min_class_count = np.min(label_counts)
            
            print(f"Label distribution before splitting:")
            for label, count in zip(unique_labels, label_counts):
                print(f"  Class {label}: {count} samples")
            
            # USING GROUPSHUFFLESPLIT TO PREVENT DATA LEAKAGE
            # Extract subdirectory information to use as groups
            # This ensures samples from the same subject/environment stay together
            subdir_groups = []
            for i in range(len(names)):
                if i % n_tot == 0 and processed:
                    # Extract a unique identifier for the recording session/subject
                    # This uses the naming convention from the input files
                    session_id = names[i].split('_')[0]  # Adjust based on your naming convention
                    subdir_groups.append(session_id)
            
            print(f"Using GroupShuffleSplit to prevent data leakage across {len(np.unique(subdir_groups))} unique groups")
            
            if total_samples >= 3 and min_class_count >= 3:
                print("Using group-based splitting with 60/20/20 ratio")
                
                # First split: Train vs (Val+Test)
                gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
                train_idx, temp_idx = next(gss.split(csi_matrices_for_split, groups=subdir_groups))
                
                X_train = [csi_matrices_for_split[i] for i in train_idx]
                y_train = labels_for_split[train_idx]
                
                # Create temporary lists for second split
                X_temp = [csi_matrices_for_split[i] for i in temp_idx]
                y_temp = labels_for_split[temp_idx]
                temp_groups = [subdir_groups[i] for i in temp_idx]
                
                # Second split: Val vs Test (ensuring groups stay together)
                if len(X_temp) >= 2:
                    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
                    val_idx, test_idx = next(gss_val_test.split(X_temp, groups=temp_groups))
                    
                    X_val = [X_temp[i] for i in val_idx]
                    y_val = np.array([y_temp[i] for i in val_idx])
                    
                    X_test = [X_temp[i] for i in test_idx]
                    y_test = np.array([y_temp[i] for i in test_idx])
                else:
                    # If we don't have enough samples for a proper split
                    if len(X_temp) == 1:
                        X_val = X_temp
                        y_val = y_temp
                        X_test = []
                        y_test = np.array([])
                    else:
                        X_val = []
                        y_val = np.array([])
                        X_test = []
                        y_test = np.array([])
            
            else:
                print(f"WARNING: Not enough samples for proper group splitting (Total: {total_samples}, Min per class: {min_class_count})")
                print(f"Using a prioritized allocation strategy:")
                
                # Instead of duplicating samples, we'll prioritize allocation based on how many samples we have
                if total_samples == 0:
                    print("CRITICAL ERROR: No samples available for this domain-activity combination")
                    print("Skipping this combination")
                    continue  # Skip to the next subdirectory
                
                elif total_samples == 1:
                    print("One sample available - allocating to training set only")
                    X_train = csi_matrices_for_split
                    y_train = labels_for_split
                    X_val = []  # Empty list
                    y_val = np.array([])
                    X_test = []  # Empty list
                    y_test = np.array([])
                    
                    print(f"Allocation: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                
                elif total_samples == 2:
                    print("Two samples available - allocating to training and validation sets")
                    X_train = [csi_matrices_for_split[0]]
                    y_train = np.array([labels_for_split[0]])
                    X_val = [csi_matrices_for_split[1]]
                    y_val = np.array([labels_for_split[1]])
                    X_test = []  # Empty list
                    y_test = np.array([])
                    
                    print(f"Allocation: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                
                else:  # 3+ samples but not enough per class for proper stratification
                    # Allocate based on groups to maintain separation
                    unique_groups = np.unique(subdir_groups)
                    
                    if len(unique_groups) >= 3:
                        # If we have at least 3 unique groups, assign to train/val/test
                        np.random.seed(42)
                        np.random.shuffle(unique_groups)
                        
                        # Allocate groups based on 60/20/20 split
                        train_size = max(1, int(len(unique_groups) * 0.6))
                        val_size = max(1, int(len(unique_groups) * 0.2))
                        test_size = len(unique_groups) - train_size - val_size
                        
                        train_groups = unique_groups[:train_size]
                        val_groups = unique_groups[train_size:train_size+val_size]
                        test_groups = unique_groups[train_size+val_size:]
                        
                        # Assign samples to splits based on their group
                        train_indices = [i for i, g in enumerate(subdir_groups) if g in train_groups]
                        val_indices = [i for i, g in enumerate(subdir_groups) if g in val_groups]
                        test_indices = [i for i, g in enumerate(subdir_groups) if g in test_groups]
                        
                        X_train = [csi_matrices_for_split[i] for i in train_indices]
                        y_train = labels_for_split[train_indices]
                        
                        X_val = [csi_matrices_for_split[i] for i in val_indices]
                        y_val = labels_for_split[val_indices]
                        
                        X_test = [csi_matrices_for_split[i] for i in test_indices]
                        y_test = labels_for_split[test_indices]
                    
                    elif len(unique_groups) == 2:
                        # If we have 2 unique groups, assign to train/val
                        group1_indices = [i for i, g in enumerate(subdir_groups) if g == unique_groups[0]]
                        group2_indices = [i for i, g in enumerate(subdir_groups) if g == unique_groups[1]]
                        
                        X_train = [csi_matrices_for_split[i] for i in group1_indices]
                        y_train = labels_for_split[group1_indices]
                        
                        X_val = [csi_matrices_for_split[i] for i in group2_indices]
                        y_val = labels_for_split[group2_indices]
                        
                        X_test = []  # Empty list
                        y_test = np.array([])
                    
                    else:  # Only 1 unique group
                        # All samples go to training
                        X_train = csi_matrices_for_split
                        y_train = labels_for_split
                        X_val = []
                        y_val = np.array([])
                        X_test = []
                        y_test = np.array([])
                    
                    print(f"Allocation by groups: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Prepare for window creation (convert back to the format expected by create_windows_antennas)
            csi_train = X_train
            csi_val = X_val
            csi_test = X_test
            
            # Update the labels to match the new split
            train_labels = y_train
            val_labels = y_val
            test_labels = y_test
            
            # Compute lengths for each split
            length_train = [x.shape[2] for x in csi_train]
            length_val = [x.shape[2] for x in csi_val]
            length_test = [x.shape[2] for x in csi_test]
            
            # Print out statistics about the split to verify stratification
            print(f"\nData splitting statistics:")
            print(f"Original data: {len(labels)} samples")
            print(f"After split: Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
            
            # Check label distribution
            unique_labels = np.unique(labels_for_split)
            print("\nLabel distribution:")
            for label in unique_labels:
                orig_count = np.sum(labels_for_split == label)
                train_count = np.sum(train_labels == label)
                val_count = np.sum(val_labels == label)
                test_count = np.sum(test_labels == label)
                print(f"  Label {label}: Original: {orig_count}, Train: {train_count} ({train_count/orig_count:.2%}), Val: {val_count} ({val_count/orig_count:.2%}), Test: {test_count} ({test_count/orig_count:.2%})")

            window_length = args.windows_length  # number of windows considered
            stride_length = args.stride_lengths

            list_sets_name = ['train', 'val', 'test']
            list_sets = [csi_train, csi_val, csi_test]
            list_sets_lengths = [length_train, length_val, length_test]
            list_sets_labels = [train_labels, val_labels, test_labels]

            for set_idx in range(3):
                # Skip processing empty sets
                if len(list_sets[set_idx]) == 0:
                    print(f"Skipping {list_sets_name[set_idx]} set - no samples allocated")
                    continue

                # Ensure the output directory exists
                output_dir = exp_dir + list_sets_name[set_idx] + '_antennas_' + str(activities)
                if not os.path.exists(output_dir):
                    print(f"Creating output directory: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)

                csi_matrices_set, labels_set = create_windows_antennas(list_sets[set_idx], list_sets_labels[set_idx], window_length,
                                                                       stride_length, remove_mean=False)

                num_windows = np.floor((np.asarray(list_sets_lengths[set_idx]) - window_length) / stride_length + 1)
                if not len(csi_matrices_set) == np.sum(num_windows):
                    print('ERROR - shapes mismatch')

                names_set = []
                suffix = '.txt'
                for ii in range(len(csi_matrices_set)):
                    # Construct the output filepath in the appropriate directory
                    name_file = os.path.join(output_dir, f"{ii}{suffix}")
                    
                    # Double-check that the directory exists
                    os.makedirs(os.path.dirname(name_file), exist_ok=True)
                    
                    names_set.append(name_file)
                    try:
                        with open(name_file, "wb") as fp:  # Pickling
                            # Save the full antenna data as is
                            pickle.dump(csi_matrices_set[ii], fp)
                    except Exception as e:
                        print(f"Error writing to {name_file}: {e}")
                        
                # Create label file in the proper directory
                name_labels = os.path.join(exp_dir, f"labels_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                print(f"Creating label file: {name_labels}")
                try:
                    with open(name_labels, "wb") as fp:
                        # Use labels_set which contains the correct labels for each window
                        pickle.dump([int(label) for label in labels_set], fp)
                except Exception as e:
                    print(f"Error creating label file {name_labels}: {e}")
                    
                # Create metadata files in the proper directory
                name_f = os.path.join(exp_dir, f"files_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                print(f"Creating metadata file: {name_f}")
                try:
                    with open(name_f, "wb") as fp:  # Pickling
                        pickle.dump(names_set, fp)
                except Exception as e:
                    print(f"Error creating metadata file {name_f}: {e}")
                    
                name_f = os.path.join(exp_dir, f"num_windows_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                try:
                    with open(name_f, "wb") as fp:  # Pickling
                        pickle.dump(num_windows, fp)
                except Exception as e:
                    print(f"Error creating window metadata file {name_f}: {e}")
