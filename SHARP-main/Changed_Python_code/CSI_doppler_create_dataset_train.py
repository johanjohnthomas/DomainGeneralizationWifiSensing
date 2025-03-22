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
        exp_dir = args.dir + subdir + '/'

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

            label = subdir.split("_")[-1]

            if label not in csi_label_dict:
                processed = False
                continue
            processed = True

            print(name)

            label = convert_to_grouped_number(label, csi_label_dict)
            if i_name % n_tot == 0:
                prev_label = label
            elif label != prev_label:
                print('error in ' + str(name))
                break

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
            
            # Generate windows first and assign groups based on original streams
            window_length = args.windows_length  # number of windows considered
            stride_length = args.stride_lengths
            
            # Define minimum required stream length for meaningful window generation
            MIN_STREAM_LENGTH = 100  # Minimum time steps required in a stream (reduced from 300)
            
            # Override stride length to generate more windows
            stride_length = 1  # Was using args.stride_lengths (typically 30)
            
            print(f"\nWindow generation parameters:")
            print(f"  Window length: {window_length} time steps")
            print(f"  Stride length: {stride_length} time steps (overridden to 1)")
            print(f"  Minimum stream length: {MIN_STREAM_LENGTH} time steps")
            
            csi_windows = []
            window_labels = []
            window_groups = []  # Track original stream for each window
            
            valid_streams = 0
            skipped_streams = 0

            for i, csi_matrix in enumerate(csi_matrices_for_split):
                time_dim = csi_matrix.shape[1]  # Time steps in this stream
                
                # Calculate number of potential windows
                num_windows = (time_dim - window_length) // stride_length + 1
                
                if time_dim < MIN_STREAM_LENGTH:
                    print(f"  Skipping short stream {i}: {time_dim} < {MIN_STREAM_LENGTH} time steps")
                    skipped_streams += 1
                    continue
                    
                if num_windows < 2:
                    print(f"  Skipping stream {i}: Would only generate {num_windows} window(s)")
                    skipped_streams += 1
                    continue
                
                print(f"  Stream {i}: Generating {num_windows} windows from {time_dim} time steps")
                valid_streams += 1
                
                # Generate windows for this stream
                stream_windows, stream_labels = create_windows_antennas(
                    [csi_matrix], [labels_for_split[i]], 
                    window_length, stride_length, remove_mean=False
                )
                
                # Verify we got the expected number of windows
                if len(stream_windows) != num_windows:
                    print(f"  Warning: Expected {num_windows} windows but got {len(stream_windows)} for stream {i}")
                
                # Assign group ID (original stream index)
                group_id = i  # Use stream index as group ID
                window_groups.extend([group_id] * len(stream_windows))
                
                csi_windows.extend(stream_windows)
                window_labels.extend(stream_labels)

            # Convert to numpy arrays
            window_labels = np.array(window_labels)
            window_groups = np.array(window_groups)
            
            print(f"\nWindow generation summary:")
            print(f"  Used {valid_streams} streams, skipped {skipped_streams} streams")
            print(f"  Generated {len(csi_windows)} windows total")
            
            if valid_streams == 0:
                print(f"ERROR: No valid streams available for window generation")
                print(f"Consider reducing MIN_STREAM_LENGTH or adjusting window/stride parameters")
                continue  # Skip to the next subdirectory

            # Check if we have enough windows for splitting
            if len(csi_windows) < 3:
                print(f"WARNING: Not enough windows ({len(csi_windows)}) for proper splitting")
                
                # Handle edge cases with few windows
                if len(csi_windows) == 0:
                    print("CRITICAL ERROR: No windows available for this domain-activity combination")
                    print("Skipping this combination")
                    continue  # Skip to the next subdirectory
                
                elif len(csi_windows) == 1:
                    print("One window available - allocating to training set only")
                    X_train = csi_windows
                    y_train = window_labels
                    X_val = []  # Empty list
                    y_val = np.array([])
                    X_test = []  # Empty list
                    y_test = np.array([])
                    
                    print(f"Allocation: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                
                elif len(csi_windows) == 2:
                    print("Two windows available - allocating to training and validation sets")
                    X_train = [csi_windows[0]]
                    y_train = np.array([window_labels[0]])
                    X_val = [csi_windows[1]]
                    y_val = np.array([window_labels[1]])
                    X_test = []  # Empty list
                    y_test = np.array([])
                    
                    print(f"Allocation: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            else:
                # Split the WINDOWS (not streams) into train/val/test while preserving groups
                print("Using GroupShuffleSplit to split windows while preserving stream integrity")
                
                # First split: Train vs (Val+Test) - 60/40 split
                gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
                train_idx, temp_idx = next(gss.split(csi_windows, groups=window_groups))
                
                X_train = [csi_windows[i] for i in train_idx]
                y_train = window_labels[train_idx]
                
                # Second split: Val vs Test (ensuring groups stay together) - 50/50 split of the remaining 40%
                if len(temp_idx) >= 2:
                    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
                    val_indices, test_indices = next(gss_val_test.split(
                        [csi_windows[i] for i in temp_idx], 
                        groups=window_groups[temp_idx]
                    ))
                    
                    # Convert indices relative to temp_idx back to original indices
                    val_idx = temp_idx[val_indices]
                    test_idx = temp_idx[test_indices]
                    
                    X_val = [csi_windows[i] for i in val_idx]
                    y_val = window_labels[val_idx]
                    
                    X_test = [csi_windows[i] for i in test_idx]
                    y_test = window_labels[test_idx]
                else:
                    # If we don't have enough samples in temp for a proper split
                    if len(temp_idx) == 1:
                        val_idx = temp_idx
                        X_val = [csi_windows[i] for i in val_idx]
                        y_val = window_labels[val_idx]
                        X_test = []
                        y_test = np.array([])
                    else:
                        X_val = []
                        y_val = np.array([])
                        X_test = []
                        y_test = np.array([])
                
                # Print statistics about the split
                print(f"\nWindow splitting statistics:")
                print(f"Total windows: {len(csi_windows)}")
                print(f"After split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                
                # Check label distribution
                unique_labels = np.unique(window_labels)
                print("\nLabel distribution in windows:")
                for label in unique_labels:
                    orig_count = np.sum(window_labels == label)
                    train_count = np.sum(y_train == label) if len(y_train) > 0 else 0
                    val_count = np.sum(y_val == label) if len(y_val) > 0 else 0  
                    test_count = np.sum(y_test == label) if len(y_test) > 0 else 0
                    print(f"  Label {label}: Original: {orig_count}, Train: {train_count} ({train_count/orig_count:.2%}), Val: {val_count} ({val_count/orig_count:.2%}), Test: {test_count} ({test_count/orig_count:.2%})")

            list_sets_name = ['train', 'val', 'test']
            list_sets = [X_train, X_val, X_test]
            list_sets_labels = [y_train, y_val, y_test]

            for set_idx in range(3):
                # Skip processing empty sets
                if len(list_sets[set_idx]) == 0:
                    print(f"Skipping {list_sets_name[set_idx]} set - no samples allocated")
                    continue
                
                csi_matrices_set = list_sets[set_idx]
                labels_set = list_sets_labels[set_idx]

                # Since windows are already created, we don't need to compute num_windows
                # We already have exactly the right number of windows in csi_matrices_set
                num_windows = [len(csi_matrices_set)]  # Just store the total count
                
                print(f"Saving {len(csi_matrices_set)} windows to {list_sets_name[set_idx]} set")

                names_set = []
                suffix = '.txt'
                for ii in range(len(csi_matrices_set)):
                    name_file = exp_dir + list_sets_name[set_idx] + '_antennas_' + str(activities) + '/' + \
                                str(ii) + suffix
                    names_set.append(name_file)
                    with open(name_file, "wb") as fp:  # Pickling
                        # Save the full antenna data as is
                        pickle.dump(csi_matrices_set[ii], fp)
                name_labels = exp_dir + '/labels_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_labels, "wb") as fp:
                    # Use labels_set which contains the correct labels for each window
                    pickle.dump([int(label) for label in labels_set], fp)
                name_f = exp_dir + '/files_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(names_set, fp)
                name_f = exp_dir + '/num_windows_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(num_windows, fp)
