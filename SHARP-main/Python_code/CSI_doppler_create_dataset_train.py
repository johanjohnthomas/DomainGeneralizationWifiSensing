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
from dataset_utility import create_windows_antennas, convert_to_number


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

    activities = "_".join(labels_activities.split(','))  # Convert "E,W,R,J,L,S,C,G" to "E_W_R_J_L_S_C_G"

    n_tot = args.n_tot
    num_packets = args.sample_lengths  # 51
    middle = int(np.floor(num_packets / 2))
    list_subdir = args.subdirs  # string
    
    print(f"Labels dictionary: {csi_label_dict}")
    print(f"Activities string: {activities}")
    print(f"Number of streams * antennas: {n_tot}")

    for subdir in list_subdir.split(','):
        exp_dir = os.path.join(args.dir, subdir)
        
        # Debug: Print the directory being processed
        print(f"Processing directory: {exp_dir}")

        path_train = os.path.join(exp_dir, f'train_antennas_{activities}')
        path_val = os.path.join(exp_dir, f'val_antennas_{activities}')
        path_test = os.path.join(exp_dir, f'test_antennas_{activities}')
        paths = [path_train, path_val, path_test]
        for pat in paths:
            if os.path.exists(pat):
                remove_files = glob.glob(os.path.join(pat, '*'))
                for f in remove_files:
                    os.remove(f)
                print(f"Cleaned existing directory: {pat}")
            else:
                os.makedirs(pat, exist_ok=True)
                print(f"Created directory: {pat}")

        path_complete = os.path.join(exp_dir, f'complete_antennas_{activities}')
        if os.path.exists(path_complete):
            shutil.rmtree(path_complete)
            print(f"Removed existing directory: {path_complete}")

        names = []
        # Recursively find all .txt files in the subdirectory
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith('.txt') and 'stream' in file:
                    names.append(os.path.join(root, file[:-4]))  # Remove .txt extension
        names.sort()

        # Debug: Print the number of files found
        print(f"Found {len(names)} files")
        print(f"First few files: {names[:min(5, len(names))]}")

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
                        print(f"Shape mismatch in antenna {i_ant}: expected {ll}, got {csi_matrix[i_ant].shape[1]}")
                        break
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                csi_matrix = []
                print(f"Added matrix with shape {csi_matrices[-1].shape} and label {label}")

            # Extract activity label from filename (e.g., "J" from "AR1a_J1_stream_0")
            label = os.path.basename(name).split('_')[1][0]

            # Debug: Print the extracted label
            print(f"Extracted label: {label} from file: {os.path.basename(name)}")

            if label not in csi_label_dict:
                processed = False
                print(f"Skipping label {label} not in dictionary {csi_label_dict}")
                continue
            processed = True

            label = convert_to_number(label, csi_label_dict)
            if i_name % n_tot == 0:
                prev_label = label
            elif label != prev_label:
                print('error in ' + str(name))
                break

            name_file = name + '.txt'
            with open(name_file, "rb") as fp:  # Unpickling
                stft_sum_1 = pickle.load(fp)

            # Debug: Print the shape of the loaded data
            print(f"Loaded data shape: {stft_sum_1.shape}")

            stft_sum_1_mean = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_mean.T)

        error = False
        if processed:
            # for the last block
            if len(csi_matrix) < n_tot:
                print(f"Error in {name}: insufficient antennas. Got {len(csi_matrix)}, expected {n_tot}")
            ll = csi_matrix[0].shape[1]

            for i_ant in range(1, n_tot):
                if ll != csi_matrix[i_ant].shape[1]:
                    print(f"Error in {name}: shape mismatch in antenna {i_ant}. Expected {ll}, got {csi_matrix[i_ant].shape[1]}")
                    error = True
            if not error:
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                print(f"Added final matrix with shape {csi_matrices[-1].shape} and label {label}")

        if not error:
            lengths = np.asarray(lengths)
            length_min = np.min(lengths)
            print(f"Total matrices: {len(csi_matrices)}, Total labels: {len(labels)}")
            print(f"Lengths: min={length_min}, max={np.max(lengths)}, avg={np.mean(lengths)}")

            csi_train = []
            csi_val = []
            csi_test = []
            length_train = []
            length_val = []
            length_test = []
            for i in range(len(labels)):
                ll = lengths[i]
                truncate = mt.ceil(num_packets / args.sliding)
                # Truncate the data to remove initial and final segments
                truncated_data = csi_matrices[i][:, :, truncate : ll - truncate]
                total_len = truncated_data.shape[2]
                
                train_len = int(total_len * 0.6)
                val_len = int(total_len * 0.2)
                test_len = total_len - train_len - val_len

                # Split the truncated data contiguously
                csi_train.append(truncated_data[:, :, :train_len])
                length_train.append(train_len)

                csi_val.append(truncated_data[:, :, train_len : train_len + val_len])
                length_val.append(val_len)

                csi_test.append(truncated_data[:, :, train_len + val_len :])
                length_test.append(test_len)

            print(f"Split data: train={len(csi_train)}, val={len(csi_val)}, test={len(csi_test)}")
            print(f"Truncated data shape: {truncated_data.shape}")
            print(f"Train len: {train_len}, Val len: {val_len}, Test len: {test_len}")


            window_length = args.windows_length  # number of windows considered
            stride_length = args.stride_lengths

            list_sets_name = ['train', 'val', 'test']
            list_sets = [csi_train, csi_val, csi_test]
            list_sets_lengths = [length_train, length_val, length_test]

            for set_idx in range(3):
                print(f"Processing {list_sets_name[set_idx]} set")
                csi_matrices_set, labels_set = create_windows_antennas(list_sets[set_idx], labels, window_length,
                                                                       stride_length, remove_mean=False)
                num_windows = ((np.asarray(list_sets_lengths[set_idx]) - window_length) // stride_length) + 1
                print(f"Window length: {window_length}, Stride length: {stride_length}")
                print(f"Lengths of data: {list_sets_lengths[set_idx]}")
                print(f"Calculated number of windows: {num_windows}")
                print(f"Actual number of windows: {len(csi_matrices_set)}")
                if not len(csi_matrices_set) == np.sum(num_windows):
                    print(f'ERROR - shapes mismatch in {list_sets_name[set_idx]}: {len(csi_matrices_set)} vs {np.sum(num_windows)}')
                else:
                    print(f"Created {len(csi_matrices_set)} windows for {list_sets_name[set_idx]} set")

                names_set = []
                suffix = '.txt'
                for ii in range(len(csi_matrices_set)):
                    name_file = os.path.join(exp_dir, f"{list_sets_name[set_idx]}_antennas_{activities}", f"{ii}{suffix}")
                    names_set.append(name_file)
                    with open(name_file, "wb") as fp:  # Pickling
                        pickle.dump(csi_matrices_set[ii], fp)
                
                print(f"Saved {len(names_set)} files for {list_sets_name[set_idx]} set")
                
                name_labels = os.path.join(exp_dir, f"labels_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                with open(name_labels, "wb") as fp:  # Pickling
                    pickle.dump(labels_set, fp)
                
                name_f = os.path.join(exp_dir, f"files_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(names_set, fp)
                
                name_f = os.path.join(exp_dir, f"num_windows_{list_sets_name[set_idx]}_antennas_{activities}{suffix}")
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(num_windows, fp)
                
                print(f"Finished processing {list_sets_name[set_idx]} set")