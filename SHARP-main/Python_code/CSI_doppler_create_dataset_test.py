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
from dataset_utility import create_windows_antennas, convert_to_number
import shutil


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
                shutil.rmtree(pat)
                print(f"Removed directory: {pat}")

        path_complete = os.path.join(exp_dir, f'complete_antennas_{activities}')
        if os.path.exists(path_complete):
            remove_files = glob.glob(os.path.join(path_complete, '*'))
            for f in remove_files:
                os.remove(f)
            print(f"Cleaned directory: {path_complete}")
        else:
            os.mkdir(path_complete)
            print(f"Created directory: {path_complete}")

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

            stft_sum_1_log = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_log.T)

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
            print(f"Total matrices: {len(csi_matrices)}, Total labels: {len(labels)}")
            csi_complete = []
            for i in range(len(labels)):
                csi_complete.append(csi_matrices[i])

            window_length = args.windows_length  # number of windows considered
            stride_length = args.stride_lengths

            print(f"Creating windows with length={window_length}, stride={stride_length}")
            csi_matrices_wind, labels_wind = create_windows_antennas(csi_complete, labels, window_length, stride_length,
                                                                     remove_mean=False)

            num_windows = sum((length_i - window_length) // stride_length + 1 for length_i in lengths)

            print(f"Window length: {window_length}, Stride length: {stride_length}")
            print(f"Lengths of data: {lengths}")  # Use correct variable name
            print(f"Calculated number of windows: {num_windows}")
            print(f"Actual number of windows: {len(csi_matrices_wind)}")             
            if not len(csi_matrices_wind) == np.sum(num_windows):
                print(f'ERROR - shapes mismatch: got {len(csi_matrices_wind)}, expected {np.sum(num_windows)}')
            else:
                print(f"Created {len(csi_matrices_wind)} windows successfully")
            print("Per-sequence window counts:")
            for i, length_i in enumerate(lengths):
                seq_windows = (length_i - window_length) // stride_length + 1
                print(f"  Sequence {i} (len={length_i}): {seq_windows} windows")

            names_complete = []
            suffix = '.txt'
            for ii in range(len(csi_matrices_wind)):
                name_file = os.path.join(exp_dir, f'complete_antennas_{activities}', f'{ii}{suffix}')
                names_complete.append(name_file)
                with open(name_file, "wb") as fp:  # Pickling
                    pickle.dump(csi_matrices_wind[ii], fp)
            
            print(f"Saved {len(names_complete)} window files")
            

            name_labels = os.path.join(exp_dir, f'labels_complete_antennas_{activities}{suffix}')
            with open(name_labels, "wb") as fp:  # Pickling
                pickle.dump(labels_wind, fp)
                
            name_f = os.path.join(exp_dir, f'files_complete_antennas_{activities}{suffix}')
            with open(name_f, "wb") as fp:  # Pickling
                pickle.dump(names_complete, fp)
                
            name_f = os.path.join(exp_dir, f'num_windows_antennas_{activities}{suffix}')
            with open(name_f, "wb") as fp:  # Pickling
                pickle.dump(num_windows, fp)
                
            print(f"Finished processing and saved metadata files")