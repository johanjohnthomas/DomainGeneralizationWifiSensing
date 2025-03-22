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
from dataset_utility import create_windows_antennas, convert_to_number, convert_to_grouped_number
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
                shutil.rmtree(pat)

        path_complete = exp_dir + 'complete_antennas_' + str(activities)
        if os.path.exists(path_complete):
            remove_files = glob.glob(path_complete + '/*')
            for f in remove_files:
                os.remove(f)
        else:
            os.mkdir(path_complete)

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
            with open(name_file, "rb") as fp:  # Unpickling
                stft_sum_1 = pickle.load(fp)

            stft_sum_1_log = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_log.T)

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
            csi_complete = []
            for i in range(len(labels)):
                csi_complete.append(csi_matrices[i])

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
            
            # Filter out streams that are too short
            filtered_csi = []
            filtered_labels = []
            valid_streams = 0
            skipped_streams = 0
            
            for i, csi_matrix in enumerate(csi_complete):
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
                
                print(f"  Stream {i}: Using for testing with {num_windows} potential windows")
                filtered_csi.append(csi_matrix)
                filtered_labels.append(labels[i])
                valid_streams += 1
            
            if valid_streams == 0:
                print(f"ERROR: No valid streams available for window generation")
                print(f"Consider reducing MIN_STREAM_LENGTH or adjusting window/stride parameters")
                continue  # Skip to the next subdirectory
            
            print(f"\nStream filtering summary:")
            print(f"  Using {valid_streams} streams, skipped {skipped_streams} streams")
            
            # Generate windows from filtered streams
            csi_matrices_wind, labels_wind = create_windows_antennas(filtered_csi, filtered_labels, 
                                                                     window_length, stride_length,
                                                                     remove_mean=False)
            
            print(f"  Generated {len(csi_matrices_wind)} windows total")

            # Calculate expected number of windows
            filtered_lengths = [csi.shape[1] for csi in filtered_csi]
            expected_windows = np.floor((np.asarray(filtered_lengths)-window_length)/stride_length+1)
            if not len(csi_matrices_wind) == np.sum(expected_windows):
                print(f'WARNING - shapes mismatch: Expected {np.sum(expected_windows)} windows, got {len(csi_matrices_wind)}')

            names_complete = []
            suffix = '.txt'
            for ii in range(len(csi_matrices_wind)):
                name_file = exp_dir + 'complete_antennas_' + str(activities) + '/' + str(ii) + suffix
                names_complete.append(name_file)
                with open(name_file, "wb") as fp:  # Pickling
                    pickle.dump(csi_matrices_wind[ii], fp)
            name_labels = exp_dir + '/labels_complete_antennas_' + str(activities) + suffix
            with open(name_labels, "wb") as fp:
                # Convert numpy ints to Python ints before saving
                labels_set = [int(label) for label in labels]
                pickle.dump(labels_set, fp)
            name_f = exp_dir + '/files_complete_antennas_' + str(activities) + suffix
            with open(name_f, "wb") as fp:  # Pickling
                pickle.dump(names_complete, fp)
            name_f = exp_dir + '/num_windows_complete_antennas_' + str(activities) + suffix
            with open(name_f, "wb") as fp:  # Pickling
                pickle.dump(expected_windows, fp)
