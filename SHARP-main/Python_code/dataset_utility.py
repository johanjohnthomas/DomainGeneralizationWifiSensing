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

import numpy as np
import pickle
import tensorflow as tf


def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return lab_num


def create_windows(csi_list, labels_list, sample_length, stride_length):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[1]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_matrix_stride.append(csi_i[:, ii:ii+sample_length])
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def create_windows_antennas(csi_list, labels_list, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        
        # ======== Key Change 1: Window Calculation Fix ========
        # Original code might miss final window. Now using:
        num_windows = (len_csi - sample_length) // stride_length + 1
        
        # ======== Key Change 2: Progress Tracking ========
        from tqdm import tqdm  # Add import at top if needed
        for ii in tqdm(range(0, (len_csi - sample_length) + 1, stride_length),
                      desc=f"Processing label {label_i}",
                      leave=False):
            
            # ======== Key Change 3: Dynamic Stride Adjustment ========
            # Handle edge case where remaining samples < stride_length
            if ii + sample_length > len_csi:
                if len_csi >= sample_length:  # Final valid window
                    ii = len_csi - sample_length
                else:  # Skip incomplete window
                    break
                    
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            
            # ======== Key Change 4: Shape Validation ========
            if csi_wind.shape[2] != sample_length:
                print(f"Window shape mismatch at index {ii}: {csi_wind.shape}")
                continue

            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=2, keepdims=True)
                csi_wind = csi_wind - csi_mean
                
            csi_matrix_stride.append(csi_wind)
            labels_stride.append(label_i)

        # ======== Key Change 5: Sanity Check ========
        actual_windows = len([w for w in csi_matrix_stride if w.shape[2] == sample_length])
        print(f"Label {label_i}: Calculated {num_windows} vs Actual {actual_windows} windows")
        
    return csi_matrix_stride, labels_stride


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


def load_data(csi_file_t):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi = tf.transpose(matrix_csi, perm=[2, 1, 0])
    matrix_csi = tf.cast(matrix_csi, tf.float32)
    return matrix_csi


def create_dataset(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file, prefetch=True,
                   repeat=True):
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi


def randomize_antennas(csi_data):
    stream_order = np.random.permutation(csi_data.shape[2])
    csi_data_randomized = csi_data[:, :, stream_order]
    return csi_data_randomized


def create_dataset_randomized_antennas(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file,
                                       prefetch=True, repeat=True):
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)

    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()

    randomize_funct = lambda csi_data, label: (tf.ensure_shape(tf.numpy_function(randomize_antennas, [csi_data],
                                                                                 tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(randomize_funct)

    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi


def load_data_single(csi_file_t, stream_a):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    matrix_csi_single = matrix_csi[stream_a, ...].T
    if len(matrix_csi_single.shape) < 3:
        matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
    matrix_csi_single = tf.cast(matrix_csi_single, tf.float32)
    return matrix_csi_single


def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=True):
    stream_ant = list(stream_ant)
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride, stream_ant))
    py_funct = lambda csi_file, label, stream: (tf.ensure_shape(tf.numpy_function(load_data_single,
                                                                                  [csi_file, stream],
                                                                                  tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(py_funct)
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(tf.data.AUTOTUNE)
    return dataset_csi


def balance_classes_by_undersampling(data, labels, random_seed=42):
    """
    Balance classes in a dataset by undersampling the majority classes to match the minority class count.
    
    Args:
        data: List of data samples
        labels: List of corresponding labels for each data sample
        random_seed: Random seed for reproducibility 
        
    Returns:
        balanced_data: List of data samples after balancing
        balanced_labels: List of corresponding labels after balancing
    """
    np.random.seed(random_seed)
    
    # Convert labels to numpy array if they're not already
    labels_array = np.array(labels)
    
    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique_labels, counts))}")
    
    # Find the size of the minority class
    min_count = np.min(counts)
    print(f"Minority class count: {min_count}")
    
    # Create balanced dataset by undersampling
    balanced_data = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get indices of samples with this label
        indices = np.where(labels_array == label)[0]
        
        # If we have more samples than the minority class, randomly select subset
        if len(indices) > min_count:
            # Randomly select samples
            selected_indices = np.random.choice(indices, min_count, replace=False)
        else:
            # Use all samples for minority class
            selected_indices = indices
        
        # Add selected samples to balanced dataset
        for idx in selected_indices:
            balanced_data.append(data[idx])
            balanced_labels.append(labels[idx])
    
    # Shuffle the balanced dataset
    combined = list(zip(balanced_data, balanced_labels))
    np.random.shuffle(combined)
    balanced_data, balanced_labels = zip(*combined)
    
    # Convert back to lists
    balanced_data = list(balanced_data)
    balanced_labels = list(balanced_labels)
    
    print(f"Balanced class distribution: {dict(zip(unique_labels, np.unique(balanced_labels, return_counts=True)[1]))}")
    print(f"Original dataset size: {len(data)}, Balanced dataset size: {len(balanced_data)}")
    
    return balanced_data, balanced_labels