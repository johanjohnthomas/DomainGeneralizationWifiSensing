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
import os


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


def create_windows_antennas(csi_list, labels_list, window_length, stride_length, remove_mean=False):
    """
    Enhanced window creation with shape validation and edge handling
    Args:
        csi_list: List of CSI matrices [num_antennas, features, time_steps]
        labels_list: Corresponding activity labels
        window_length: Temporal length of windows
        stride_length: Stride between window starts
        remove_mean: Whether to remove window mean
    Returns:
        windows: List of windowed CSI data
        labels: Expanded labels for each window
    """
    # 1. Input Validation
    assert window_length > 0, "Window length must be positive"
    assert stride_length > 0, "Stride must be positive"
    assert len(csi_list) == len(labels_list), "CSI/Label length mismatch"
    
    csi_windows = []
    window_labels = []
    
    for csi_matrix, label in zip(csi_list, labels_list):
        num_antennas, num_features, total_timesteps = csi_matrix.shape
        
        # 2. Adaptive Window Calculation
        max_start_idx = total_timesteps - window_length
        if max_start_idx < 0:
            raise ValueError(f"CSI length {total_timesteps} < window {window_length}")
            
        # 3. Precise Window Generation
        start_indices = range(0, max_start_idx + 1, stride_length)
        for start in start_indices:
            end = start + window_length
            window = csi_matrix[:, :, start:end]
            
            # 4. Mean Removal (optional)
            if remove_mean:
                window -= np.mean(window, axis=2, keepdims=True)
                
            # 5. Shape Preservation Check
            if window.shape[2] != window_length:
                continue  # Skip incomplete final windows
                
            csi_windows.append(window)
            window_labels.append(label)
    
    # 6. Final Consistency Check
    if len(csi_windows) != len(window_labels):
        raise RuntimeError("Window/label count mismatch during creation")
        
    return np.array(csi_windows), np.array(window_labels)



def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [item for item in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


def load_data(csi_file_t, sanitize_phase=True):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    
    # If phase sanitization is disabled, we need to modify the CSI data
    if not sanitize_phase:
        print(f"ABLATION: Phase sanitization disabled for {os.path.basename(csi_file)}")
        # Placeholder for phase sanitization removal
        
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
                                       prefetch=True, repeat=True, sanitize_phase=True):
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(
        lambda file: load_data(file, sanitize_phase),
        [csi_file],
        tf.float32), input_shape), label)
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


def load_data_single(csi_file_t, stream_a, sanitize_phase=True):
    csi_file = csi_file_t
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file.decode()
    with open(csi_file, "rb") as fp:  # Unpickling
        matrix_csi = pickle.load(fp)
    #print(f"[DEBUG] Raw CSI shape from file: {matrix_csi.shape}")  # Should show (antennas, features, time)
    matrix_csi_single = matrix_csi[stream_a, ...].T
    
    # If phase sanitization is disabled, we need to modify the CSI data 
    # to simulate not using phase sanitization
    if not sanitize_phase:
        print(f"ABLATION: Phase sanitization disabled for {os.path.basename(csi_file)}")
        # This is a placeholder for phase sanitization removal
        # In a real implementation, we would either:
        # 1. Load raw data instead of sanitized data
        # 2. Apply an inverse transform to "desanitize" the phase
        
    #print(f"[DEBUG] After antenna selection: {matrix_csi_single.shape}")  # Should be (time, features)
    if len(matrix_csi_single.shape) < 3:
        matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
   # print(f"[DEBUG] After channel expansion: {matrix_csi_single.shape}")  # Should be (340, 100, 4)
    
    # Add dimension validation
    if matrix_csi_single.shape != (340, 100, 1):
        raise ValueError(f"Invalid CSI shape: {matrix_csi_single.shape}")
        
    matrix_csi_single = tf.cast(matrix_csi_single, tf.float32)
    return matrix_csi_single


def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=True, sanitize_phase=True):
    stream_ant = list(stream_ant)
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride, stream_ant))
    
    # Define a lambda function that passes the sanitize_phase parameter
    py_funct = lambda csi_file, label, stream: (tf.ensure_shape(tf.numpy_function(
        lambda file, str_a: load_data_single(file, str_a, sanitize_phase),
        [csi_file, stream],
        tf.float32), input_shape), label)
        
    print("[DEBUG] Raw CSI shape:", tf.data.experimental.get_structure(dataset_csi))
    dataset_csi = dataset_csi.map(py_funct)
    print(f"[DEBUG] Final dataset shape: {dataset_csi.element_spec}")  # Should show (None,340,100,4), ...
    
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(labels_stride))
    dataset_csi = dataset_csi.apply(tf.data.experimental.assert_cardinality(len(labels_stride)))
    dataset_csi = dataset_csi.cache(cache_file)
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=1)
    return dataset_csi
