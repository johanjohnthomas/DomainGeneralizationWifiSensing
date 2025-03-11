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
import shutil

tf.random.set_seed(42)
def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return int(lab_num)


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
        for ii in range(0, (len_csi - sample_length) + 1, stride_length):
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=2, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind)
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [int(label) for label in labels for _ in range(num_antennas)]  # Force Python ints
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
    
    print(f"Loading data from file: {csi_file}")
    if not os.path.exists(csi_file):
        raise FileNotFoundError(f"Data file not found: {csi_file}")
        
    try:
        with open(csi_file, "rb") as fp:  # Unpickling
            matrix_csi = pickle.load(fp)
        
        # Check the raw loaded data
        print(f"Raw data type: {type(matrix_csi)}")
        if isinstance(matrix_csi, np.ndarray):
            print(f"Raw data shape: {matrix_csi.shape}")
            print(f"Raw data min/max: {np.min(matrix_csi)}/{np.max(matrix_csi)}")
            print(f"Is data all zeros? {np.all(matrix_csi == 0)}")
        else:
            print(f"Raw data is not numpy array: {type(matrix_csi)}")
        
        # Check the indexed data
        print(f"Stream/antenna index: {stream_a}")
        matrix_csi_single = matrix_csi[stream_a, ...].T
        
        print(f"Indexed data shape: {matrix_csi_single.shape}")
        print(f"Indexed data min/max: {np.min(matrix_csi_single)}/{np.max(matrix_csi_single)}")
        print(f"Is indexed data all zeros? {np.all(matrix_csi_single == 0)}")
        
        if len(matrix_csi_single.shape) < 3:
            matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)
        
        # Verify final data
        print(f"Final data shape: {matrix_csi_single.shape}")
        print(f"Final data min/max: {np.min(matrix_csi_single)}/{np.max(matrix_csi_single)}")
        
        # Raise an error if all zeros
        if np.all(matrix_csi_single == 0):
            raise ValueError("Data contains all zeros after processing - check your data source")
            
        matrix_csi_single = tf.cast(matrix_csi_single, tf.float32)
        return matrix_csi_single
    except Exception as e:
        print(f"Error during data loading/processing: {e}")
        raise

def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=False):
    if len(csi_matrix_files) == 0:
        print("Error: Empty dataset - no files to process!")
        raise ValueError("Cannot create dataset with empty file list - please check your data paths and make sure files exist")
        
    print(f"Creating dataset with {len(csi_matrix_files)} samples")
    dataset_csi = tf.data.Dataset.from_tensor_slices((csi_matrix_files, labels_stride, stream_ant))
    
    with tf.device('/cpu:0'):
        # Clear existing cache if present
        if cache_file and os.path.exists(cache_file):
            try:
                if os.path.isfile(cache_file):
                    os.remove(cache_file)
                elif os.path.isdir(cache_file):
                    shutil.rmtree(cache_file)
            except Exception as e:
                print(f"Error clearing cache: {e}")

        # Define the map function with proper error handling
        def safe_load_data(csi_file, label, stream):
            try:
                # Print more information about the tensors
                file_path = csi_file.numpy().decode() if hasattr(csi_file, 'numpy') else str(csi_file)
                stream_value = stream.numpy() if hasattr(stream, 'numpy') else stream
                label_value = label.numpy() if hasattr(label, 'numpy') else label
                
                print(f"Processing file: {file_path}")
                print(f"Stream/antenna index: {stream_value}, Label: {label_value}")
                
                # Call the data loading function with explicit decoding of string tensor
                data = tf.numpy_function(
                    func=load_data_single,
                    inp=[csi_file, stream],
                    Tout=tf.float32
                )
                
                # Verify the returned data
                print(f"Returned data shape: {data.shape}")
                
                # Check for all zeros as a heuristic for dummy data
                if tf.reduce_all(tf.equal(data, 0)):
                    print("WARNING: Data contains all zeros!")
                    raise ValueError("Loaded data contains all zeros - potential dummy data detected")
                    
                return data, tf.squeeze(label)
            except Exception as e:
                print(f"Error in safe_load_data: {e}")
                raise

        # Apply mapping
        dataset_csi = dataset_csi.map(
            safe_load_data,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache after mapping
        dataset_csi = dataset_csi.cache(cache_file)
        
        # Shuffle if needed
        if shuffle:
            dataset_csi = dataset_csi.shuffle(buffer_size=max(100, len(csi_matrix_files)))
        
        # Batch the data
        dataset_csi = dataset_csi.batch(batch_size)
        
        # Repeat if needed
        if repeat:
            dataset_csi = dataset_csi.repeat()
            
        # Prefetch for performance
        if prefetch:
            dataset_csi = dataset_csi.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset_csi