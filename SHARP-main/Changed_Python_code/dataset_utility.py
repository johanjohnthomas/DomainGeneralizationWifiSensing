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
import math
import hashlib
import time
from datetime import datetime

tf.random.set_seed(42)
def convert_to_number(lab, csi_label_dict):
    """
    Maps a label to its corresponding index in the csi_label_dict.
    In the original implementation, each label (E, E1, E2, etc.) has its own unique index.
    """
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return int(lab_num)

def convert_to_grouped_number(lab, csi_label_dict):
    """
    Groups activity labels by their base letter.
    For example, 'E', 'E1', 'E2' will all map to the same index (the index of 'E').
    If the base letter doesn't exist in csi_label_dict, it falls back to the original mapping.
    
    Args:
        lab: The activity label to convert (e.g., 'E1', 'J2')
        csi_label_dict: List of all possible activity labels
    
    Returns:
        int: The index corresponding to the base letter of the activity
    """
    if not lab:  # Handle empty labels
        return 0
    
    base_letter = lab[0].upper()  # Ensure case-insensitive
    
    # Look for the base letter in the label dictionary (case-insensitive)
    base_indices = np.where(np.array([label.upper().startswith(base_letter) for label in csi_label_dict]))[0]
    
    if len(base_indices) > 0:
        # Return the first occurrence of the base letter or a label starting with it
        return int(base_indices[0])
    else:
        # Fallback to original mapping if base letter not found
        return convert_to_number(lab, csi_label_dict)

def get_label_mappings(csi_label_dict):
    """
    Returns mappings from original labels to both original and grouped indices.
    
    Args:
        csi_label_dict: List of all possible activity labels
    
    Returns:
        tuple: (original_mapping, grouped_mapping) where each is a dict mapping labels to indices
    """
    original_mapping = {label: convert_to_number(label, csi_label_dict) for label in csi_label_dict}
    grouped_mapping = {label: convert_to_grouped_number(label, csi_label_dict) for label in csi_label_dict}
    
    return original_mapping, grouped_mapping

def create_windows(csi_list, labels_list, sample_length, stride_length):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[1]
        for ii in range(0, len_csi - sample_length + 1, stride_length):
            csi_matrix_stride.append(csi_i[:, ii:ii+sample_length])
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def create_windows_antennas(csi_list, labels_list, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    labels_stride = []
    
    REQUIRED_FEATURES = 100  # Updated to match standardized feature dimension
    
    for i in range(len(labels_list)):
        # Expected shape: (antennas, time, features)
        csi_i = csi_list[i]
        label_i = labels_list[i]
        
        # Validate dimensions
        if csi_i.ndim != 3:
            print(f"Invalid CSI matrix dims {csi_i.shape} in sample {i} - skipping")
            continue
            
        antennas, time_steps, features = csi_i.shape
        
        if features != REQUIRED_FEATURES:
            if features > REQUIRED_FEATURES:
                print(f"WARNING: Sample {i} has {features} features, which exceeds required {REQUIRED_FEATURES}")
                print(f"         Trimming to first {REQUIRED_FEATURES} features")
                csi_i = csi_i[:, :, :REQUIRED_FEATURES]
            else:
                print(f"Padding features from {features} to {REQUIRED_FEATURES} in sample {i}")
                pad_width = ((0,0), (0,0), (0, REQUIRED_FEATURES-features))
                csi_i = np.pad(csi_i, pad_width, mode='constant')
        
        len_time = csi_i.shape[1]  # Time is axis 1
        
        for ii in range(0, len_time - sample_length + 1, stride_length):
            # Slice along time axis -> (antennas, window_length, features)
            window = csi_i[:, ii:ii + sample_length, :]
            if remove_mean:
                window = window - np.mean(window, axis=1, keepdims=True)
            csi_matrix_stride.append(window)
            labels_stride.append(label_i)
            
    return csi_matrix_stride, labels_stride


def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [int(label) for label in labels for _ in range(num_antennas)]  # Force Python ints
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant


# Added new function based on original researcher's implementation
def expand_antennas_matrix(data_matrix):
    """
    Convert (400, 242) matrix to 4× (100, 242, 1) samples.
    
    Args:
        data_matrix: Input matrix with shape (400, 242) containing data from 4 antennas
                     concatenated along the time dimension
    
    Returns:
        List of 4 matrices, each with shape (100, 242, 1) representing each antenna's data
    """
    if len(data_matrix.shape) != 2:
        raise ValueError(f"Expected 2D input matrix, got shape {data_matrix.shape}")
    
    if data_matrix.shape[0] % 4 != 0:
        raise ValueError(f"Input matrix first dimension must be divisible by 4, got {data_matrix.shape[0]}")
    
    time_per_antenna = data_matrix.shape[0] // 4
    return [data_matrix[i*time_per_antenna:(i+1)*time_per_antenna, :, np.newaxis] 
            for i in range(4)]


def load_data(csi_file_t):
    # Ensure csi_file is a string, not bytes
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file_t.decode('utf-8')
    else:
        csi_file = str(csi_file_t)
    
    # Check if file exists before attempting to load
    if not os.path.exists(csi_file):
        print(f"Error loading file {csi_file}: [Errno 2] No such file or directory: '{csi_file}'")
        # Create a placeholder array with the expected shape instead of crashing
        # This allows the pipeline to continue with empty data
        placeholder = np.zeros((1, 1, 1), dtype=np.float32)
        placeholder = tf.transpose(placeholder, perm=[2, 1, 0])
        placeholder = tf.cast(placeholder, tf.float32)
        return placeholder
    
    # Check if file is a numpy .npy file
    if csi_file.endswith('.npy'):
        try:
            # Ensure the file path is a string for np.load
            matrix_csi = np.load(csi_file)
        except Exception as e:
            print(f"Error loading numpy file {csi_file}: {e}")
            # Try a fallback approach if standard loading fails
            try:
                # Some .npy files might be saved with pickle protocol
                with open(csi_file, 'rb') as f:
                    matrix_csi = pickle.load(f)
            except Exception as inner_e:
                print(f"Fallback loading also failed: {inner_e}")
                # Return a placeholder to avoid crashing the pipeline
                print(f"Returning empty placeholder for {csi_file}")
                matrix_csi = np.zeros((4, 100, 100), dtype=np.float32)
    else:
        # Original pickle loading
        try:
            with open(csi_file, "rb") as fp:
                matrix_csi = pickle.load(fp)
        except Exception as e:
            print(f"Error loading pickle file {csi_file}: {e}")
            # Return a placeholder to avoid crashing the pipeline
            print(f"Returning empty placeholder for {csi_file}")
            matrix_csi = np.zeros((4, 100, 100), dtype=np.float32)
    
    matrix_csi = tf.transpose(matrix_csi, perm=[2, 1, 0])
    matrix_csi = tf.cast(matrix_csi, tf.float32)
    return matrix_csi


def generate_dynamic_cache_filename(cache_base, csi_matrix_files, cache_version=None):
    """
    Generate a unique cache filename based on dataset properties and timestamps.
    
    Args:
        cache_base: Base name for the cache file
        csi_matrix_files: List of CSI matrix files that are being processed
        cache_version: Optional version identifier for the cache
        
    Returns:
        A unique cache filename that will change if the dataset changes
    """
    # Create a hash based on the dataset file list
    if isinstance(csi_matrix_files, (list, tuple, np.ndarray)):
        # Sort to ensure the same files always produce the same hash regardless of order
        files_str = ''.join(sorted([os.path.basename(f) for f in csi_matrix_files]))
        files_hash = hashlib.md5(files_str.encode()).hexdigest()[:10]
    else:
        # For non-list inputs, use a hash of the string representation
        files_hash = hashlib.md5(str(csi_matrix_files).encode()).hexdigest()[:10]
    
    # Add dataset size information
    dataset_size = len(csi_matrix_files) if isinstance(csi_matrix_files, (list, tuple, np.ndarray)) else 0
    
    # Add timestamp information (can be commented out if you want only content-based hashing)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Combine components to form the cache filename
    if cache_version:
        cache_filename = f"{cache_base}_v{cache_version}_{dataset_size}_{files_hash}_{timestamp}"
    else:
        cache_filename = f"{cache_base}_{dataset_size}_{files_hash}_{timestamp}"
    
    return cache_filename


def create_dataset(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file, prefetch=True,
                   repeat=True):
    # Verify all files exist first and filter out missing ones
    valid_files = []
    valid_labels = []
    
    for i, file_path in enumerate(csi_matrix_files):
        if os.path.exists(file_path):
            valid_files.append(file_path)
            valid_labels.append(labels_stride[i])
        else:
            print(f"Warning: Skipping missing file {file_path}")
    
    if len(valid_files) == 0:
        print("Error: No valid files found. Cannot create dataset.")
        # Return an empty dataset with the correct structure
        dummy_data = np.zeros((1,) + input_shape, dtype=np.float32)
        dummy_labels = np.zeros(1, dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((dummy_data, dummy_labels)).batch(batch_size)
    
    print(f"Creating dataset with {len(valid_files)} valid files out of {len(csi_matrix_files)} total files")
    
    # Use dynamic cache filename if cache_file is provided
    if cache_file:
        cache_file = generate_dynamic_cache_filename(cache_file, valid_files)
    
    dataset_csi = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetch earlier in pipeline
    dataset_csi = dataset_csi.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    dataset_csi = dataset_csi.cache(cache_file)
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(valid_labels))
    if repeat:
        dataset_csi = dataset_csi.repeat()
    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    
    # Keep the final prefetch if requested
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_csi


def randomize_antennas(csi_data):
    stream_order = np.random.permutation(csi_data.shape[2])
    csi_data_randomized = csi_data[:, :, stream_order]
    return csi_data_randomized


def create_dataset_randomized_antennas(csi_matrix_files, labels_stride, input_shape, batch_size, shuffle, cache_file,
                                       prefetch=True, repeat=True):
    # Verify all files exist first and filter out missing ones
    valid_files = []
    valid_labels = []
    
    for i, file_path in enumerate(csi_matrix_files):
        if os.path.exists(file_path):
            valid_files.append(file_path)
            valid_labels.append(labels_stride[i])
        else:
            print(f"Warning: Skipping missing file {file_path}")
    
    if len(valid_files) == 0:
        print("Error: No valid files found. Cannot create dataset.")
        # Return an empty dataset with the correct structure
        dummy_data = np.zeros((1,) + input_shape, dtype=np.float32)
        dummy_labels = np.zeros(1, dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((dummy_data, dummy_labels)).batch(batch_size)
    
    print(f"Creating randomized dataset with {len(valid_files)} valid files out of {len(csi_matrix_files)} total files")
    
    # Use dynamic cache filename if cache_file is provided
    if cache_file:
        cache_file = generate_dynamic_cache_filename(cache_file, valid_files)
    
    dataset_csi = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
    py_funct = lambda csi_file, label: (tf.ensure_shape(tf.numpy_function(load_data, [csi_file], tf.float32),
                                                        input_shape), label)
    dataset_csi = dataset_csi.map(py_funct, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetch earlier in pipeline
    dataset_csi = dataset_csi.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    dataset_csi = dataset_csi.cache(cache_file)

    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(valid_labels))
    if repeat:
        dataset_csi = dataset_csi.repeat()

    randomize_funct = lambda csi_data, label: (tf.ensure_shape(tf.numpy_function(randomize_antennas, [csi_data],
                                                                                 tf.float32), input_shape), label)
    dataset_csi = dataset_csi.map(randomize_funct, num_parallel_calls=tf.data.AUTOTUNE)

    dataset_csi = dataset_csi.batch(batch_size=batch_size)
    
    if prefetch:
        dataset_csi = dataset_csi.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_csi


def load_data_single(csi_file_t, stream_a):
    """
    Load data from a single file - Note: this function is being kept for backward compatibility
    but may not be used in the new multi-channel approach.
    """
    # Ensure csi_file is a string, not bytes
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file_t.decode('utf-8')
    else:
        csi_file = str(csi_file_t)
    
    print(f"Loading data from file: {csi_file}")
    if not os.path.exists(csi_file):
        print(f"Data file not found: {csi_file}")
        # Return a placeholder to avoid crashing the pipeline
        placeholder = np.zeros((100, 100, 4), dtype=np.float32)  # (time, features, antennas)
        return tf.cast(placeholder, tf.float32)
        
    try:
        # Check if file is a numpy .npy file
        if csi_file.endswith('.npy'):
            try:
                matrix_csi = np.load(csi_file)
            except Exception as e:
                print(f"Error loading numpy file {csi_file}: {e}")
                # Try a fallback approach if standard loading fails
                try:
                    # Some .npy files might be saved with pickle protocol
                    with open(csi_file, 'rb') as f:
                        matrix_csi = pickle.load(f)
                except Exception as inner_e:
                    print(f"Fallback loading also failed: {inner_e}")
                    # Return a placeholder to avoid crashing the pipeline
                    print(f"Returning empty placeholder for {csi_file}")
                    placeholder = np.zeros((4, 100, 100), dtype=np.float32)  # (antennas, time, features)
                    matrix_csi = placeholder
        else:
            # Original pickle loading
            try:
                with open(csi_file, "rb") as fp:
                    matrix_csi = pickle.load(fp)
            except Exception as e:
                print(f"Error loading pickle file {csi_file}: {e}")
                # Return a placeholder to avoid crashing the pipeline
                print(f"Returning empty placeholder for {csi_file}")
                placeholder = np.zeros((4, 100, 100), dtype=np.float32)  # (antennas, time, features)
                matrix_csi = placeholder
        
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
        
        # STEP 1: Extract the data for this antenna/stream
        print("DATA TRANSFORMATION PROCESS:")
        print(f"  STEP 1: Extract single antenna data (index {stream_a})")
        try:
            # Safely access the antenna index
            stream_a_idx = int(stream_a) % matrix_csi.shape[0]  # Ensure index is in bounds
            matrix_csi_single = matrix_csi[stream_a_idx, ...]
        except Exception as e:
            print(f"Error extracting antenna data: {e}")
            # Return a placeholder with the expected dimensions
            matrix_csi_single = np.zeros((100, 100), dtype=np.float32)
        
        print(f"  - Single antenna data shape: {matrix_csi_single.shape}")
        
        # STEP 2: Transpose the data to match expected dimensions
        #print(f"  STEP 2: Transpose data from (100, 340) to (340, 100)")
        #matrix_csi_single = np.transpose(matrix_csi_single)  # Explicitly use numpy transpose for clarity
        #rint(f"  - After transpose shape: {matrix_csi_single.shape}")
        
        # STEP 3: Add channel dimension if needed
        print(f"  STEP 3: Add channel dimension")
        if len(matrix_csi_single.shape) < 3:
            matrix_csi_single = np.expand_dims(matrix_csi_single, axis=-1)  # Shape (340, 100, 1)
            print(f"  - Final data shape with channel: {matrix_csi_single.shape}")
        
        # Verify final data
        print(f"FINAL data shape: {matrix_csi_single.shape}")
        print(f"FINAL data min/max: {np.min(matrix_csi_single)}/{np.max(matrix_csi_single)}")
        
        matrix_csi_single = tf.cast(matrix_csi_single, tf.float32)
        return matrix_csi_single
    except Exception as e:
        print(f"Unhandled error during data loading/processing: {e}")
        # Return a placeholder with the expected dimensions as a last resort
        placeholder = np.zeros((100, 100, 4), dtype=np.float32)  # (time, features, antennas)
        return tf.cast(placeholder, tf.float32)


def load_data_multi_channel(csi_file_t):
    """
    Load data from a file and process all antennas as channels.
    This is the preferred approach for processing CSI data.
    """
    # Ensure csi_file is a string, not bytes
    if isinstance(csi_file_t, (bytes, bytearray)):
        csi_file = csi_file_t.decode('utf-8')
    else:
        csi_file = str(csi_file_t)
    
    print(f"Loading data from file: {csi_file}")
    if not os.path.exists(csi_file):
        print(f"Data file not found: {csi_file}")
        # Return a placeholder to avoid crashing the pipeline
        placeholder = np.zeros((100, 100, 4), dtype=np.float32)  # (time, features, antennas)
        return tf.cast(placeholder, tf.float32)
        
    try:
        # Check if file is a numpy .npy file
        if csi_file.endswith('.npy'):
            try:
                matrix_csi = np.load(csi_file)
            except Exception as e:
                print(f"Error loading numpy file {csi_file}: {e}")
                # Try a fallback approach if standard loading fails
                try:
                    # Some .npy files might be saved with pickle protocol
                    with open(csi_file, 'rb') as f:
                        matrix_csi = pickle.load(f)
                except Exception as inner_e:
                    print(f"Fallback loading also failed: {inner_e}")
                    # Return a placeholder to avoid crashing the pipeline
                    print(f"Returning empty placeholder for {csi_file}")
                    # Create a placeholder with expected dimensions
                    matrix_csi = np.zeros((4, 100, 100), dtype=np.float32)
        else:
            # Original pickle loading
            try:
                with open(csi_file, "rb") as fp:
                    matrix_csi = pickle.load(fp)
            except Exception as e:
                print(f"Error loading pickle file {csi_file}: {e}")
                # Return a placeholder to avoid crashing the pipeline
                print(f"Returning empty placeholder for {csi_file}")
                matrix_csi = np.zeros((4, 100, 100), dtype=np.float32)
        
        # Check the raw loaded data
        print(f"Raw data type: {type(matrix_csi)}")
        if isinstance(matrix_csi, np.ndarray):
            print(f"Raw data shape: {matrix_csi.shape}")
            print(f"Raw data min/max: {np.min(matrix_csi)}/{np.max(matrix_csi)}")
            print(f"Is data all zeros? {np.all(matrix_csi == 0)}")
        else:
            print(f"Raw data is not numpy array: {type(matrix_csi)}")
            # Convert to numpy array if possible
            try:
                matrix_csi = np.array(matrix_csi, dtype=np.float32)
            except:
                matrix_csi = np.zeros((4, 100, 100), dtype=np.float32)
        
        # Transpose to (time, features, antennas)
        print("MULTI-CHANNEL DATA TRANSFORMATION:")
        print(f"  STEP 1: Transpose data from (antennas, time, features) to (time, features, antennas)")
        try:
            matrix_csi_multi = np.transpose(matrix_csi, (1, 2, 0))  # (time, features, antennas)
            print(f"  - After transpose shape: {matrix_csi_multi.shape}")
            
            # Standardize to 100 features by slicing or padding
            if matrix_csi_multi.shape[1] > 100:
                print(f"  - Feature dimension too large ({matrix_csi_multi.shape[1]}), truncating to 100")
                matrix_csi_multi = matrix_csi_multi[:, :100, :]
            elif matrix_csi_multi.shape[1] < 100:
                print(f"  - Feature dimension too small ({matrix_csi_multi.shape[1]}), padding to 100")
                pad_width = ((0,0), (0, 100-matrix_csi_multi.shape[1]), (0,0))
                matrix_csi_multi = np.pad(matrix_csi_multi, pad_width, mode='constant')
            print(f"  - After standardizing features shape: {matrix_csi_multi.shape}")
            
        except Exception as e:
            print(f"Error during transpose or standardization: {e}")
            matrix_csi_multi = np.zeros((matrix_csi.shape[1], 100, matrix_csi.shape[0]), dtype=np.float32)
            
        # Normalize across time and features (not antennas)
        print(f"  STEP 2: Mean and standard deviation normalization")
        try:
            mean = np.mean(matrix_csi_multi, axis=(0, 1), keepdims=True)  # Global mean across time and features
            std = np.std(matrix_csi_multi, axis=(0, 1), keepdims=True)
            matrix_csi_multi = (matrix_csi_multi - mean) / (std + 1e-9)
            print(f"  - After normalization shape: {matrix_csi_multi.shape}")
            print(f"  - After normalization min/max: {np.min(matrix_csi_multi)}/{np.max(matrix_csi_multi)}")
        except Exception as e:
            print(f"Error during normalization: {e}")
            # If normalization fails, just use the original or placeholder data
            if not isinstance(matrix_csi_multi, np.ndarray) or matrix_csi_multi.size == 0:
                matrix_csi_multi = np.zeros((100, 100, 4), dtype=np.float32)  # (time, features, antennas)
        
        # Verify final data
        print(f"FINAL multi-channel data shape: {matrix_csi_multi.shape}")
        print(f"FINAL data min/max: {np.min(matrix_csi_multi)}/{np.max(matrix_csi_multi)}")
        
        return tf.cast(matrix_csi_multi, tf.float32)
    except Exception as e:
        print(f"Unhandled error during data loading/processing: {e}")
        # Return a placeholder with the expected dimensions as a last resort
        placeholder = np.zeros((100, 100, 4), dtype=np.float32)  # (time, features, antennas)
        return tf.cast(placeholder, tf.float32)

def create_dataset_single(csi_matrix_files, labels_stride, stream_ant, input_shape, batch_size, shuffle, cache_file,
                          prefetch=True, repeat=False):
    # Verify all files exist first and filter out missing ones
    valid_files = []
    valid_labels = []
    valid_streams = []
    
    for i, file_path in enumerate(csi_matrix_files):
        if os.path.exists(file_path):
            valid_files.append(file_path)
            valid_labels.append(labels_stride[i])
            valid_streams.append(stream_ant[i])
        else:
            print(f"Warning: Skipping missing file {file_path}")
    
    if len(valid_files) == 0:
        print("Error: No valid files found. Cannot create dataset.")
        # Return an empty dataset with the correct structure
        dummy_data = np.zeros((1,) + input_shape, dtype=np.float32)
        dummy_labels = np.zeros(1, dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((dummy_data, dummy_labels)).batch(batch_size)
    
    print(f"Creating single dataset with {len(valid_files)} valid files out of {len(csi_matrix_files)} total files")
    
    # Use dynamic cache filename if cache_file is provided
    if cache_file:
        cache_file = generate_dynamic_cache_filename(cache_file, valid_files)
    
    # Create a dataset from the file paths, labels, and antenna streams
    dataset_csi = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels, valid_streams))
    
    # Define a safe load function that handles errors gracefully
    def safe_load_data(csi_file, label, stream):
        try:
            data = load_data_single(csi_file, stream)
            # Ensure data has the correct shape
            if data.shape != input_shape:
                print(f"Warning: Data shape mismatch in {csi_file}. Expected {input_shape}, got {data.shape}")
                # Create empty placeholder with correct shape
                data = tf.zeros(input_shape, dtype=tf.float32)
            return data, label
        except Exception as e:
            print(f"Error loading file {csi_file} for stream {stream}: {e}")
            return tf.zeros(input_shape, dtype=tf.float32), label
    
    # Use a Python function to load the data and get the specified stream
    py_funct = lambda csi_file, label, stream: (
        tf.ensure_shape(
            tf.numpy_function(
                lambda file, stream: load_data_single(file, stream),
                [csi_file, stream],
                tf.float32
            ),
            input_shape
        ),
        label
    )
    
    # Map the function over the dataset
    dataset_csi = dataset_csi.map(py_funct, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache the dataset if a cache file is provided
    dataset_csi = dataset_csi.cache(cache_file)
    
    # Shuffle if specified
    if shuffle:
        dataset_csi = dataset_csi.shuffle(len(valid_labels))
    
    # Repeat if specified
    if repeat:
        dataset_csi = dataset_csi.repeat()
    
    # Batch the dataset
    dataset_csi = dataset_csi.batch(batch_size)
    
    # Prefetch for better performance
    if prefetch:
        dataset_csi = dataset_csi.prefetch(tf.data.AUTOTUNE)
    
    return dataset_csi

def create_dataset_multi_channel(csi_matrix_files, labels, input_shape, batch_size, shuffle, cache_file, buffer_size=100):
    # Verify all files exist first and filter out missing ones
    valid_files = []
    valid_labels = []
    
    for i, file_path in enumerate(csi_matrix_files):
        if os.path.exists(file_path):
            valid_files.append(file_path)
            valid_labels.append(labels[i])
        else:
            print(f"Warning: Skipping missing file {file_path}")
    
    if len(valid_files) == 0:
        print("Error: No valid files found. Cannot create dataset.")
        # Return an empty dataset with the correct structure
        dummy_data = np.zeros((1,) + input_shape, dtype=np.float32)
        dummy_labels = np.zeros(1, dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((dummy_data, dummy_labels)).batch(batch_size)
    
    print(f"Creating multi-channel dataset with {len(valid_files)} valid files out of {len(csi_matrix_files)} total files")
    
    # Use dynamic cache filename if cache_file is provided
    if cache_file:
        cache_file = generate_dynamic_cache_filename(cache_file, valid_files)
    
    # Define a function to load and process the data
    def load_and_process_file(file_path, label):
        def _parse_function(file_path, label):
            try:
                # Load the multi-channel data
                data = load_data_multi_channel(file_path)
                if data.shape != input_shape:
                    print(f"Warning: Data shape mismatch in {file_path}. Expected {input_shape}, got {data.shape}")
                    data = tf.zeros(input_shape, dtype=tf.float32)
                return data, label
            except Exception as e:
                print(f"Error loading multi-channel file {file_path}: {e}")
                return tf.zeros(input_shape, dtype=tf.float32), label
        
        return tf.py_function(_parse_function, [file_path, label], [tf.float32, tf.int32])

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
    
    # Map the loading function
    dataset = dataset.map(load_and_process_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache, shuffle, batch, prefetch
    dataset = dataset.cache(cache_file)
    if shuffle:
        dataset = dataset.shuffle(len(valid_labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset