#!/usr/bin/env python
"""
Script to fix dataset splitting issues across all subdirectories.
This applies the modified train/val/test split approach to all activity datasets.
"""

import os
import sys
import glob
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Configuration - modify these to match your setup
base_dir = "./SHARP-main/"
code_dir = os.path.join(base_dir, "Python_code/")
data_dir = os.path.join(base_dir, "Python_code/doppler_traces/")
activities = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"

# Import necessary functions from your codebase
sys.path.append(code_dir)
try:
    from dataset_utility import create_windows_antennas
except ImportError:
    print("Error importing create_windows_antennas. Make sure dataset_utility.py is in the correct location.")
    sys.exit(1)

def get_all_subdirs():
    """Get all subdirectories in the data directory."""
    all_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return all_subdirs

def load_raw_data(subdir):
    """Load the raw data files for processing."""
    print(f"\n=== Loading Raw Data for {subdir} ===")
    
    exp_dir = os.path.join(data_dir, subdir) + '/'
    
    # Find all stream files
    all_files = glob.glob(os.path.join(exp_dir, "*_stream_*.txt"))
    print(f"Found {len(all_files)} raw data files")
    
    if len(all_files) == 0:
        print("No data files found. Check the directory path.")
        return None, None
    
    # Extract names without extension
    names = [os.path.basename(f)[:-4] for f in all_files]
    names.sort()
    
    # Count streams
    stream_counts = {}
    for name in names:
        if '_stream_' in name:
            stream_num = name.split('_stream_')[1]
            stream_counts[stream_num] = stream_counts.get(stream_num, 0) + 1
    
    print(f"Found {len(stream_counts)} streams")
    for stream, count in sorted(stream_counts.items()):
        print(f"  Stream {stream}: {count} files")
    
    # Get the label from the subdir
    label = subdir.split("_")[-1]
    print(f"Using label: {label}")
    
    return exp_dir, label

def fix_dataset_split(exp_dir, label):
    """Apply a fixed dataset splitting strategy."""
    print("\n=== Applying Fixed Dataset Split ===")
    
    activities_list = activities.split(",")
    if label not in activities_list:
        print(f"Warning: Label '{label}' is not in the activities list. This may cause problems.")
    
    # Get CSI matrix files - these should be the raw data before any windowing
    names = []
    all_files = os.listdir(exp_dir)
    for filename in all_files:
        if filename.endswith('.txt') and not filename.startswith('.') and '_stream_' in filename:
            names.append(filename[:-4])
    names.sort()
    
    print(f"Found {len(names)} raw data files for processing")
    
    # Group files by their prefix (before _stream_)
    prefixes = set()
    for name in names:
        if '_stream_' in name:
            prefix = name.split('_stream_')[0]
            prefixes.add(prefix)
    
    print(f"Found {len(prefixes)} unique recording sessions/groups")
    
    # Determine n_tot (number of streams/antennas)
    stream_nums = set()
    for name in names:
        if '_stream_' in name:
            stream_num = name.split('_stream_')[1]
            stream_nums.add(stream_num)
    
    n_tot = len(stream_nums)
    print(f"Detected {n_tot} streams/antennas")
    
    # Load and organize the data
    csi_matrices = []
    labels = []
    errors = 0
    
    # Process files in batches of n_tot (all antennas for a single sample)
    current_batch = []
    for i, name in enumerate(names):
        name_file = os.path.join(exp_dir, name + '.txt')
        try:
            with open(name_file, "rb") as fp:
                stft_sum = pickle.load(fp)
                
                # Convert to numpy array if it's a list
                if isinstance(stft_sum, list):
                    stft_sum = np.array(stft_sum, dtype=np.float32)
                
                # Validate array type
                if stft_sum.dtype != np.float32 and stft_sum.dtype != np.float64:
                    stft_sum = stft_sum.astype(np.float32)
                
                # Subtract mean
                stft_sum_mean = stft_sum - np.mean(stft_sum, axis=0, keepdims=True)
                
                current_batch.append(stft_sum_mean.T)
                
                # When we have collected data for all antennas
                if len(current_batch) == n_tot:
                    # Check that all files in the batch have the same length
                    lengths = [data.shape[1] for data in current_batch]
                    if all(l == lengths[0] for l in lengths):
                        csi_matrices.append(np.array(current_batch))
                        labels.append(label_to_num(label, activities_list))
                    else:
                        print(f"Warning: Mismatched lengths in batch: {lengths}")
                        errors += 1
                    
                    # Reset for next batch
                    current_batch = []
                
        except Exception as e:
            print(f"Error loading {name_file}: {e}")
            errors += 1
    
    print(f"Loaded {len(csi_matrices)} complete samples with {errors} errors")
    
    if len(csi_matrices) == 0:
        print("No valid samples found. Check data format and paths.")
        return
    
    # Create train/val/test split
    # Use regular train_test_split instead of GroupShuffleSplit if we have few samples
    X = csi_matrices
    y = labels
    
    # Force a split even with limited data
    use_regular_split = len(X) < 3 or len(prefixes) < 3
    
    if use_regular_split:
        print("Using regular train_test_split due to limited data")
        # If we have just 1 sample, put it in train
        if len(X) == 1:
            X_train, y_train = X, y
            X_val, y_val = [], []
            X_test, y_test = [], []
        # If we have 2 samples, put them in train and val
        elif len(X) == 2:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
            X_test, y_test = [], []
        # Otherwise do a proper split
        else:
            # First split: train vs (val+test)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            # Second split: val vs test
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        # Use your original GroupShuffleSplit approach if you have enough data
        print("Using GroupShuffleSplit approach - this would be implemented here")
        # For now, just use regular split as a fallback
        if len(X) < 2:
            X_train, y_train = X, y
            X_val, y_val = [], []
            X_test, y_test = [], []
        else:
            # First split: train vs (val+test)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            # Second split: val vs test
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Split result: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Generate windows from the samples
    window_length = 51  # Default value - adjust as needed
    stride_length = 30  # Default value - adjust as needed
    
    # Create output directories
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(exp_dir, f"{split_name}_antennas_{activities}")
        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)
    
    # Process train set
    if len(X_train) > 0:
        process_split(X_train, y_train, "train", exp_dir, window_length, stride_length)
    
    # Process val set
    if len(X_val) > 0:
        process_split(X_val, y_val, "val", exp_dir, window_length, stride_length)
    
    # Process test set
    if len(X_test) > 0:
        process_split(X_test, y_test, "test", exp_dir, window_length, stride_length)
    
    print(f"\nFixed dataset split complete for {os.path.basename(exp_dir)}!")

def label_to_num(label, activities_list):
    """Convert a label to its numerical index."""
    try:
        return activities_list.index(label)
    except ValueError:
        # For handling grouped labels like "R" when only "R1", "R2" exist
        for i, act in enumerate(activities_list):
            if act.startswith(label) or label.startswith(act):
                return i
        # Default to 0 if not found
        print(f"Warning: Label {label} not found in activities list. Using index 0.")
        return 0

def process_split(X_split, y_split, split_name, exp_dir, window_length, stride_length):
    """Process and save a data split."""
    try:
        print(f"\nProcessing {split_name} split with {len(X_split)} samples")
        
        # Generate windows using the imported function
        csi_matrices_windows, labels_windows = create_windows_antennas(X_split, y_split, window_length, stride_length)
        
        print(f"Generated {len(csi_matrices_windows)} windows for {split_name} split")
        
        # Calculate expected windows
        lengths = [x.shape[2] for x in X_split]
        expected_windows = sum(np.floor((length - window_length) / stride_length + 1) for length in lengths)
        print(f"Expected windows: {int(expected_windows)}, Actual windows: {len(csi_matrices_windows)}")
        
        # Save the windows
        names_set = []
        for i, window in enumerate(csi_matrices_windows):
            # Save each window as a separate file
            file_path = os.path.join(exp_dir, f"{split_name}_antennas_{activities}", f"{i}.txt")
            names_set.append(file_path)
            with open(file_path, "wb") as fp:
                pickle.dump(window, fp)
        
        # Save the labels
        label_path = os.path.join(exp_dir, f"labels_{split_name}_antennas_{activities}.txt")
        with open(label_path, "wb") as fp:
            pickle.dump([int(label) for label in labels_windows], fp)
        
        # Save the file names
        files_path = os.path.join(exp_dir, f"files_{split_name}_antennas_{activities}.txt")
        with open(files_path, "wb") as fp:
            pickle.dump(names_set, fp)
        
        # Save the number of windows per original sample
        num_windows = [int(np.floor((length - window_length) / stride_length + 1)) for length in lengths]
        num_windows_path = os.path.join(exp_dir, f"num_windows_{split_name}_antennas_{activities}.txt")
        with open(num_windows_path, "wb") as fp:
            pickle.dump(num_windows, fp)
        
        print(f"{split_name.capitalize()} split processing complete.")
        
    except Exception as e:
        print(f"Error processing {split_name} split: {e}")
        import traceback
        traceback.print_exc()

def process_all_subdirs():
    """Process all subdirectories in the data directory."""
    subdirs = get_all_subdirs()
    print(f"Found {len(subdirs)} subdirectories to process:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    for i, subdir in enumerate(subdirs):
        print(f"\n[{i+1}/{len(subdirs)}] Processing {subdir}...")
        exp_dir, label = load_raw_data(subdir)
        if exp_dir and label:
            fix_dataset_split(exp_dir, label)
        else:
            print(f"Failed to load raw data for {subdir}. Skipping.")

if __name__ == "__main__":
    print("=== Dataset Split Fix Tool (All Subdirectories) ===")
    
    # Process all subdirectories
    process_all_subdirs()
    
    print("\nAll processing complete. Check the output above for results.") 