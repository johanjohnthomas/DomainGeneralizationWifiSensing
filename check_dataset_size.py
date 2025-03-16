#!/usr/bin/env python
"""
Diagnostic script to check dataset sizes and structure.
This script helps identify why only one sample is being processed.
"""

import os
import glob
import numpy as np
import pickle

# Configuration - modify these to match your setup
base_dir = "./SHARP-main/"
data_dir = os.path.join(base_dir, "Python_code/doppler_traces/")  # Updated path based on find results
subdir = "AR3a_R"  # Based on your error output showing AR3a_R_stream_*
activities = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"

def inspect_raw_files():
    """Check the raw input files to understand the data structure."""
    print("\n=== Inspecting Raw Files ===")
    
    # Check for stream files in the subdir
    stream_files = glob.glob(os.path.join(data_dir, subdir, "*_stream_*.txt"))
    print(f"Found {len(stream_files)} stream files in {os.path.join(data_dir, subdir)}")
    
    # Show some examples
    for i, file in enumerate(sorted(stream_files)[:10]):
        print(f"  {i+1}. {os.path.basename(file)}")
        
        # Try to load the file to check its content
        try:
            with open(file, "rb") as fp:
                data = pickle.load(fp)
                if isinstance(data, list):
                    data_shape = f"list of length {len(data)}"
                elif hasattr(data, 'shape'):
                    data_shape = f"array with shape {data.shape}"
                else:
                    data_shape = f"type {type(data)}"
                    
                print(f"     Content: {data_shape}")
        except Exception as e:
            print(f"     Error reading file: {e}")
    
    print(f"\nTotal stream files: {len(stream_files)}")
    
    # Analyze file naming patterns
    prefixes = set()
    for file in stream_files:
        base = os.path.basename(file)
        if '_stream_' in base:
            prefix = base.split('_stream_')[0]
            prefixes.add(prefix)
    
    print(f"Found {len(prefixes)} unique prefixes: {prefixes}")
    
    # Count files per stream
    streams = {}
    for file in stream_files:
        base = os.path.basename(file)
        if '_stream_' in base:
            stream_num = base.split('_stream_')[1].split('.')[0]
            streams[stream_num] = streams.get(stream_num, 0) + 1
    
    print("\nFiles per stream:")
    for stream, count in sorted(streams.items()):
        print(f"  Stream {stream}: {count} files")

def inspect_processed_data():
    """Check the processed data files after train/val/test split."""
    print("\n=== Inspecting Processed Data ===")
    
    # Check for processed data directories
    train_dir = os.path.join(data_dir, subdir, f"train_antennas_{activities}")
    val_dir = os.path.join(data_dir, subdir, f"val_antennas_{activities}")
    test_dir = os.path.join(data_dir, subdir, f"test_antennas_{activities}")
    
    print(f"Train directory exists: {os.path.exists(train_dir)}")
    print(f"Val directory exists: {os.path.exists(val_dir)}")
    print(f"Test directory exists: {os.path.exists(test_dir)}")
    
    # Count files in each directory
    train_files = glob.glob(os.path.join(train_dir, "*.txt")) if os.path.exists(train_dir) else []
    val_files = glob.glob(os.path.join(val_dir, "*.txt")) if os.path.exists(val_dir) else []
    test_files = glob.glob(os.path.join(test_dir, "*.txt")) if os.path.exists(test_dir) else []
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Check label files
    label_files = [
        os.path.join(data_dir, subdir, f"labels_train_antennas_{activities}.txt"),
        os.path.join(data_dir, subdir, f"labels_val_antennas_{activities}.txt"),
        os.path.join(data_dir, subdir, f"labels_test_antennas_{activities}.txt")
    ]
    
    for label_file in label_files:
        if os.path.exists(label_file):
            try:
                with open(label_file, "rb") as fp:
                    labels = pickle.load(fp)
                    print(f"{os.path.basename(label_file)}: {len(labels)} labels")
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        print(f"  Label {label}: {count} samples")
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
        else:
            print(f"{os.path.basename(label_file)} does not exist")

def check_sample_allocation():
    """Check the sample allocation logic in the dataset creation process."""
    print("\n=== Analyzing Sample Allocation Logic ===")
    
    # Look for original CSI matrices before windowing
    num_windows_files = [
        os.path.join(data_dir, subdir, f"num_windows_train_antennas_{activities}.txt"),
        os.path.join(data_dir, subdir, f"num_windows_val_antennas_{activities}.txt"),
        os.path.join(data_dir, subdir, f"num_windows_test_antennas_{activities}.txt")
    ]
    
    for window_file in num_windows_files:
        if os.path.exists(window_file):
            try:
                with open(window_file, "rb") as fp:
                    windows = pickle.load(fp)
                    print(f"{os.path.basename(window_file)}: {windows}")
                    if isinstance(windows, (list, np.ndarray)):
                        print(f"  Total windows: {sum(windows) if len(windows) > 0 else 0}")
            except Exception as e:
                print(f"Error reading {window_file}: {e}")
        else:
            print(f"{os.path.basename(window_file)} does not exist")
    
    # Check the file storing group information if it exists
    group_file = os.path.join(data_dir, subdir, "group_info.txt")
    if os.path.exists(group_file):
        try:
            with open(group_file, "rb") as fp:
                groups = pickle.load(fp)
                print(f"Group information: {len(groups)} groups")
                unique_groups = np.unique(groups)
                print(f"Unique groups: {unique_groups}")
        except Exception as e:
            print(f"Error reading group file: {e}")
    else:
        print("No group information file found")

def suggest_solutions():
    """Provide potential solutions based on the diagnostic information."""
    print("\n=== Suggested Solutions ===")
    print("1. Data Volume Issue: If you only have one unique recording session/subject in your dataset,")
    print("   the GroupShuffleSplit strategy won't be able to split it across train/val/test.")
    print("   Solution: Add more data from different recording sessions.")
    
    print("\n2. Filtering Issue: The code might be filtering out data that doesn't match expected labels.")
    print("   Solution: Check if your labels in the raw data match the expected format.")
    
    print("\n3. Data Loading Issue: There might be issues with loading/parsing your data files.")
    print("   Solution: Verify the format of your data files matches what the code expects.")
    
    print("\n4. Splitting Strategy: You might need to adjust the splitting strategy.")
    print("   Solution: Modify the splitting logic in CSI_doppler_create_dataset_train.py to use")
    print("   regular train_test_split instead of GroupShuffleSplit if you only have one group.")

if __name__ == "__main__":
    print("=== Dataset Diagnostic Tool ===")
    print(f"Checking data for subdirectory: {subdir}")
    
    # Run diagnostic functions
    inspect_raw_files()
    inspect_processed_data()
    check_sample_allocation()
    suggest_solutions()
    
    print("\nDiagnostics complete. Check the output above for insights.") 