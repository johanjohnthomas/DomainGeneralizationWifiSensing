#!/usr/bin/env python
"""
Balance an existing dataset by undersampling the majority classes to match the minority class count.
This script can be used to balance classes in a dataset that has already been created.
"""

import os
import sys
import argparse
import numpy as np
import pickle
import glob
from SHARP-main.Python_code.dataset_utility import balance_classes_by_undersampling


def balance_dataset_in_directory(dataset_dir, random_seed=42, dry_run=False):
    """
    Balance all label files in a directory by undersampling.
    
    Args:
        dataset_dir: Directory containing the dataset files
        random_seed: Random seed for reproducibility
        dry_run: If True, only show what would be done but don't make changes
    """
    # Find all label files
    label_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.startswith("labels_") and file.endswith(".txt"):
                label_files.append(os.path.join(root, file))
    
    if not label_files:
        print(f"No label files found in {dataset_dir}")
        return
    
    print(f"Found {len(label_files)} label files to process")
    
    # Process each label file
    for label_file in label_files:
        print(f"\nProcessing: {label_file}")
        
        # Load labels
        try:
            with open(label_file, "rb") as fp:
                labels = pickle.load(fp)
        except Exception as e:
            print(f"Error loading label file {label_file}: {e}")
            continue
        
        # Determine the corresponding data files
        base_name = os.path.basename(label_file).replace("labels_", "")
        data_dir = os.path.dirname(label_file)
        
        # Extract the pattern for data files (e.g., "complete_antennas_E_J_L_W")
        pattern = base_name.replace(".txt", "")
        
        # Get associated file list if it exists
        files_list_path = os.path.join(data_dir, f"files_{pattern}.txt")
        if os.path.exists(files_list_path):
            try:
                with open(files_list_path, "rb") as fp:
                    data_files = pickle.load(fp)
                print(f"Loaded file list with {len(data_files)} entries")
            except Exception as e:
                print(f"Error loading file list {files_list_path}: {e}")
                continue
        else:
            # Try to find data files based on naming pattern
            data_dir_pattern = os.path.join(data_dir, pattern)
            if os.path.isdir(data_dir_pattern):
                data_files = [os.path.join(data_dir_pattern, f"{i}.txt") for i in range(len(labels))]
                print(f"Generated file list with {len(data_files)} entries")
            else:
                print(f"Could not find data directory: {data_dir_pattern}")
                continue
        
        # Load data files
        data = []
        for i, file_path in enumerate(data_files):
            try:
                with open(file_path, "rb") as fp:
                    data.append(pickle.load(fp))
            except Exception as e:
                print(f"Error loading data file {file_path}: {e}")
                # Stop after failing to load more than 5 files
                if i > 5 and i / len(data_files) > 0.1:
                    print("Too many errors loading data files. Aborting.")
                    break
        
        if len(data) != len(labels):
            print(f"Mismatch between data ({len(data)}) and labels ({len(labels)}). Skipping.")
            continue
        
        # Apply class balancing
        print(f"Applying class balancing to {len(data)} samples...")
        balanced_data, balanced_labels = balance_classes_by_undersampling(data, labels, random_seed=random_seed)
        
        if dry_run:
            print("Dry run mode - not saving changes.")
            continue
        
        # Backup original files
        backup_label_file = f"{label_file}.bak"
        if not os.path.exists(backup_label_file):
            print(f"Backing up original label file to {backup_label_file}")
            with open(backup_label_file, "wb") as fp:
                pickle.dump(labels, fp)
        
        backup_files_list = f"{files_list_path}.bak"
        if os.path.exists(files_list_path) and not os.path.exists(backup_files_list):
            print(f"Backing up original file list to {backup_files_list}")
            with open(backup_files_list, "wb") as fp:
                pickle.dump(data_files, fp)
        
        # Save balanced labels
        print(f"Saving balanced labels with {len(balanced_labels)} entries")
        with open(label_file, "wb") as fp:
            pickle.dump(balanced_labels, fp)
        
        # Save balanced data files
        balanced_data_files = []
        data_dir_balanced = os.path.join(data_dir, f"{pattern}_balanced")
        os.makedirs(data_dir_balanced, exist_ok=True)
        
        print(f"Saving {len(balanced_data)} balanced data files to {data_dir_balanced}")
        for i, data_item in enumerate(balanced_data):
            file_path = os.path.join(data_dir_balanced, f"{i}.txt")
            balanced_data_files.append(file_path)
            with open(file_path, "wb") as fp:
                pickle.dump(data_item, fp)
        
        # Update file list
        print(f"Updating file list")
        with open(files_list_path, "wb") as fp:
            pickle.dump(balanced_data_files, fp)
        
        print(f"Successfully balanced dataset in {label_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Path to the dataset directory')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Do not make changes, just show what would be done')
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset directory not found: {args.dataset}")
        return 1
    
    print(f"Balancing dataset in: {args.dataset}")
    balance_dataset_in_directory(args.dataset, random_seed=args.random_seed, dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 