#!/usr/bin/env python3
"""
Check the label distribution in the processed dataset to verify correct activity labeling.
This script helps debug issues with label assignment by examining the processed train/val/test sets.
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_labels_file(filepath):
    """Load label data from a pickled file."""
    with open(filepath, "rb") as fp:
        try:
            labels = pickle.load(fp)
            return labels
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []


def analyze_dataset(directory, activity_labels):
    """Analyze label distribution in a dataset directory."""
    print(f"\nAnalyzing dataset in: {directory}")
    
    # Find label files
    sets = ['train', 'val', 'test']
    label_files = {}
    
    for set_name in sets:
        label_file = os.path.join(directory, f'labels_{set_name}_antennas_{activity_labels}.txt')
        if os.path.exists(label_file):
            label_files[set_name] = label_file
        else:
            print(f"Warning: No {set_name} set found at {label_file}")
    
    if not label_files:
        print("No label files found. Dataset may not have been processed yet.")
        return
    
    # Load and analyze labels
    all_labels = {}
    for set_name, file_path in label_files.items():
        labels = load_labels_file(file_path)
        if labels:
            all_labels[set_name] = labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{set_name.upper()} Set:")
            print(f"  Total samples: {len(labels)}")
            print("  Label distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"    Label {label}: {count} samples ({count/len(labels):.2%})")
    
    # Plot distribution
    if all_labels:
        plt.figure(figsize=(10, 6))
        
        # Find all unique labels across all sets
        all_unique_labels = set()
        for labels in all_labels.values():
            all_unique_labels.update(labels)
        all_unique_labels = sorted(list(all_unique_labels))
        
        # Prepare data for plotting
        x = np.arange(len(all_unique_labels))
        width = 0.25
        offsets = np.linspace(-width, width, len(all_labels))
        
        # Plot bars for each set
        for i, (set_name, labels) in enumerate(all_labels.items()):
            # Count occurrences of each label
            counts = np.zeros(len(all_unique_labels))
            unique, set_counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, set_counts):
                idx = all_unique_labels.index(label)
                counts[idx] = count
            
            plt.bar(x + offsets[i], counts, width, label=set_name.upper())
        
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Label Distribution Across Sets')
        plt.xticks(x, all_unique_labels)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(directory, f'label_distribution_{activity_labels}.png')
        plt.savefig(output_path)
        print(f"\nPlot saved to: {output_path}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check label distribution in processed datasets")
    parser.add_argument('dir', help='Directory containing the processed data')
    parser.add_argument('--activities', help='Activities string used in file naming', default='')
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist")
        exit(1)
    
    # Find all processed subdirectories
    processed_dirs = []
    for root, dirs, files in os.walk(args.dir):
        # Check if this directory contains processed data files
        has_label_files = any('labels_' in f for f in files)
        if has_label_files:
            processed_dirs.append(root)
    
    if not processed_dirs:
        print(f"No processed data directories found in {args.dir}")
        exit(1)
    
    print(f"Found {len(processed_dirs)} processed directories")
    
    # Analyze each directory
    for directory in processed_dirs:
        # Try to determine the activity labels string from files
        if not args.activities:
            label_files = [f for f in os.listdir(directory) if f.startswith('labels_')]
            if label_files:
                # Extract activities string from filename
                parts = label_files[0].split('antennas_')
                if len(parts) > 1:
                    activities = parts[1].split('.')[0]
                else:
                    activities = ''
            else:
                activities = ''
        else:
            activities = args.activities
        
        analyze_dataset(directory, activities)
    
    print("\nAnalysis complete.") 