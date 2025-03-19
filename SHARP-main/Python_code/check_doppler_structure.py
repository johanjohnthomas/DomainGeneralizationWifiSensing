#!/usr/bin/env python3
"""
Script to check the structure of the doppler_traces directory
and validate/print the expected activity codes.
"""

import os
import argparse
import glob
import re


def analyze_doppler_directory(base_dir):
    """
    Analyze the structure of a doppler traces directory.
    """
    print(f"\nAnalyzing doppler directory structure: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all .txt files (stream files)
    txt_files = glob.glob(os.path.join(base_dir, "**/*.txt"), recursive=True)
    print(f"Found {len(txt_files)} .txt files")
    
    # Find all subdirectories
    subdirs = []
    subdir_activities = {}
    
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            subdirs = dirs
            print(f"Found {len(subdirs)} top-level subdirectories:")
            for subdir in sorted(subdirs):
                print(f"  - {subdir}")
                
                # Check if this is a setup_activity format
                if '_' in subdir:
                    parts = subdir.split('_')
                    if len(parts) >= 2:
                        setup = parts[0]
                        activity = parts[1]
                        if setup not in subdir_activities:
                            subdir_activities[setup] = []
                        subdir_activities[setup].append(activity)
    
    # Analyze activity distribution
    print("\nActivity distribution per setup:")
    for setup, activities in subdir_activities.items():
        print(f"  Setup {setup}: {len(activities)} activities")
        print(f"    Activities: {', '.join(sorted(activities))}")
    
    # Check file patterns
    print("\nAnalyzing file patterns...")
    file_patterns = {}
    for txt_file in txt_files[:100]:  # Analyze first 100 files
        filename = os.path.basename(txt_file)
        pattern = re.sub(r'\d+', '#', filename)
        if pattern not in file_patterns:
            file_patterns[pattern] = 0
        file_patterns[pattern] += 1
    
    print("Common file patterns found:")
    for pattern, count in sorted(file_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} files")
    
    # Analyze example filenames
    print("\nExample filenames:")
    for txt_file in txt_files[:5]:
        filename = os.path.basename(txt_file)
        parts = os.path.splitext(filename)[0].split('_')
        print(f"  {filename}: {len(parts)} parts -> {parts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check doppler traces directory structure")
    parser.add_argument('dir', help='Base doppler traces directory')
    args = parser.parse_args()
    
    analyze_doppler_directory(args.dir) 