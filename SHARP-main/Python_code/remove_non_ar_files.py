#!/usr/bin/env python3
"""
Script to recursively remove files and directories that don't start with 'AR' from the doppler_traces directory.
"""

import os
import sys
import shutil
from pathlib import Path

def remove_non_ar_items(directory):
    """
    Recursively remove files and directories that don't start with 'AR' from the specified directory.
    
    Args:
        directory (str): Path to the directory to process
    
    Returns:
        tuple: (removed_files, removed_dirs, preserved_files, preserved_dirs) - counts of processed items
    """
    directory_path = Path(directory)
    
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return 0, 0, 0, 0
    
    removed_files = 0
    removed_dirs = 0
    preserved_files = 0
    preserved_dirs = 0
    
    print(f"Scanning directory: {directory}")
    
    # Process all items in the current directory
    # We need to collect a list first because we might remove items during iteration
    items = list(directory_path.iterdir())
    
    for item in items:
        name = item.name
        
        # Check if item starts with 'AR'
        if not name.startswith('AR'):
            if item.is_file():
                try:
                    print(f"Removing file: {item}")
                    item.unlink()
                    removed_files += 1
                except Exception as e:
                    print(f"Error removing file {item}: {e}")
            elif item.is_dir():
                try:
                    print(f"Removing directory: {item}")
                    shutil.rmtree(item)
                    removed_dirs += 1
                except Exception as e:
                    print(f"Error removing directory {item}: {e}")
        else:
            # If it starts with AR, preserve it
            if item.is_file():
                preserved_files += 1
            elif item.is_dir():
                preserved_dirs += 1
                # Recursively process this directory
                sub_rf, sub_rd, sub_pf, sub_pd = remove_non_ar_items(item)
                removed_files += sub_rf
                removed_dirs += sub_rd
                preserved_files += sub_pf
                preserved_dirs += sub_pd
    
    return removed_files, removed_dirs, preserved_files, preserved_dirs

def main():
    # Default directory path
    default_dir = Path(__file__).parent / "doppler_traces"
    
    # Allow custom directory path from command line
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = default_dir
    
    # Execute the removal
    removed_files, removed_dirs, preserved_files, preserved_dirs = remove_non_ar_items(directory)
    
    print(f"\nSummary:")
    print(f"  Files removed: {removed_files}")
    print(f"  Directories removed: {removed_dirs}")
    print(f"  AR files preserved: {preserved_files}")
    print(f"  AR directories preserved: {preserved_dirs}")
    print(f"  Total items processed: {removed_files + removed_dirs + preserved_files + preserved_dirs}")

if __name__ == "__main__":
    main() 