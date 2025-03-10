#!/usr/bin/env python3
import os
import glob
import shutil

# Get all directories in doppler_traces
doppler_dir = './doppler_traces/'
all_dirs = [d for d in os.listdir(doppler_dir) if os.path.isdir(os.path.join(doppler_dir, d)) and not d.startswith('complete_antennas_')]

# Check each directory for stream files
empty_dirs = []
for directory in all_dirs:
    dir_path = os.path.join(doppler_dir, directory)
    stream_files = glob.glob(f'{dir_path}/*_stream_*.txt')
    if not stream_files:
        empty_dirs.append(directory)

# Print the directories that will be removed
print('The following directories will be removed:')
for d in sorted(empty_dirs):
    print(f'- {d}')

# Ask for confirmation
confirmation = input(f'\nAre you sure you want to remove these {len(empty_dirs)} directories? (yes/no): ')

# Remove directories if confirmed
if confirmation.lower() == 'yes':
    for directory in empty_dirs:
        dir_path = os.path.join(doppler_dir, directory)
        shutil.rmtree(dir_path)
        print(f'Removed: {dir_path}')
    print(f'\nSuccessfully removed {len(empty_dirs)} empty directories.')
else:
    print('Operation cancelled. No directories were removed.') 