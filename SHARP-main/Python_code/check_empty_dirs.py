#!/usr/bin/env python3
import os
import glob

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

print('Directories without stream files:')
for d in sorted(empty_dirs):
    print(f'- {d}')
print(f'\nTotal: {len(empty_dirs)} directories without stream files out of {len(all_dirs)} total directories') 