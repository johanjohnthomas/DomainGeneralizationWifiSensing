import pickle
import os
import numpy as np

# Configuration
base_dir = "./doppler_traces/"
activities = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"
subdirs = ["AR1a_S", "AR1b_E", "AR1c_C", "AR3a_R", "AR4a_R", "AR5a_C", "AR6a_E", "AR7a_J1", "AR8a_E1", "AR9a_J1", "AR9b_J1"]

def check_file(filepath):
    try:
        with open(filepath, "rb") as fp:
            data = pickle.load(fp)
            print(f"  File: {filepath}")
            print(f"  Data type: {type(data)}")
            print(f"  Length: {len(data)}")
            print(f"  Content: {data[:10]}")  # Show first 10 items
            print(f"  Unique values: {np.unique(data)}")
            print()
            return data
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        print()
        return None

# Check each subdirectory
for subdir in subdirs:
    print(f"\nChecking subdirectory: {subdir}")
    
    # Check training files
    label_path = f"{base_dir}{subdir}/labels_train_antennas_{activities}.txt"
    file_path = f"{base_dir}{subdir}/files_train_antennas_{activities}.txt"
    
    print("Training labels:")
    labels = check_file(label_path)
    
    print("Training files:")
    files = check_file(file_path)
    
    if labels is not None and files is not None:
        print(f"  Number of labels: {len(labels)}")
        print(f"  Number of files: {len(files)}")
        if len(labels) != len(files):
            print("  WARNING: Mismatch between number of labels and files!")
            
# Print total counts
total_labels = []
total_files = []

for subdir in subdirs:
    label_path = f"{base_dir}{subdir}/labels_train_antennas_{activities}.txt"
    file_path = f"{base_dir}{subdir}/files_train_antennas_{activities}.txt"
    
    try:
        with open(label_path, "rb") as fp:
            labels = pickle.load(fp)
            total_labels.extend(labels)
        with open(file_path, "rb") as fp:
            files = pickle.load(fp)
            total_files.extend(files)
    except Exception as e:
        print(f"Error loading {subdir}: {e}")

print("\nTotal Statistics:")
print(f"Total number of labels: {len(total_labels)}")
print(f"Total number of files: {len(total_files)}")
print(f"Unique labels: {np.unique(total_labels)}")
print(f"Label distribution:")
unique_labels, counts = np.unique(total_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  Label {label}: {count} samples") 