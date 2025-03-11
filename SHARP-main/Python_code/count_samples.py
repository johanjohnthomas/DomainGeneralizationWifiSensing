import pickle
import os
import numpy as np

base_dir = "./doppler_traces/"
subdirs = ["AR1a_S", "AR1b_E", "AR1c_C", "AR3a_R", "AR4a_R", "AR5a_C", "AR6a_E", "AR7a_J1", "AR8a_E1", "AR9a_J1", "AR9b_J1"]
activities = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"
num_antennas = 4

def expand_antennas(file_names, labels, num_antennas):
    file_names_expanded = [item for item in file_names for _ in range(num_antennas)]
    labels_expanded = [int(label) for label in labels for _ in range(num_antennas)]
    stream_ant = np.tile(np.arange(num_antennas), len(labels))
    return file_names_expanded, labels_expanded, stream_ant

# Collect all data
train_files = []
train_labels = []
val_files = []
val_labels = []
test_files = []
test_labels = []

# Load data for each subdirectory
for subdir in subdirs:
    # Load train data
    file_path = f"{base_dir}{subdir}/files_train_antennas_{activities}.txt"
    label_path = f"{base_dir}{subdir}/labels_train_antennas_{activities}.txt"
    
    if os.path.exists(file_path) and os.path.exists(label_path):
        with open(file_path, "rb") as f:
            files = pickle.load(f)
            train_files.extend(files)
        
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            # Replicate labels for each file
            train_labels.extend([labels[0]] * len(files))
    
    # Load validation data
    file_path = f"{base_dir}{subdir}/files_val_antennas_{activities}.txt"
    label_path = f"{base_dir}{subdir}/labels_val_antennas_{activities}.txt"
    
    if os.path.exists(file_path) and os.path.exists(label_path):
        with open(file_path, "rb") as f:
            files = pickle.load(f)
            val_files.extend(files)
        
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            # Replicate labels for each file
            val_labels.extend([labels[0]] * len(files))
    
    # Load test data
    file_path = f"{base_dir}{subdir}/files_test_antennas_{activities}.txt"
    label_path = f"{base_dir}{subdir}/labels_test_antennas_{activities}.txt"
    
    if os.path.exists(file_path) and os.path.exists(label_path):
        with open(file_path, "rb") as f:
            files = pickle.load(f)
            test_files.extend(files)
        
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            # Replicate labels for each file
            test_labels.extend([labels[0]] * len(files))

# Count samples before expansion
print("Before expansion:")
print(f"Train samples: {len(train_files)}")
print(f"Training labels: {np.unique(train_labels, return_counts=True)}")
print(f"Val samples: {len(val_files)}")
print(f"Val labels: {np.unique(val_labels, return_counts=True)}")
print(f"Test samples: {len(test_files)}")
print(f"Test labels: {np.unique(test_labels, return_counts=True)}")

# Expand for antennas
train_files_expanded, train_labels_expanded, _ = expand_antennas(train_files, train_labels, num_antennas)
val_files_expanded, val_labels_expanded, _ = expand_antennas(val_files, val_labels, num_antennas)
test_files_expanded, test_labels_expanded, _ = expand_antennas(test_files, test_labels, num_antennas)

# Count samples after expansion
print("\nAfter expansion:")
print(f"Train samples: {len(train_files_expanded)}")
print(f"Training labels: {np.unique(train_labels_expanded, return_counts=True)}")
print(f"Val samples: {len(val_files_expanded)}")
print(f"Val labels: {np.unique(val_labels_expanded, return_counts=True)}")
print(f"Test samples: {len(test_files_expanded)}")
print(f"Test labels: {np.unique(test_labels_expanded, return_counts=True)}")

# Count total files
total_train = len(train_files)
total_val = len(val_files)
total_test = len(test_files)

total_train_expanded = len(train_files_expanded)
total_val_expanded = len(val_files_expanded)
total_test_expanded = len(test_files_expanded)

print(f"\nTotal samples before expansion: {total_train + total_val + total_test}")
print(f"Total samples after expansion: {total_train_expanded + total_val_expanded + total_test_expanded}") 