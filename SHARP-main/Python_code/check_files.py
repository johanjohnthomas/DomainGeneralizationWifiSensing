import pickle
import os

base_dir = "./doppler_traces/"
subdirs = ["AR1a_S", "AR1b_E", "AR1c_C", "AR3a_R", "AR4a_R", "AR5a_C", "AR6a_E", "AR7a_J1", "AR8a_E1", "AR9a_J1", "AR9b_J1"]
activities = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"

def print_file_info(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"File: {file_path}")
            print(f"  Length: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"  First few items: {data[:min(3, len(data))]}")
            print()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

total_train = 0
total_val = 0
total_test = 0

for subdir in subdirs:
    print(f"\nChecking {subdir}:")
    
    # Train files
    train_files_path = f"{base_dir}{subdir}/files_train_antennas_{activities}.txt"
    print_file_info(train_files_path)
    
    # Train labels
    train_labels_path = f"{base_dir}{subdir}/labels_train_antennas_{activities}.txt"
    print_file_info(train_labels_path)
    
    # Validation files
    val_files_path = f"{base_dir}{subdir}/files_val_antennas_{activities}.txt"
    print_file_info(val_files_path)
    
    # Validation labels
    val_labels_path = f"{base_dir}{subdir}/labels_val_antennas_{activities}.txt"
    print_file_info(val_labels_path)
    
    # Test files
    test_files_path = f"{base_dir}{subdir}/files_test_antennas_{activities}.txt"
    print_file_info(test_files_path)
    
    # Test labels
    test_labels_path = f"{base_dir}{subdir}/labels_test_antennas_{activities}.txt"
    print_file_info(test_labels_path)
    
    # Count files in directories
    try:
        train_files = len(os.listdir(f"{base_dir}{subdir}/train_antennas_{activities}/"))
        val_files = len(os.listdir(f"{base_dir}{subdir}/val_antennas_{activities}/"))
        test_files = len(os.listdir(f"{base_dir}{subdir}/test_antennas_{activities}/"))
        
        print(f"Files in train directory: {train_files}")
        print(f"Files in val directory: {val_files}")
        print(f"Files in test directory: {test_files}")
        
        total_train += train_files
        total_val += val_files
        total_test += test_files
    except Exception as e:
        print(f"Error counting files: {e}")

print("\nTotal counts:")
print(f"Total train files: {total_train}")
print(f"Total val files: {total_val}")
print(f"Total test files: {total_test}") 