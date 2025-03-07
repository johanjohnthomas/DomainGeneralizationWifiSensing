import os
import shutil

# Base directory containing all the experiment folders
base_dir = "doppler_traces/"

# Target suffix to delete (consistent across all folders)
activities_suffix = "C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2"

# Iterate through all experiment folders
for experiment_folder in os.listdir(base_dir):
    exp_path = os.path.join(base_dir, experiment_folder)
    
    if os.path.isdir(exp_path):
        # Delete complete_antennas directory
        dir_to_delete = os.path.join(exp_path, f"complete_antennas_{activities_suffix}")
        if os.path.exists(dir_to_delete):
            print(f"Deleting directory: {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
        dir_to_delete = os.path.join(exp_path, f"train_antennas_{activities_suffix}")
        if os.path.exists(dir_to_delete):
            print(f"Deleting directory: {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
        dir_to_delete = os.path.join(exp_path, f"test_antennas_{activities_suffix}")
        if os.path.exists(dir_to_delete):
            print(f"Deleting directory: {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
        dir_to_delete = os.path.join(exp_path, f"val_antennas_{activities_suffix}")
        if os.path.exists(dir_to_delete):
            print(f"Deleting directory: {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
        # Delete related files
        files_to_delete = [
            f"files_complete_antennas_{activities_suffix}.txt",
            f"labels_complete_antennas_{activities_suffix}.txt",
            f"num_windows_complete_antennas_{activities_suffix}.txt",
            f"files_test_antennas_{activities_suffix}.txt",
            f"labels_test_antennas_{activities_suffix}.txt",
            f"num_windows_test_antennas_{activities_suffix}.txt",
            f"files_train_antennas_{activities_suffix}.txt",
            f"labels_train_antennas_{activities_suffix}.txt",
            f"num_windows_train_antennas_{activities_suffix}.txt",
            f"files_val_antennas_{activities_suffix}.txt",
            f"labels_val_antennas_{activities_suffix}.txt",
            f"num_windows_val_antennas_{activities_suffix}.txt"
        ]

        
        for file_name in files_to_delete:
            file_path = os.path.join(exp_path, file_name)
            if os.path.exists(file_path):
                print(f"Deleting file: {file_path}")
                os.remove(file_path)

print("Cleanup complete!")