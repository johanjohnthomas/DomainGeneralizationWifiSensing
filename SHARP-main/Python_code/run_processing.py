# run_processing.py
import os
import glob
import subprocess
import sys

def main(script_name, input_dir, *args):
    # Find all AR-* directories
    dirs = glob.glob(os.path.join(input_dir, "AR-*"))
    
    for dir_path in dirs:
        cmd = ["python", script_name, dir_path + os.sep, *args]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_processing.py <script> <input_dir> [args...]")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])