import os
import glob
import subprocess
from multiprocessing import Pool

# Configuration from Makefile
PHASE_PROCESSING_DIR = "./phase_processing/"
NSS = 1
NCORE = 4
START_IDX = 0
END_IDX = -1

def process_signal_file(signal_file):
    name = os.path.basename(signal_file).replace("signal_", "").replace(".txt", "")
    
    # Check if all 8 output files exist (4 r_vector and 4 Tr_vector)
    all_files_exist = True
    for stream in range(4):
        r_file = os.path.join(PHASE_PROCESSING_DIR, f"r_vector_{name}_stream_{stream}.txt")
        tr_file = os.path.join(PHASE_PROCESSING_DIR, f"Tr_vector_{name}_stream_{stream}.txt")
        if not (os.path.exists(r_file) and os.path.exists(tr_file)):
            all_files_exist = False
            break
    
    if all_files_exist:
        return f"Skipped {name}, already processed."
    
    # Run processing if any files are missing
    cmd = [
        "python", 
        "CSI_phase_sanitization_H_estimation.py",
        PHASE_PROCESSING_DIR,
        "0",
        name,
        str(NSS),
        str(NCORE),
        str(START_IDX),
        str(END_IDX)
    ]
    print(f"Processing {name}...")
    subprocess.run(cmd, check=True)
    return f"Completed {name}"

if __name__ == "__main__":
    # Get list of signal files
    signal_files = glob.glob(os.path.join(PHASE_PROCESSING_DIR, "signal_*.txt"))
    
    # Use all available cores
    num_processes = os.cpu_count()
    
    print(f"Starting parallel processing with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_signal_file, signal_files)
    
    for result in results:
        print(result)
    print("All H estimations completed!")