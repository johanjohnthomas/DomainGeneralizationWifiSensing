#!/bin/bash

# Clear output file at the start
: > specific_files_export.txt

# Define the specific files to extract
files=(
    "CSI_doppler_create_dataset_test.py"
    "CSI_doppler_create_dataset_train.py"
    "CSI_network.py"
    "CSI_network_test.py"
    "dataset_utility.py"
    # "network_utility.py"
    # "optimization_utility.py"
    "Makefile"
)

# Go to Python_code directory (adjust path if needed)
cd /home/johan/Documents/DomainGeneralizationWifiSensing/SHARP-main/Python_code

# Export each specific file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "./$file:" >> specific_files_export.txt
        echo '```' >> specific_files_export.txt
        cat "$file" >> specific_files_export.txt
        echo '```' >> specific_files_export.txt
        echo "" >> specific_files_export.txt
    else
        echo "Warning: File $file not found" >> specific_files_export.txt
        echo "" >> specific_files_export.txt
    fi
done

# Use a simpler approach with grep to handle the directory structure
echo "# Directory Structure" >> specific_files_export.txt
echo '```' >> specific_files_export.txt

# Print non-AR directories (like ".")
find . -type d | grep -v "/doppler_traces/AR" | sort >> specific_files_export.txt

# Print doppler_traces and AR1a main directory
echo "./doppler_traces" >> specific_files_export.txt
echo "./doppler_traces/AR1a" >> specific_files_export.txt

# Print AR1a subdirectories with simple truncation, but show AR1a_C completely
find ./doppler_traces/AR1a -type d | grep -v "antennas" | sort | awk '
BEGIN { prev_prefix = ""; count = 0; }
{
    # Show all AR1a_C directories
    if ($0 ~ /AR1a_C/) {
        print $0;
    }
    else if (count > 0 && $0 ~ prev_prefix) {
        count++;
        if (count <= 3) print $0;
        else if (count == 4) print prev_prefix"...";
    } else {
        print $0;
        split($0, parts, "/");
        if (length(parts) >= 5) {
            prev_prefix = "./doppler_traces/AR1a/" parts[4] "/";
            count = 1;
        } else {
            prev_prefix = "";
            count = 0;
        }
    }
}' >> specific_files_export.txt

# Print just the existence of other AR directories without contents
find ./doppler_traces -maxdepth 2 -type d | grep -v "AR1a" | grep "AR" | sort >> specific_files_export.txt

echo '```' >> specific_files_export.txt
echo "" >> specific_files_export.txt

# ADD A CLEAR SECTION FOR AR1a_C CONTENTS
echo "# CONTENTS OF AR1a_C DIRECTORY" >> specific_files_export.txt
echo '```' >> specific_files_export.txt

echo "## Directories in AR1a_C:" >> specific_files_export.txt
find ./doppler_traces/AR1a/AR1a_C -type d | grep -v "^./doppler_traces/AR1a/AR1a_C$" | sed 's|./doppler_traces/AR1a/AR1a_C/|  |' >> specific_files_export.txt
echo "" >> specific_files_export.txt

echo "## Files in AR1a_C:" >> specific_files_export.txt
find ./doppler_traces/AR1a/AR1a_C -maxdepth 1 -type f | sort | sed 's|./doppler_traces/AR1a/AR1a_C/|  |' | head -n 20 >> specific_files_export.txt
if [ $(find ./doppler_traces/AR1a/AR1a_C -maxdepth 1 -type f | wc -l) -gt 20 ]; then
    echo "  ... (more files in AR1a_C)" >> specific_files_export.txt
fi

# If there are no files directly in AR1a_C, check for files in subdirectories
if [ $(find ./doppler_traces/AR1a/AR1a_C -maxdepth 1 -type f | wc -l) -eq 0 ]; then
    echo "" >> specific_files_export.txt
    echo "## Files in AR1a_C subdirectories (sample):" >> specific_files_export.txt
    find ./doppler_traces/AR1a/AR1a_C -type f | head -n 20 | sed 's|./doppler_traces/AR1a/AR1a_C/|  |' >> specific_files_export.txt
    if [ $(find ./doppler_traces/AR1a/AR1a_C -type f | wc -l) -gt 20 ]; then
        echo "  ... (more files in subdirectories)" >> specific_files_export.txt
    fi
fi
echo '```' >> specific_files_export.txt
echo "" >> specific_files_export.txt

# Better tree output truncation
echo "# Tree Structure of Python_code Directory" >> specific_files_export.txt
echo '```' >> specific_files_export.txt
tree -L 3 | head -n 50 >> specific_files_export.txt
echo "... (tree output truncated for brevity)" >> specific_files_export.txt
echo '```' >> specific_files_export.txt

echo "Export completed to specific_files_export.txt" 