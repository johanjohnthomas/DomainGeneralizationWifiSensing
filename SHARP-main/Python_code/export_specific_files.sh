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
    "network_utility.py"
    "optimization_utility.py"
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

# Append directory structure at the end with truncation
echo "# Directory Structure" >> specific_files_export.txt
echo '```' >> specific_files_export.txt
find . -type d | sort | awk '
{
    # Simple truncation - if we see more than 3 items with the same pattern
    if (NR > 1 && $0 ~ prev_pattern && count >= 2) {
        if (count == 2) print "...";
        count++;
    } else {
        print $0;
        # Extract pattern (directory prefix) for comparison
        split($0, parts, "/");
        if (parts[length(parts)] == "") {
            prev_pattern = "^" substr($0, 1, length($0)-1);
        } else {
            prev_pattern = "^" $0 "/";
        }
        count = 1;
    }
}' >> specific_files_export.txt
echo '```' >> specific_files_export.txt
echo "" >> specific_files_export.txt

# Append tree command output with truncation
echo "# Tree Structure of Python_code Directory" >> specific_files_export.txt
echo '```' >> specific_files_export.txt
tree | awk '
{
    # Extract the indentation prefix
    prefix = "";
    for (i = 1; i <= length($0); i++) {
        c = substr($0, i, 1);
        if (c == " " || c == "│" || c == "├" || c == "└" || c == "─") {
            prefix = prefix c;
        } else {
            break;
        }
    }
    
    # Check if we have seen this prefix before
    if (prefix == prev_prefix) {
        count++;
        if (count == 3) {
            print prefix "...";
        } else if (count > 3) {
            # Skip
        } else {
            print $0;
        }
    } else {
        prev_prefix = prefix;
        count = 1;
        print $0;
    }
}' >> specific_files_export.txt
echo '```' >> specific_files_export.txt

echo "Export completed to specific_files_export.txt" 