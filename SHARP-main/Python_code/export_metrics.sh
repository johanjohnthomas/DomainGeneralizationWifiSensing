#!/bin/bash

# Clear output file at the start
: > plots_metrics_export.txt

# Main directory files to extract
main_files=(
    "CSI_network_test.py"
    "CSI_network_metrics.py"
    "CSI_network_metrics_plot.py"
    "Makefile"
    "plots_utility.py"
    "update_excel_results.py"
)

# Scripts directory files to extract
script_files=(
    "plot_utils.py"
    "accuracy_vs_generalizability.py"
    "source_scaling_plots.py"
    "distribute_test_metrics.py"
    "generate_final_report.py"
    "calculate_generalization_gap.py"
    "component_impact.py"
)

# Export files from main directory
for file in "${main_files[@]}"; do
    if [ -f "$file" ]; then
        echo "./$file:" >> plots_metrics_export.txt
        echo '```' >> plots_metrics_export.txt
        cat "$file" >> plots_metrics_export.txt
        echo '```' >> plots_metrics_export.txt
        echo "" >> plots_metrics_export.txt
    else
        echo "Warning: File $file not found" >> plots_metrics_export.txt
        echo "" >> plots_metrics_export.txt
    fi
done

# Try to find scripts directory and export files
scripts_dir="./scripts"
if [ ! -d "$scripts_dir" ]; then
    scripts_dir="../../scripts"
fi

if [ -d "$scripts_dir" ]; then
    for file in "${script_files[@]}"; do
        fullpath="$scripts_dir/$file"
        if [ -f "$fullpath" ]; then
            echo "./$fullpath:" >> plots_metrics_export.txt
            echo '```' >> plots_metrics_export.txt
            cat "$fullpath" >> plots_metrics_export.txt
            echo '```' >> plots_metrics_export.txt
            echo "" >> plots_metrics_export.txt
        else
            echo "Warning: File $fullpath not found" >> plots_metrics_export.txt
            echo "" >> plots_metrics_export.txt
        fi
    done
else
    echo "Warning: Scripts directory not found" >> plots_metrics_export.txt
    echo "" >> plots_metrics_export.txt
fi

echo "Export completed to plots_metrics_export.txt" 