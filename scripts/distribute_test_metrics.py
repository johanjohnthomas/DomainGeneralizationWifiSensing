#!/usr/bin/env python3
"""
Helper script to distribute test metrics to individual experiment directories.
"""

import os
import pandas as pd

# Read the central metrics file
df = pd.read_csv('test_output/experiment_metrics.csv')

# Create the metrics directories if they don't exist
for exp_id in df['experiment_id']:
    if exp_id.startswith('no_'):
        os.makedirs(f'test_output/RQ1_generalization/leave_one_out/{exp_id}/metrics', exist_ok=True)
    elif exp_id.startswith('source'):
        os.makedirs(f'test_output/RQ4_source_scaling/{exp_id}', exist_ok=True)
    elif 'phase' in exp_id or 'antenna' in exp_id:
        os.makedirs(f'test_output/RQ5_component_analysis/{exp_id}', exist_ok=True)

# Copy metrics to individual experiment directories
for exp_id in df['experiment_id']:
    if exp_id.startswith('no_'):
        df[df['experiment_id'] == exp_id].to_csv(
            f'test_output/RQ1_generalization/leave_one_out/{exp_id}/metrics/metrics.csv', 
            index=False
        )
        print(f"Created metrics for {exp_id}")
    elif exp_id.startswith('source'):
        df[df['experiment_id'] == exp_id].to_csv(
            f'test_output/RQ4_source_scaling/{exp_id}/metrics.csv', 
            index=False
        )
        print(f"Created metrics for {exp_id}")
    elif 'phase' in exp_id or 'antenna' in exp_id:
        df[df['experiment_id'] == exp_id].to_csv(
            f'test_output/RQ5_component_analysis/{exp_id}/metrics.csv', 
            index=False
        )
        print(f"Created metrics for {exp_id}")

print("All metrics distributed successfully!") 