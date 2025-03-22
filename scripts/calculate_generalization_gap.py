#!/usr/bin/env python3
"""
Calculate generalization gaps for domain-leave-out experiments (RQ1)
and create visualizations for the analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import standardized plotting utilities
from plot_utils import (
    apply_standard_style, 
    save_plot_standardized, 
    create_barplot, 
    create_scatterplot,
    set_rq_color_scheme,
    COLOR_SCHEMES,
    load_metadata
)

# Set research question for this script
RQ = 'RQ1'


def collect_metrics(input_dir):
    """
    Collect metrics from RQ1 leave-one-out experiments
    """
    metrics_data = []
    
    # Search for metrics files in the leave-one-out directory
    leave_one_out_dir = os.path.join(input_dir, "leave_one_out")
    
    for domain_dir in os.listdir(leave_one_out_dir):
        domain_path = os.path.join(leave_one_out_dir, domain_dir)
        
        if not os.path.isdir(domain_path):
            continue
            
        metrics_file = os.path.join(domain_path, "metrics", "metrics.csv")
        
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                metrics_data.append(df)
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
    
    # Also check for central metrics file
    central_metrics = os.path.join(input_dir, "..", "experiment_metrics.csv")
    if os.path.exists(central_metrics):
        try:
            df = pd.read_csv(central_metrics)
            # Filter for leave-one-out experiments
            leave_one_out_df = df[df['experiment_id'].str.startswith('no_')]
            if not leave_one_out_df.empty:
                metrics_data.append(leave_one_out_df)
        except Exception as e:
            print(f"Error reading central metrics file: {e}")
    
    if not metrics_data:
        raise ValueError("No metrics data found")
        
    # Combine all metrics
    combined_df = pd.concat(metrics_data)
    combined_df = combined_df.drop_duplicates(subset=['experiment_id'])
    
    return combined_df


def calculate_gaps(df):
    """
    Calculate statistics on generalization gaps
    """
    gap_stats = {
        'mean_gap': df['gap'].mean(),
        'std_gap': df['gap'].std(),
        'max_gap': df['gap'].max(),
        'min_gap': df['gap'].min(),
        'median_gap': df['gap'].median()
    }
    
    # Calculate per-domain stats
    domain_stats = df.groupby('target_domains').agg({
        'train_acc': 'mean',
        'test_acc': 'mean',
        'gap': ['mean', 'std']
    }).reset_index()
    
    return gap_stats, domain_stats


def plot_generalization_gaps(df, output_dir):
    """
    Create visualizations for generalization gaps using standardized styling
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the RQ-specific color scheme
    set_rq_color_scheme(RQ)
    
    # Extract domain from experiment_id for better labeling
    df['domain'] = df['target_domains'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    # Plot 1: Bar chart of generalization gaps by domain
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='domain', y='gap', data=df, palette=COLOR_SCHEMES[RQ]['palette'])
    plt.title('Generalization Gap by Left-Out Domain', fontsize=16)
    plt.xlabel('Left-Out Domain', fontsize=14)
    plt.ylabel('Generalization Gap (train_acc - test_acc)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add gap values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'generalization_gap_by_domain', 
        'all_domains'
    )
    
    # Plot 2: Paired bar chart of train vs test accuracy
    plt.figure(figsize=(12, 7))
    
    # Reshape data for paired bar chart
    plot_data = pd.melt(df, 
                        id_vars=['domain'], 
                        value_vars=['train_acc', 'test_acc'],
                        var_name='Accuracy Type', 
                        value_name='Accuracy')
    
    # Use color palette with two distinct colors from the RQ1 scheme
    palette = [COLOR_SCHEMES[RQ]['primary'], COLOR_SCHEMES[RQ]['secondary']]
    
    ax = sns.barplot(x='domain', y='Accuracy', hue='Accuracy Type', data=plot_data, palette=palette)
    plt.title('Training vs Testing Accuracy by Domain', fontsize=16)
    plt.xlabel('Left-Out Domain', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='', fontsize=12)
    
    # Add accuracy values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'train_vs_test_accuracy', 
        'all_domains'
    )
    
    # Plot 3: Gap vs Test Accuracy Scatter Plot with domain labels
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with RQ1 primary color
    sns.scatterplot(x='test_acc', y='gap', s=100, data=df, color=COLOR_SCHEMES[RQ]['primary'])
    
    # Add domain labels to each point
    for i, row in df.iterrows():
        plt.annotate(row['domain'], 
                    (row['test_acc'], row['gap']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12)
    
    plt.title('Generalization Gap vs Test Accuracy', fontsize=16)
    plt.xlabel('Test Accuracy', fontsize=14)
    plt.ylabel('Generalization Gap', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'gap_vs_accuracy_scatter', 
        'all_domains'
    )
    
    # Create individual domain plots
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        exp_id = domain_df['experiment_id'].iloc[0]
        
        # Use the helper function for creating a bar plot
        create_barplot(
            x=['Train Acc', 'Test Acc', 'Gap'],
            y=[domain_df['train_acc'].iloc[0], domain_df['test_acc'].iloc[0], domain_df['gap'].iloc[0]],
            rq=RQ,
            experiment_id=exp_id,
            title=f'Performance Metrics for {domain.capitalize()} Domain',
            xlabel='Metric',
            ylabel='Value'
        )
        
        # Save domain-specific plot
        save_plot_standardized(
            output_dir, 
            RQ, 
            'metrics', 
            exp_id
        )


def export_results(gap_stats, domain_stats, output_dir):
    """
    Export statistics to CSV files
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall stats
    pd.DataFrame([gap_stats]).to_csv(os.path.join(output_dir, 'gap_statistics.csv'), index=False)
    
    # Format domain stats for better readability
    domain_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in domain_stats.columns.values]
    domain_stats.to_csv(os.path.join(output_dir, 'domain_statistics.csv'), index=False)
    
    print(f"Results exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Calculate generalization gaps and create visualizations')
    parser.add_argument('--input-dir', required=True, help='Directory containing the experiment metrics')
    parser.add_argument('--output', required=True, help='Directory to save the outputs')
    args = parser.parse_args()
    
    # Collect metrics from all leave-one-out experiments
    try:
        df = collect_metrics(args.input_dir)
        print(f"Found data for {len(df)} experiments")
        
        # Calculate stats
        gap_stats, domain_stats = calculate_gaps(df)
        
        # Generate plots with standardized styling
        plot_generalization_gaps(df, args.output)
        
        # Export results
        export_results(gap_stats, domain_stats, args.output)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 