#!/usr/bin/env python3
"""
Analyze how model performance scales with increasing number of source domains (RQ4).
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
    create_lineplot, 
    create_barplot,
    set_rq_color_scheme,
    COLOR_SCHEMES,
    load_metadata
)

# Set research question for this script
RQ = 'RQ4'


def collect_scaling_metrics(input_dir):
    """
    Collect metrics from source scaling experiments
    """
    metrics_data = []
    
    # Search for metrics in source_N directories
    for i in range(1, 5):  # sources 1-4
        source_dir = os.path.join(input_dir, f"source_{i}")
        
        if not os.path.isdir(source_dir):
            continue
            
        metrics_file = os.path.join(source_dir, "metrics.csv")
        
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                # Add number of sources if not present
                if 'num_sources' not in df.columns:
                    df['num_sources'] = i
                metrics_data.append(df)
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
    
    # Also check central metrics file
    central_metrics = os.path.join(input_dir, "..", "experiment_metrics.csv")
    if os.path.exists(central_metrics):
        try:
            df = pd.read_csv(central_metrics)
            # Filter for source scaling experiments
            scaling_df = df[df['experiment_id'].str.startswith('source')]
            
            if not scaling_df.empty:
                # Extract number of sources from experiment_id if needed
                if 'num_sources' not in scaling_df.columns:
                    scaling_df['num_sources'] = scaling_df['experiment_id'].apply(
                        lambda x: int(x.replace('source', '')) if x.replace('source', '').isdigit() else 0
                    )
                metrics_data.append(scaling_df)
        except Exception as e:
            print(f"Error reading central metrics file: {e}")
    
    if not metrics_data:
        raise ValueError("No source scaling metrics found")
        
    # Combine all metrics
    combined_df = pd.concat(metrics_data)
    
    # Remove duplicates (prefer metrics from dedicated directories over central file)
    combined_df = combined_df.sort_values('num_sources').drop_duplicates(subset=['experiment_id'], keep='first')
    
    return combined_df


def plot_scaling_trends(df, output_dir):
    """
    Create visualizations for source domain scaling analysis using standardized styling
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the RQ-specific color scheme (green for RQ4)
    set_rq_color_scheme(RQ)
    
    # Sort by number of sources
    df = df.sort_values('num_sources')
    
    # Plot 1: Line plot of accuracy vs number of source domains
    # Use the helper function for creating a line plot
    create_lineplot(
        x=df['num_sources'],
        y={
            'Training Accuracy': df['train_acc'],
            'Testing Accuracy': df['test_acc']
        },
        rq=RQ,
        experiment_id='source_scaling',
        title='Model Accuracy vs. Number of Source Domains',
        xlabel='Number of Source Domains',
        ylabel='Accuracy'
    )
    
    # Add accuracy values at each point
    ax = plt.gca()
    for i, row in df.iterrows():
        plt.annotate(f"{row['train_acc']:.3f}", 
                    (row['num_sources'], row['train_acc']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10)
        plt.annotate(f"{row['test_acc']:.3f}", 
                    (row['num_sources'], row['test_acc']),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10)
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'accuracy_vs_source_domains', 
        'source_scaling'
    )
    
    # Plot 2: Line plot of generalization gap vs number of source domains
    create_lineplot(
        x=df['num_sources'],
        y=df['gap'],
        rq=RQ,
        experiment_id='source_scaling',
        title='Generalization Gap vs. Number of Source Domains',
        xlabel='Number of Source Domains',
        ylabel='Generalization Gap (train_acc - test_acc)'
    )
    
    # Add gap values at each point
    ax = plt.gca()
    for i, row in df.iterrows():
        plt.annotate(f"{row['gap']:.3f}", 
                    (row['num_sources'], row['gap']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=11)
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'gap_vs_source_domains', 
        'source_scaling'
    )
    
    # Plot 3: Bar chart comparing metrics across number of source domains
    # Reshape data for grouped bar chart
    plot_data = pd.melt(df, 
                        id_vars=['num_sources'], 
                        value_vars=['train_acc', 'test_acc', 'gap'],
                        var_name='Metric', 
                        value_name='Value')
    
    # Replace metric names for better display
    plot_data['Metric'] = plot_data['Metric'].replace({
        'train_acc': 'Training Acc',
        'test_acc': 'Testing Acc',
        'gap': 'Gen. Gap'
    })
    
    plt.figure(figsize=(12, 8))
    
    # Create a custom color palette with green gradient for RQ4
    palette = sns.color_palette(COLOR_SCHEMES[RQ]['cmap'], n_colors=3)
    
    ax = sns.barplot(x='num_sources', y='Value', hue='Metric', data=plot_data, palette=palette)
    plt.title('Model Performance Metrics vs. Number of Source Domains', fontsize=16)
    plt.xlabel('Number of Source Domains', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='', fontsize=12)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.3f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    
    # Save using standardized naming
    save_plot_standardized(
        output_dir, 
        RQ, 
        'metrics_comparison_by_sources', 
        'all_sources'
    )
    
    # Create individual source count plots
    for num_sources in df['num_sources'].unique():
        source_df = df[df['num_sources'] == num_sources]
        exp_id = source_df['experiment_id'].iloc[0]
        
        # Use the helper function for creating a bar plot
        create_barplot(
            x=['Train Acc', 'Test Acc', 'Gap'],
            y=[source_df['train_acc'].iloc[0], source_df['test_acc'].iloc[0], source_df['gap'].iloc[0]],
            rq=RQ,
            experiment_id=exp_id,
            title=f'Performance Metrics with {int(num_sources)} Source Domain{"s" if num_sources > 1 else ""}',
            xlabel='Metric',
            ylabel='Value'
        )
        
        # Save source-specific plot
        save_plot_standardized(
            output_dir, 
            RQ, 
            'metrics', 
            exp_id
        )


def export_scaling_data(df, output_dir):
    """
    Export scaling analysis to CSV file
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate trend statistics
    scaling_stats = {
        'training_accuracy_slope': np.polyfit(df['num_sources'], df['train_acc'], 1)[0],
        'testing_accuracy_slope': np.polyfit(df['num_sources'], df['test_acc'], 1)[0],
        'gap_slope': np.polyfit(df['num_sources'], df['gap'], 1)[0],
        'max_test_accuracy': df['test_acc'].max(),
        'min_gap': df['gap'].min()
    }
    
    # Save scaling metrics
    df.to_csv(os.path.join(output_dir, 'scaling_metrics.csv'), index=False)
    
    # Save trend statistics
    pd.DataFrame([scaling_stats]).to_csv(os.path.join(output_dir, 'scaling_trends.csv'), index=False)
    
    print(f"Scaling analysis exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze source domain scaling effects')
    parser.add_argument('--input-dir', required=True, help='Directory containing the source scaling metrics')
    parser.add_argument('--output', required=True, help='Directory to save the output plots and data')
    args = parser.parse_args()
    
    try:
        # Collect metrics from all source scaling experiments
        df = collect_scaling_metrics(args.input_dir)
        print(f"Found data for {len(df)} source scaling configurations")
        
        # Generate plots with standardized styling
        plot_scaling_trends(df, args.output)
        
        # Export analysis data
        export_scaling_data(df, args.output)
        
        print("Source scaling analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 