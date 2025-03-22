#!/usr/bin/env python3
"""
Analyze tradeoffs between model accuracy and generalizability (RQ3).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


def collect_all_metrics(source_dir, target_dir):
    """
    Collect all available metrics from experiment results
    """
    metrics_data = []
    
    # Check for central metrics file
    central_metrics = os.path.join(target_dir, "experiment_metrics.csv")
    if os.path.exists(central_metrics):
        try:
            metrics_data.append(pd.read_csv(central_metrics))
            print(f"Loaded metrics from {central_metrics}")
        except Exception as e:
            print(f"Error reading central metrics file: {e}")
    
    # Check for metrics in RQ1 leave-one-out experiments
    rq1_dir = os.path.join(target_dir, "RQ1_generalization/leave_one_out")
    if os.path.exists(rq1_dir):
        for domain_dir in os.listdir(rq1_dir):
            domain_metrics = os.path.join(rq1_dir, domain_dir, "metrics", "metrics.csv")
            if os.path.exists(domain_metrics):
                try:
                    metrics_data.append(pd.read_csv(domain_metrics))
                    print(f"Loaded metrics from {domain_metrics}")
                except Exception as e:
                    print(f"Error reading {domain_metrics}: {e}")
    
    # Check for metrics in RQ4 source scaling experiments
    rq4_dir = os.path.join(target_dir, "RQ4_source_scaling")
    if os.path.exists(rq4_dir):
        for source_dir in glob.glob(os.path.join(rq4_dir, "source_*")):
            source_metrics = os.path.join(source_dir, "metrics.csv")
            if os.path.exists(source_metrics):
                try:
                    metrics_data.append(pd.read_csv(source_metrics))
                    print(f"Loaded metrics from {source_metrics}")
                except Exception as e:
                    print(f"Error reading {source_metrics}: {e}")
    
    # Check for component ablation experiments
    rq5_dir = os.path.join(target_dir, "RQ5_component_analysis")
    if os.path.exists(rq5_dir):
        for comp_dir in os.listdir(rq5_dir):
            if os.path.isdir(os.path.join(rq5_dir, comp_dir)) and not comp_dir.endswith("plots"):
                comp_metrics = os.path.join(rq5_dir, comp_dir, "metrics.csv")
                if os.path.exists(comp_metrics):
                    try:
                        metrics_data.append(pd.read_csv(comp_metrics))
                        print(f"Loaded metrics from {comp_metrics}")
                    except Exception as e:
                        print(f"Error reading {comp_metrics}: {e}")
    
    if not metrics_data:
        raise ValueError("No metrics data found in any location")
        
    # Combine all metrics and deduplicate
    combined_df = pd.concat(metrics_data)
    combined_df = combined_df.drop_duplicates(subset=['experiment_id'])
    
    return combined_df


def analyze_tradeoffs(df):
    """
    Analyze tradeoffs between accuracy and generalization
    """
    # Calculate correlation between metrics
    correlations = {
        'train_test_corr': df['train_acc'].corr(df['test_acc']),
        'train_gap_corr': df['train_acc'].corr(df['gap']),
        'test_gap_corr': df['test_acc'].corr(df['gap']),
    }
    
    # Linear regression for train_acc vs test_acc
    train_test_coef = np.polyfit(df['train_acc'], df['test_acc'], 1)
    train_test_poly = np.poly1d(train_test_coef)
    
    # Linear regression for train_acc vs gap
    train_gap_coef = np.polyfit(df['train_acc'], df['gap'], 1)
    train_gap_poly = np.poly1d(train_gap_coef)
    
    # Group experiments by component_modified
    component_stats = df.groupby('component_modified').agg({
        'train_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std'],
        'gap': ['mean', 'std']
    }).reset_index()
    
    return correlations, train_test_poly, train_gap_poly, component_stats


def plot_tradeoff_analysis(df, correlations, train_test_poly, train_gap_poly, output_dir):
    """
    Create visualizations for accuracy vs generalizability tradeoffs
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "accuracy_vs_gap"), exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Plot 1: Scatter plot of test accuracy vs training accuracy with regression line
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='train_acc', y='test_acc', data=df, s=80, alpha=0.7)
    
    # Generate x values for regression line
    train_acc_range = np.linspace(df['train_acc'].min() - 0.05, df['train_acc'].max() + 0.05, 100)
    plt.plot(train_acc_range, train_test_poly(train_acc_range), 'r--', 
             label=f'y = {train_test_coef[0]:.4f}x + {train_test_coef[1]:.4f}')
    
    # Add experiment labels
    for i, row in df.iterrows():
        plt.annotate(row['experiment_id'], 
                    (row['train_acc'], row['test_acc']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(f'Test Accuracy vs. Training Accuracy (r = {correlations["train_test_corr"]:.4f})', fontsize=16)
    plt.xlabel('Training Accuracy', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'test_vs_train_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'test_vs_train_accuracy.pdf'))
    plt.close()
    
    # Plot 2: Scatter plot of generalization gap vs training accuracy with regression line
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='train_acc', y='gap', data=df, s=80, alpha=0.7)
    
    # Generate x values for regression line
    plt.plot(train_acc_range, train_gap_poly(train_acc_range), 'r--', 
             label=f'y = {train_gap_coef[0]:.4f}x + {train_gap_coef[1]:.4f}')
    
    # Add experiment labels
    for i, row in df.iterrows():
        plt.annotate(row['experiment_id'], 
                    (row['train_acc'], row['gap']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(f'Generalization Gap vs. Training Accuracy (r = {correlations["train_gap_corr"]:.4f})', fontsize=16)
    plt.xlabel('Training Accuracy', fontsize=14)
    plt.ylabel('Generalization Gap', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'gap_vs_train_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'gap_vs_train_accuracy.pdf'))
    plt.close()
    
    # Plot 3: Scatter plot of gap vs test accuracy
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='test_acc', y='gap', data=df, s=80, alpha=0.7)
    
    # Add experiment labels
    for i, row in df.iterrows():
        plt.annotate(row['experiment_id'], 
                    (row['test_acc'], row['gap']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(f'Generalization Gap vs. Test Accuracy (r = {correlations["test_gap_corr"]:.4f})', fontsize=16)
    plt.xlabel('Test Accuracy', fontsize=14)
    plt.ylabel('Generalization Gap', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'gap_vs_test_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", 'gap_vs_test_accuracy.pdf'))
    plt.close()
    
    # Plot 4: 3D Scatter plot of train_acc, test_acc, and gap
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['train_acc'], df['test_acc'], df['gap'], s=100, alpha=0.7)
    
    # Add experiment labels
    for i, row in df.iterrows():
        ax.text(row['train_acc'], row['test_acc'], row['gap'], row['experiment_id'], fontsize=8)
    
    ax.set_xlabel('Training Accuracy', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_zlabel('Generalization Gap', fontsize=14)
    ax.set_title('3D Relationship: Training Acc, Test Acc, and Gap', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", '3d_accuracy_gap_relationship.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_gap", '3d_accuracy_gap_relationship.pdf'))
    plt.close()
    
    # Create cross-domain variance analysis directory
    cross_domain_dir = os.path.join(output_dir, "cross_domain_variance")
    os.makedirs(cross_domain_dir, exist_ok=True)
    
    # Filter for leave-one-out experiments
    leave_one_out_df = df[df['experiment_id'].str.startswith('no_')]
    
    if not leave_one_out_df.empty:
        # Plot 5: Box plots of metrics by target domain
        plt.figure(figsize=(15, 8))
        
        # Reshape data for box plots
        plot_data = pd.melt(leave_one_out_df, 
                            id_vars=['target_domains'], 
                            value_vars=['train_acc', 'test_acc', 'gap'],
                            var_name='Metric', 
                            value_name='Value')
        
        # Replace metric names for better display
        plot_data['Metric'] = plot_data['Metric'].replace({
            'train_acc': 'Training Acc',
            'test_acc': 'Testing Acc',
            'gap': 'Gen. Gap'
        })
        
        ax = sns.boxplot(x='target_domains', y='Value', hue='Metric', data=plot_data)
        plt.title('Performance Metrics Distribution by Target Domain', fontsize=16)
        plt.xlabel('Target Domain', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cross_domain_dir, 'metrics_by_domain.png'), dpi=300)
        plt.savefig(os.path.join(cross_domain_dir, 'metrics_by_domain.pdf'))
        plt.close()
        
        # Plot 6: Heatmap of metrics by domain
        # Create pivot tables for heatmap
        for metric in ['train_acc', 'test_acc', 'gap']:
            plt.figure(figsize=(10, 8))
            
            # Sort domains by gap values
            domains_order = leave_one_out_df.sort_values(by=metric)['target_domains'].values
            
            # Create data for heatmap - just 1 row with domains as columns
            heatmap_data = leave_one_out_df.set_index('target_domains')[metric].reindex(domains_order).to_frame().T
            
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
            plt.title(f'{metric.replace("_", " ").title()} by Domain', fontsize=16)
            plt.ylabel('')
            plt.xticks(rotation=45, ha='right', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(cross_domain_dir, f'{metric}_heatmap.png'), dpi=300)
            plt.savefig(os.path.join(cross_domain_dir, f'{metric}_heatmap.pdf'))
            plt.close()


def export_tradeoff_data(df, correlations, component_stats, output_dir):
    """
    Export tradeoff analysis to CSV files
    """
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "accuracy_vs_gap"), exist_ok=True)
    
    # Save correlation data
    pd.DataFrame([correlations]).to_csv(
        os.path.join(output_dir, "accuracy_vs_gap", "metric_correlations.csv"), 
        index=False
    )
    
    # Format component stats for better readability
    component_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in component_stats.columns.values]
    component_stats.to_csv(
        os.path.join(output_dir, "accuracy_vs_gap", "component_statistics.csv"), 
        index=False
    )
    
    # Calculate and save pareto optimal points (models with best tradeoff between accuracy and generalization)
    # A point is Pareto optimal if no other point has both higher test_acc and lower gap
    df['pareto_optimal'] = False
    
    for i, row in df.iterrows():
        # Check if any other model dominates this one
        dominated = False
        for j, other_row in df.iterrows():
            if i != j:
                if (other_row['test_acc'] > row['test_acc'] and other_row['gap'] < row['gap']):
                    dominated = True
                    break
        
        if not dominated:
            df.at[i, 'pareto_optimal'] = True
    
    # Save pareto optimal models
    pareto_optimal = df[df['pareto_optimal']].sort_values('test_acc', ascending=False)
    pareto_optimal.to_csv(
        os.path.join(output_dir, "accuracy_vs_gap", "pareto_optimal_models.csv"), 
        index=False
    )
    
    print(f"Tradeoff analysis exported to {output_dir}")
    print(f"Found {len(pareto_optimal)} Pareto optimal models")


def main():
    parser = argparse.ArgumentParser(description='Analyze accuracy vs generalizability tradeoffs')
    parser.add_argument('--source-dir', required=True, help='Directory containing model files')
    parser.add_argument('--target-dir', required=True, help='Directory containing results files')
    parser.add_argument('--output', required=True, help='Directory to save the analysis outputs')
    args = parser.parse_args()
    
    try:
        # Collect metrics from all experiments
        df = collect_all_metrics(args.source_dir, args.target_dir)
        print(f"Found data for {len(df)} experiments")
        
        # Analyze tradeoffs
        correlations, train_test_poly, train_gap_poly, component_stats = analyze_tradeoffs(df)
        
        # Generate plots
        plot_tradeoff_analysis(df, correlations, train_test_poly, train_gap_poly, args.output)
        
        # Export analysis data
        export_tradeoff_data(df, correlations, component_stats, args.output)
        
        print("Tradeoff analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 