#!/usr/bin/env python3
"""
Analyze the impact of different components on domain generalization performance (RQ5).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Import standardized plotting utilities
from plot_utils import (
    apply_standard_style, 
    save_plot_standardized, 
    create_barplot, 
    create_heatmap,
    set_rq_color_scheme,
    COLOR_SCHEMES,
    load_metadata
)

# Set research question for this script
RQ = 'RQ5'


def collect_component_metrics(input_dir):
    """
    Collect metrics from component ablation experiments
    """
    metrics_data = []
    
    # Check component subdirectories
    for component_dir in glob.glob(os.path.join(input_dir, "no_*")):
        if os.path.isdir(component_dir) and not component_dir.endswith("plots"):
            metrics_file = os.path.join(component_dir, "metrics.csv")
            
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    metrics_data.append(df)
                    print(f"Loaded metrics from {metrics_file}")
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}")
    
    # Check for central metrics file
    central_metrics = os.path.join(input_dir, "..", "experiment_metrics.csv")
    if os.path.exists(central_metrics):
        try:
            df = pd.read_csv(central_metrics)
            # Filter for component experiments (non-leave-one-out, non-source)
            component_df = df[
                (df['component_modified'] != 'none') & 
                (~df['experiment_id'].str.startswith('no_')) & 
                (~df['experiment_id'].str.startswith('source'))
            ]
            
            if not component_df.empty:
                metrics_data.append(component_df)
                print(f"Loaded component metrics from central file")
        except Exception as e:
            print(f"Error reading central metrics file: {e}")
    
    # Look for baseline model metrics
    baseline_metrics = os.path.join(input_dir, "..", "experiment_metrics.csv")
    baseline_df = None
    
    if os.path.exists(baseline_metrics):
        try:
            df = pd.read_csv(baseline_metrics)
            # Get the baseline model (no component modification)
            baseline_candidates = df[df['component_modified'] == 'none']
            if not baseline_candidates.empty:
                # Use model with highest test accuracy as the baseline
                baseline_df = baseline_candidates.sort_values('test_acc', ascending=False).iloc[[0]]
                metrics_data.append(baseline_df)
                print(f"Added baseline model for comparison")
        except Exception as e:
            print(f"Error finding baseline metrics: {e}")
    
    if not metrics_data:
        raise ValueError("No component metrics found")
        
    # Combine all metrics
    combined_df = pd.concat(metrics_data)
    combined_df = combined_df.drop_duplicates(subset=['experiment_id', 'component_modified'])
    
    return combined_df, baseline_df


def analyze_component_impact(df, baseline_df):
    """
    Analyze the impact of each component on performance metrics
    """
    # Create a relative impact dataframe
    impact_data = []
    
    if baseline_df is not None:
        baseline_test_acc = baseline_df['test_acc'].values[0]
        baseline_train_acc = baseline_df['train_acc'].values[0]
        baseline_gap = baseline_df['gap'].values[0]
        
        # For each component modification, calculate relative changes
        for i, row in df[df['component_modified'] != 'none'].iterrows():
            component = row['component_modified']
            
            impact = {
                'component': component,
                'test_acc_change': row['test_acc'] - baseline_test_acc,
                'test_acc_pct_change': (row['test_acc'] - baseline_test_acc) / baseline_test_acc * 100,
                'train_acc_change': row['train_acc'] - baseline_train_acc,
                'train_acc_pct_change': (row['train_acc'] - baseline_train_acc) / baseline_train_acc * 100,
                'gap_change': row['gap'] - baseline_gap,
                'gap_pct_change': (row['gap'] - baseline_gap) / baseline_gap * 100 if baseline_gap != 0 else np.nan
            }
            
            impact_data.append(impact)
    
    impact_df = pd.DataFrame(impact_data) if impact_data else None
    
    # Group experiments by component and calculate aggregate statistics
    component_stats = df.groupby('component_modified').agg({
        'train_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std'],
        'gap': ['mean', 'std']
    }).reset_index()
    
    return impact_df, component_stats


def plot_component_analysis(df, impact_df, output_dir):
    """
    Create visualizations for component impact analysis using standardized styling
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the RQ-specific color scheme (rainbow for RQ5)
    set_rq_color_scheme(RQ)
    
    # Plot 1: Bar chart comparing performance with and without each component
    plt.figure(figsize=(12, 8))
    
    # Reshape data for side-by-side bar chart
    plot_data = pd.melt(df, 
                        id_vars=['component_modified'], 
                        value_vars=['train_acc', 'test_acc', 'gap'],
                        var_name='Metric', 
                        value_name='Value')
    
    # Replace metric names for better display
    plot_data['Metric'] = plot_data['Metric'].replace({
        'train_acc': 'Training Acc',
        'test_acc': 'Testing Acc',
        'gap': 'Gen. Gap'
    })
    
    ax = sns.barplot(x='component_modified', y='Value', hue='Metric', data=plot_data, palette=COLOR_SCHEMES[RQ]['palette'])
    plt.title('Performance Metrics by Component Configuration', fontsize=16)
    plt.xlabel('Component Modified', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
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
        'metrics_by_component', 
        'all_components'
    )
    
    # If we have impact data, create impact visualizations
    if impact_df is not None and not impact_df.empty:
        # Plot 2: Impact on test accuracy
        plt.figure(figsize=(10, 6))
        
        # Sort by absolute impact
        impact_sorted = impact_df.sort_values('test_acc_change')
        
        # Create bar plot using RQ5 color scheme
        ax = sns.barplot(x='component', y='test_acc_change', data=impact_sorted, palette=COLOR_SCHEMES[RQ]['palette'])
        plt.title('Impact on Test Accuracy from Removing Each Component', fontsize=16)
        plt.xlabel('Component Removed', fontsize=14)
        plt.ylabel('Change in Test Accuracy', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add values on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(f'{height:.3f}', 
                       (p.get_x() + p.get_width() / 2., height if height > 0 else height - 0.01), 
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        # Save using standardized naming
        save_plot_standardized(
            output_dir, 
            RQ, 
            'test_acc_impact', 
            'component_removal'
        )
        
        # Plot 3: Impact on generalization gap
        plt.figure(figsize=(10, 6))
        
        # Sort by absolute impact on gap
        gap_sorted = impact_df.sort_values('gap_change')
        
        # Create bar plot using RQ5 color scheme
        ax = sns.barplot(x='component', y='gap_change', data=gap_sorted, palette=COLOR_SCHEMES[RQ]['palette'])
        plt.title('Impact on Generalization Gap from Removing Each Component', fontsize=16)
        plt.xlabel('Component Removed', fontsize=14)
        plt.ylabel('Change in Generalization Gap', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add values on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(f'{height:.3f}', 
                       (p.get_x() + p.get_width() / 2., height if height > 0 else height - 0.01), 
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        # Save using standardized naming
        save_plot_standardized(
            output_dir, 
            RQ, 
            'gap_impact', 
            'component_removal'
        )
        
        # Plot 4: Combined impact metrics in a single visualization
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        heat_data = impact_df[['component', 'test_acc_change', 'train_acc_change', 'gap_change']]
        heat_data = heat_data.set_index('component')
        
        # Rename columns for better display
        heat_data.columns = ['Test Acc Δ', 'Train Acc Δ', 'Gap Δ']
        
        # Create heatmap with RQ5 color scheme (rainbow)
        ax = sns.heatmap(heat_data, annot=True, cmap=COLOR_SCHEMES[RQ]['cmap'], center=0, fmt='.3f', linewidths=.5)
        plt.title('Component Impact on Performance Metrics', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)
        
        plt.tight_layout()
        
        # Save using standardized naming
        save_plot_standardized(
            output_dir, 
            RQ, 
            'component_impact_heatmap', 
            'all_impacts'
        )
        
        # Create individual component impact plots
        for component in impact_df['component'].unique():
            component_impact = impact_df[impact_df['component'] == component]
            
            # Use the helper function for creating a bar plot
            create_barplot(
                x=['Test Acc Δ', 'Train Acc Δ', 'Gap Δ'],
                y=[
                    component_impact['test_acc_change'].iloc[0],
                    component_impact['train_acc_change'].iloc[0],
                    component_impact['gap_change'].iloc[0]
                ],
                rq=RQ,
                experiment_id=f"no_{component}",
                title=f'Impact of Removing {component.replace("_", " ").title()}',
                xlabel='Metric',
                ylabel='Change in Value'
            )
            
            # Save component-specific plot
            save_plot_standardized(
                output_dir, 
                RQ, 
                'impact', 
                f"no_{component}"
            )
        
    # Plot 5: Spider/Radar chart for component contributions
    try:
        # Prepare data for radar chart
        components = df['component_modified'].unique()
        metrics = ['train_acc', 'test_acc', 'gap']
        
        # Get the values
        values = {}
        for component in components:
            component_df = df[df['component_modified'] == component]
            if not component_df.empty:
                values[component] = [
                    component_df['train_acc'].values[0],
                    component_df['test_acc'].values[0],
                    component_df['gap'].values[0]
                ]
        
        # Create the radar chart
        plt.figure(figsize=(10, 8))
        
        # Set up the angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # close the loop
        
        # Set up the figure
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics, size=12)
        
        # Draw the plot for each component with RQ5 colors
        for i, (component, value) in enumerate(values.items()):
            color = COLOR_SCHEMES[RQ]['palette'][i % len(COLOR_SCHEMES[RQ]['palette'])]
            value += value[:1]  # close the loop
            ax.plot(angles, value, linewidth=2, label=component, color=color)
            ax.fill(angles, value, alpha=0.1, color=color)
        
        plt.title('Component Performance Profile', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
        
        plt.tight_layout()
        
        # Save using standardized naming
        save_plot_standardized(
            output_dir, 
            RQ, 
            'component_radar_chart', 
            'all_components'
        )
    except Exception as e:
        print(f"Warning: Could not generate radar chart: {e}")


def export_component_data(df, impact_df, component_stats, output_dir):
    """
    Export component analysis to CSV files
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save component metrics
    df.to_csv(os.path.join(output_dir, 'component_metrics.csv'), index=False)
    
    # Save impact metrics if available
    if impact_df is not None and not impact_df.empty:
        impact_df.to_csv(os.path.join(output_dir, 'component_impact.csv'), index=False)
    
    # Format component stats for better readability
    if component_stats is not None:
        component_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in component_stats.columns.values]
        component_stats.to_csv(os.path.join(output_dir, 'component_statistics.csv'), index=False)
    
    # Create a summary file with conclusions
    with open(os.path.join(output_dir, 'component_analysis_summary.txt'), 'w') as f:
        f.write("Component Analysis Summary\n")
        f.write("=========================\n\n")
        
        if impact_df is not None and not impact_df.empty:
            # Find most important component for test accuracy
            most_important_test = impact_df.loc[impact_df['test_acc_change'].abs().idxmax()]
            f.write(f"Most important component for test accuracy: {most_important_test['component']}\n")
            f.write(f"Impact on test accuracy: {most_important_test['test_acc_change']:.4f} ")
            f.write(f"({most_important_test['test_acc_pct_change']:.2f}%)\n\n")
            
            # Find most important component for generalization gap
            most_important_gap = impact_df.loc[impact_df['gap_change'].abs().idxmax()]
            f.write(f"Most important component for generalization gap: {most_important_gap['component']}\n")
            f.write(f"Impact on generalization gap: {most_important_gap['gap_change']:.4f} ")
            if not np.isnan(most_important_gap['gap_pct_change']):
                f.write(f"({most_important_gap['gap_pct_change']:.2f}%)\n\n")
            else:
                f.write("\n\n")
            
            # Overall component ranking by absolute impact on test accuracy
            f.write("Component ranking by impact on test accuracy:\n")
            ranked_components = impact_df.sort_values('test_acc_change', ascending=False)
            for i, row in ranked_components.iterrows():
                f.write(f"{row['component']}: {row['test_acc_change']:.4f}\n")
            
            f.write("\nComponent ranking by impact on generalization gap:\n")
            ranked_gap = impact_df.sort_values('gap_change', ascending=True)  # Lower gap is better
            for i, row in ranked_gap.iterrows():
                f.write(f"{row['component']}: {row['gap_change']:.4f}\n")
        else:
            f.write("No component impact data available.\n")
    
    print(f"Component analysis exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze component impact on domain generalization')
    parser.add_argument('--input-dir', required=True, help='Directory containing the component metrics')
    parser.add_argument('--output', required=True, help='Directory to save the analysis outputs')
    args = parser.parse_args()
    
    try:
        # Collect metrics from component experiments
        df, baseline_df = collect_component_metrics(args.input_dir)
        print(f"Found data for {len(df)} component configurations")
        
        # Analyze component impact
        impact_df, component_stats = analyze_component_impact(df, baseline_df)
        
        # Generate plots with standardized styling
        plot_component_analysis(df, impact_df, args.output)
        
        # Export analysis data
        export_component_data(df, impact_df, component_stats, args.output)
        
        print("Component impact analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 