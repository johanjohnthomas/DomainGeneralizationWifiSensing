#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Final Report
=====================
This script consolidates results from all research questions into a final report.
It extracts key metrics and visualizations from each RQ directory and combines
them into a single comprehensive PDF report.

Usage:
    python generate_final_report.py --rq-dirs <rq1_dir> <rq3_dir> <rq4_dir> <rq5_dir> --output-dir <output_dir>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import glob
import json
import re
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate consolidated report from all research questions')
    parser.add_argument('--rq-dirs', nargs='+', required=True,
                      help='Directories containing results for each research question')
    parser.add_argument('--output-dir', required=True,
                      help='Directory to save the final report')
    return parser.parse_args()

def collect_rq1_results(rq1_dir):
    """Collect generalization gap analysis results from RQ1."""
    results = {}
    
    # Get generalization gap metrics
    gap_data_path = os.path.join(rq1_dir, 'generalization_gap', 'generalization_metrics.csv')
    if os.path.exists(gap_data_path):
        results['gap_metrics'] = pd.read_csv(gap_data_path)
    
    # Get visualization paths
    viz_pattern = os.path.join(rq1_dir, 'generalization_gap', '*.png')
    results['visualizations'] = glob.glob(viz_pattern)
    
    return results

def collect_rq3_results(rq3_dir):
    """Collect accuracy vs. generalizability tradeoff results from RQ3."""
    results = {}
    
    # Get tradeoff metrics
    metrics_pattern = os.path.join(rq3_dir, 'accuracy_vs_gap', '*.csv')
    csv_files = glob.glob(metrics_pattern)
    
    for csv_file in csv_files:
        key = os.path.basename(csv_file).replace('.csv', '')
        if os.path.exists(csv_file):
            results[key] = pd.read_csv(csv_file)
    
    # Get visualization paths
    viz_pattern = os.path.join(rq3_dir, 'accuracy_vs_gap', '*.png')
    results['visualizations'] = glob.glob(viz_pattern)
    
    return results

def collect_rq4_results(rq4_dir):
    """Collect source scaling results from RQ4."""
    results = {}
    
    # Get scaling metrics if available
    scaling_data_path = os.path.join(rq4_dir, 'scaling_plots', 'source_scaling_metrics.csv')
    if os.path.exists(scaling_data_path):
        results['scaling_metrics'] = pd.read_csv(scaling_data_path)
    
    # Get visualization paths
    viz_pattern = os.path.join(rq4_dir, 'scaling_plots', '*.png')
    results['visualizations'] = glob.glob(viz_pattern)
    
    return results

def collect_rq5_results(rq5_dir):
    """Collect component impact analysis results from RQ5."""
    results = {}
    
    # Get component impact metrics
    component_data_path = os.path.join(rq5_dir, 'contribution_plots', 'component_impact.csv')
    if os.path.exists(component_data_path):
        results['component_metrics'] = pd.read_csv(component_data_path)
    
    # Get visualization paths
    viz_pattern = os.path.join(rq5_dir, 'contribution_plots', '*.png')
    results['visualizations'] = glob.glob(viz_pattern)
    
    return results

def generate_report_title_page(pdf, rq_dirs):
    """Generate the title page for the report."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.8, 'Domain Generalization for WiFi Sensing', 
             horizontalalignment='center', fontsize=24, weight='bold')
    
    # Subtitle
    plt.text(0.5, 0.7, 'Consolidated Research Findings', 
             horizontalalignment='center', fontsize=20)
    
    # Date
    plt.text(0.5, 0.6, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}',
             horizontalalignment='center', fontsize=14)
    
    # RQ directories included
    plt.text(0.5, 0.5, 'Research Questions Included:', 
             horizontalalignment='center', fontsize=16, weight='bold')
    
    y_pos = 0.45
    for dir_path in rq_dirs:
        rq_name = os.path.basename(dir_path)
        plt.text(0.5, y_pos, rq_name, horizontalalignment='center', fontsize=14)
        y_pos -= 0.05
    
    # Save to PDF
    pdf.savefig()
    plt.close()

def generate_rq1_summary(pdf, rq1_results):
    """Generate summary page for RQ1 findings."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'RQ1: Generalization Gap Analysis', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    # Description
    plt.text(0.5, 0.9, 'Analysis of model generalization across different domains',
             horizontalalignment='center', fontsize=16)
    
    # Add metrics table if available
    if 'gap_metrics' in rq1_results and not rq1_results['gap_metrics'].empty:
        metrics_df = rq1_results['gap_metrics']
        
        # Create a table with key metrics
        table_data = []
        columns = ['Domain', 'Train Acc.', 'Test Acc.', 'Gap']
        
        for _, row in metrics_df.iterrows():
            if 'target_domain' in row and 'train_accuracy' in row and 'test_accuracy' in row:
                domain = row['target_domain']
                train_acc = f"{row['train_accuracy']:.2f}%"
                test_acc = f"{row['test_accuracy']:.2f}%"
                gap = f"{row['generalization_gap']:.2f}%"
                table_data.append([domain, train_acc, test_acc, gap])
        
        if table_data:
            ax = plt.subplot(212)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns, 
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
    
    # Save to PDF
    pdf.savefig()
    plt.close()
    
    # Add visualizations if available
    for viz_path in rq1_results.get('visualizations', []):
        if os.path.exists(viz_path) and viz_path.endswith(('.png', '.jpg')):
            try:
                img = plt.imread(viz_path)
                plt.figure(figsize=(12, 9))
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(viz_path), fontsize=14)
                pdf.savefig()
                plt.close()
            except Exception as e:
                print(f"Error including visualization {viz_path}: {e}")

def generate_rq3_summary(pdf, rq3_results):
    """Generate summary page for RQ3 findings."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'RQ3: Accuracy vs. Generalizability Tradeoff', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    # Description
    plt.text(0.5, 0.9, 'Analysis of tradeoffs between model accuracy and generalization capability',
             horizontalalignment='center', fontsize=16)
    
    # Add key statistics if available
    y_pos = 0.8
    if 'correlations' in rq3_results and not rq3_results['correlations'].empty:
        plt.text(0.5, y_pos, 'Key Correlation Findings:', 
                 horizontalalignment='center', fontsize=16, weight='bold')
        y_pos -= 0.05
        
        for _, row in rq3_results['correlations'].iterrows():
            if 'metric_1' in row and 'metric_2' in row and 'correlation' in row:
                correlation_text = f"{row['metric_1']} vs {row['metric_2']}: r = {row['correlation']:.2f}"
                plt.text(0.5, y_pos, correlation_text, horizontalalignment='center', fontsize=14)
                y_pos -= 0.04
    
    # Save to PDF
    pdf.savefig()
    plt.close()
    
    # Add visualizations if available
    for viz_path in rq3_results.get('visualizations', []):
        if os.path.exists(viz_path) and viz_path.endswith(('.png', '.jpg')):
            try:
                img = plt.imread(viz_path)
                plt.figure(figsize=(12, 9))
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(viz_path), fontsize=14)
                pdf.savefig()
                plt.close()
            except Exception as e:
                print(f"Error including visualization {viz_path}: {e}")

def generate_rq4_summary(pdf, rq4_results):
    """Generate summary page for RQ4 findings."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'RQ4: Source Domain Scaling Analysis', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    # Description
    plt.text(0.5, 0.9, 'Analysis of model performance scaling with number of source domains',
             horizontalalignment='center', fontsize=16)
    
    # Add metrics table if available
    if 'scaling_metrics' in rq4_results and not rq4_results['scaling_metrics'].empty:
        metrics_df = rq4_results['scaling_metrics']
        
        # Create a table with key metrics
        table_data = []
        columns = ['Sources', 'Train Acc.', 'Test Acc.', 'Gap']
        
        for _, row in metrics_df.iterrows():
            if 'num_sources' in row and 'train_accuracy' in row and 'test_accuracy' in row:
                sources = str(row['num_sources'])
                train_acc = f"{row['train_accuracy']:.2f}%"
                test_acc = f"{row['test_accuracy']:.2f}%"
                gap = f"{row['generalization_gap']:.2f}%"
                table_data.append([sources, train_acc, test_acc, gap])
        
        if table_data:
            ax = plt.subplot(212)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns, 
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
    
    # Save to PDF
    pdf.savefig()
    plt.close()
    
    # Add visualizations if available
    for viz_path in rq4_results.get('visualizations', []):
        if os.path.exists(viz_path) and viz_path.endswith(('.png', '.jpg')):
            try:
                img = plt.imread(viz_path)
                plt.figure(figsize=(12, 9))
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(viz_path), fontsize=14)
                pdf.savefig()
                plt.close()
            except Exception as e:
                print(f"Error including visualization {viz_path}: {e}")

def generate_rq5_summary(pdf, rq5_results):
    """Generate summary page for RQ5 findings."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'RQ5: Component Impact Analysis', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    # Description
    plt.text(0.5, 0.9, 'Analysis of the impact of different architectural components on domain generalization',
             horizontalalignment='center', fontsize=16)
    
    # Add metrics table if available
    if 'component_metrics' in rq5_results and not rq5_results['component_metrics'].empty:
        metrics_df = rq5_results['component_metrics']
        
        # Create a table with key metrics
        table_data = []
        columns = ['Component', 'Impact on Acc.', 'Impact on Gen.']
        
        for _, row in metrics_df.iterrows():
            if 'component' in row:
                component = row['component']
                acc_impact = f"{row.get('accuracy_impact', 'N/A')}"
                if acc_impact != 'N/A':
                    acc_impact = f"{float(acc_impact):.2f}%"
                
                gen_impact = f"{row.get('generalization_impact', 'N/A')}"
                if gen_impact != 'N/A':
                    gen_impact = f"{float(gen_impact):.2f}%"
                
                table_data.append([component, acc_impact, gen_impact])
        
        if table_data:
            ax = plt.subplot(212)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns, 
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
    
    # Save to PDF
    pdf.savefig()
    plt.close()
    
    # Add visualizations if available
    for viz_path in rq5_results.get('visualizations', []):
        if os.path.exists(viz_path) and viz_path.endswith(('.png', '.jpg')):
            try:
                img = plt.imread(viz_path)
                plt.figure(figsize=(12, 9))
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(viz_path), fontsize=14)
                pdf.savefig()
                plt.close()
            except Exception as e:
                print(f"Error including visualization {viz_path}: {e}")

def generate_conclusion_page(pdf, all_results):
    """Generate the conclusion page for the report."""
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, 'Conclusions and Key Findings', 
             horizontalalignment='center', fontsize=20, weight='bold')
    
    # Key findings from each RQ
    y_pos = 0.85
    
    # RQ1 findings
    plt.text(0.5, y_pos, 'RQ1: Generalization Gap Analysis', 
             horizontalalignment='center', fontsize=16, weight='bold')
    y_pos -= 0.05
    if 'rq1' in all_results and 'gap_metrics' in all_results['rq1']:
        gap_metrics = all_results['rq1']['gap_metrics']
        if not gap_metrics.empty and 'generalization_gap' in gap_metrics.columns:
            avg_gap = gap_metrics['generalization_gap'].mean()
            worst_gap = gap_metrics['generalization_gap'].max()
            worst_domain = gap_metrics.loc[gap_metrics['generalization_gap'].idxmax()].get('target_domain', 'Unknown')
            
            plt.text(0.5, y_pos, f"Average Generalization Gap: {avg_gap:.2f}%", 
                     horizontalalignment='center', fontsize=14)
            y_pos -= 0.04
            plt.text(0.5, y_pos, f"Worst Generalization Domain: {worst_domain} (Gap: {worst_gap:.2f}%)", 
                     horizontalalignment='center', fontsize=14)
            y_pos -= 0.04
    
    y_pos -= 0.04
    # RQ3 findings
    plt.text(0.5, y_pos, 'RQ3: Accuracy vs. Generalizability Tradeoff', 
             horizontalalignment='center', fontsize=16, weight='bold')
    y_pos -= 0.05
    if 'rq3' in all_results and 'correlations' in all_results['rq3']:
        correlations = all_results['rq3']['correlations']
        if not correlations.empty and 'correlation' in correlations.columns:
            plt.text(0.5, y_pos, "Key correlations between model metrics highlight tradeoffs", 
                     horizontalalignment='center', fontsize=14)
            y_pos -= 0.04
    
    y_pos -= 0.04
    # RQ4 findings
    plt.text(0.5, y_pos, 'RQ4: Source Domain Scaling Analysis', 
             horizontalalignment='center', fontsize=16, weight='bold')
    y_pos -= 0.05
    if 'rq4' in all_results and 'scaling_metrics' in all_results['rq4']:
        scaling_metrics = all_results['rq4']['scaling_metrics']
        if not scaling_metrics.empty and 'num_sources' in scaling_metrics.columns and 'test_accuracy' in scaling_metrics.columns:
            scaling_effect = "positive" if scaling_metrics.sort_values('num_sources')['test_accuracy'].corr(scaling_metrics.sort_values('num_sources')['num_sources']) > 0 else "negative"
            plt.text(0.5, y_pos, f"Source domain scaling shows a {scaling_effect} effect on generalization", 
                     horizontalalignment='center', fontsize=14)
            y_pos -= 0.04
    
    y_pos -= 0.04
    # RQ5 findings
    plt.text(0.5, y_pos, 'RQ5: Component Impact Analysis', 
             horizontalalignment='center', fontsize=16, weight='bold')
    y_pos -= 0.05
    if 'rq5' in all_results and 'component_metrics' in all_results['rq5']:
        component_metrics = all_results['rq5']['component_metrics']
        if not component_metrics.empty:
            plt.text(0.5, y_pos, "Component analysis reveals the importance of architectural choices", 
                     horizontalalignment='center', fontsize=14)
            y_pos -= 0.04
    
    # Overall conclusion
    y_pos -= 0.08
    plt.text(0.5, y_pos, 'Overall Conclusion', 
             horizontalalignment='center', fontsize=16, weight='bold')
    y_pos -= 0.05
    
    conclusion_text = (
        "This consolidated analysis demonstrates the factors affecting domain generalization " 
        "in WiFi sensing, including environmental factors, model architecture, and training strategies. "
        "The results provide insights into creating more robust and generalizable WiFi sensing systems."
    )
    
    # Wrap text to fit in the page
    text_obj = plt.text(0.5, y_pos, conclusion_text, 
                horizontalalignment='center', fontsize=14, wrap=True)
    text_obj.set_path_effects([])
    
    # Save to PDF
    pdf.savefig()
    plt.close()

def main():
    """Main function to generate the report."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dictionary
    all_results = {}
    
    # Collect results from each RQ
    for rq_dir in args.rq_dirs:
        if not os.path.isdir(rq_dir):
            print(f"Warning: {rq_dir} is not a valid directory. Skipping.")
            continue
            
        rq_name = os.path.basename(rq_dir)
        
        if "RQ1" in rq_name:
            all_results['rq1'] = collect_rq1_results(rq_dir)
        elif "RQ3" in rq_name:
            all_results['rq3'] = collect_rq3_results(rq_dir)
        elif "RQ4" in rq_name:
            all_results['rq4'] = collect_rq4_results(rq_dir)
        elif "RQ5" in rq_name:
            all_results['rq5'] = collect_rq5_results(rq_dir)
    
    # Generate the PDF report
    output_pdf = os.path.join(args.output_dir, 'final_report.pdf')
    
    with PdfPages(output_pdf) as pdf:
        # Generate title page
        generate_report_title_page(pdf, args.rq_dirs)
        
        # Generate RQ specific pages
        if 'rq1' in all_results:
            generate_rq1_summary(pdf, all_results['rq1'])
        
        if 'rq3' in all_results:
            generate_rq3_summary(pdf, all_results['rq3'])
        
        if 'rq4' in all_results:
            generate_rq4_summary(pdf, all_results['rq4'])
        
        if 'rq5' in all_results:
            generate_rq5_summary(pdf, all_results['rq5'])
        
        # Generate conclusion page
        generate_conclusion_page(pdf, all_results)
    
    print(f"Final report generated: {output_pdf}")
    
    # Also generate a CSV summary
    summary_csv = os.path.join(args.output_dir, 'results_summary.csv')
    
    summary_data = []
    if 'rq1' in all_results and 'gap_metrics' in all_results['rq1'] and not all_results['rq1']['gap_metrics'].empty:
        rq1_metrics = all_results['rq1']['gap_metrics']
        summary_data.append({
            'RQ': 'RQ1',
            'Description': 'Generalization Gap Analysis',
            'Avg_Train_Accuracy': rq1_metrics.get('train_accuracy', pd.Series()).mean() if 'train_accuracy' in rq1_metrics else 'N/A',
            'Avg_Test_Accuracy': rq1_metrics.get('test_accuracy', pd.Series()).mean() if 'test_accuracy' in rq1_metrics else 'N/A',
            'Avg_Generalization_Gap': rq1_metrics.get('generalization_gap', pd.Series()).mean() if 'generalization_gap' in rq1_metrics else 'N/A'
        })
    
    # Add other RQs to the summary as needed
    
    if summary_data:
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
        print(f"Results summary CSV generated: {summary_csv}")

if __name__ == "__main__":
    main() 