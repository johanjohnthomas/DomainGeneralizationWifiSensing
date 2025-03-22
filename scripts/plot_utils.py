#!/usr/bin/env python3
"""
Utilities for standardized plot generation across research questions.
Provides consistent color schemes, styling, and file naming conventions.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Color scheme definitions for each research question
COLOR_SCHEMES = {
    'RQ1': {
        'primary': '#1f77b4',  # Blue
        'secondary': '#aec7e8',
        'palette': sns.color_palette('Blues', n_colors=6),
        'cmap': 'Blues'
    },
    'RQ2': {
        'primary': '#9467bd',  # Purple
        'secondary': '#c5b0d5',
        'palette': sns.color_palette('Purples', n_colors=6),
        'cmap': 'Purples'
    },
    'RQ3': {
        'primary': '#d62728',  # Red
        'secondary': '#ff9896',
        'palette': sns.color_palette('RdYlBu_r', n_colors=6),
        'cmap': 'RdYlBu_r'
    },
    'RQ4': {
        'primary': '#2ca02c',  # Green
        'secondary': '#98df8a',
        'palette': sns.color_palette('Greens', n_colors=6),
        'cmap': 'Greens'
    },
    'RQ5': {
        'primary': '#ff7f0e',  # Orange
        'secondary': '#ffbb78',
        'palette': sns.color_palette('rainbow', n_colors=6),
        'cmap': 'rainbow'
    }
}


def load_metadata(experiment_dir):
    """
    Load metadata from an experiment directory
    Returns a dictionary with metadata or None if not found
    """
    metadata_file = os.path.join(experiment_dir, 'meta.json')
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
    
    return None


def get_standard_filename(rq, metric, experiment_id, extension='png'):
    """
    Generate a standardized filename for a plot
    Example: RQ1_generalization_gap_no_bedroom.png
    """
    return f"{rq}_{metric}_{experiment_id}.{extension}"


def apply_standard_style(rq, experiment_id=None):
    """
    Apply standardized styling to the current plot
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Apply RQ-specific colors
    if rq in COLOR_SCHEMES:
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLOR_SCHEMES[rq]['palette'])
    
    # Modify the title to include experiment ID if provided
    if experiment_id and plt.gca().get_title():
        current_title = plt.gca().get_title()
        if experiment_id not in current_title:
            plt.title(f"{current_title} - {experiment_id}")


def save_plot_standardized(output_dir, rq, metric, experiment_id, formats=None):
    """
    Save the current plot using standardized naming conventions
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the plot
    rq : str
        Research question (e.g., 'RQ1')
    metric : str
        Metric being plotted (e.g., 'generalization_gap')
    experiment_id : str
        Experiment identifier (e.g., 'no_bedroom')
    formats : list, optional
        List of file formats to save (default: ['png', 'pdf'])
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in formats:
        filename = get_standard_filename(rq, metric, experiment_id, fmt)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")


def get_color_for_experiment(rq, experiment_index, n_experiments=1):
    """
    Get a color from the RQ's color palette
    
    Parameters:
    -----------
    rq : str
        Research question (e.g., 'RQ1')
    experiment_index : int
        Index of the experiment (used for color variation)
    n_experiments : int
        Total number of experiments (used to create color variation)
        
    Returns:
    --------
    color : tuple
        RGB color tuple
    """
    if rq in COLOR_SCHEMES:
        palette = COLOR_SCHEMES[rq]['palette']
        
        if n_experiments > len(palette):
            # If we have more experiments than colors, create a new color palette
            palette = sns.color_palette(COLOR_SCHEMES[rq]['cmap'], n_colors=n_experiments)
        
        return palette[experiment_index % len(palette)]
    
    # Default color if RQ not defined
    return (0.5, 0.5, 0.5)


def set_rq_color_scheme(rq):
    """
    Set the appropriate color scheme for a research question
    
    Parameters:
    -----------
    rq : str
        Research question (e.g., 'RQ1')
    """
    if rq in COLOR_SCHEMES:
        sns.set_palette(COLOR_SCHEMES[rq]['palette'])
        plt.rcParams['image.cmap'] = COLOR_SCHEMES[rq]['cmap']
    else:
        # Default to a neutral color scheme
        sns.set_palette('deep')


# Helper functions for specific plot types with standardized styling
def create_barplot(x, y, rq, experiment_id=None, title=None, xlabel=None, ylabel=None):
    """Create a standardized bar plot"""
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x, y=y, color=COLOR_SCHEMES[rq]['primary'])
    
    if title:
        if experiment_id:
            plt.title(f"{title} - {experiment_id}", fontsize=16)
        else:
            plt.title(title, fontsize=16)
    
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.3f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return ax


def create_lineplot(x, y, rq, experiment_id=None, title=None, xlabel=None, ylabel=None):
    """Create a standardized line plot"""
    plt.figure(figsize=(10, 6))
    
    # If y is a dictionary, plot multiple lines
    if isinstance(y, dict):
        for i, (label, values) in enumerate(y.items()):
            color = get_color_for_experiment(rq, i, len(y))
            plt.plot(x, values, 'o-', label=label, linewidth=2, markersize=8, color=color)
        plt.legend()
    else:
        plt.plot(x, y, 'o-', linewidth=2, markersize=8, color=COLOR_SCHEMES[rq]['primary'])
    
    if title:
        if experiment_id:
            plt.title(f"{title} - {experiment_id}", fontsize=16)
        else:
            plt.title(title, fontsize=16)
    
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gca()


def create_heatmap(data, rq, experiment_id=None, title=None, xlabel=None, ylabel=None):
    """Create a standardized heatmap"""
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        data, 
        annot=True, 
        fmt='.3f', 
        cmap=COLOR_SCHEMES[rq]['cmap'],
        linewidths=0.5, 
        center=0 if np.min(data.values) < 0 else None
    )
    
    if title:
        if experiment_id:
            plt.title(f"{title} - {experiment_id}", fontsize=16)
        else:
            plt.title(title, fontsize=16)
    
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    
    plt.tight_layout()
    return ax


def create_scatterplot(x, y, rq, experiment_id=None, labels=None, title=None, xlabel=None, ylabel=None):
    """Create a standardized scatter plot"""
    plt.figure(figsize=(10, 8))
    
    ax = sns.scatterplot(
        x=x, 
        y=y, 
        s=100, 
        alpha=0.7, 
        color=COLOR_SCHEMES[rq]['primary']
    )
    
    # Add labels to points if provided
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(
                label, 
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
    
    if title:
        if experiment_id:
            plt.title(f"{title} - {experiment_id}", fontsize=16)
        else:
            plt.title(title, fontsize=16)
    
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return ax 