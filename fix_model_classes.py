#!/usr/bin/env python
"""
Fix model classes issue by ensuring the correct number of output classes based on specified activities.

This script:
1. Checks your dataset labels to ensure only the specified activities are included
2. Can regenerate filtered_activities.txt with only desired activity classes
3. Creates a modified model architecture with the exact number of output classes needed

Default is to use 4 classes (E, J, L, W).
"""

import os
import sys
import glob
import numpy as np
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
import shutil

def fix_activities_file(doppler_dir, output_file="filtered_activities.txt", allowed_activities="E,J,L,W"):
    """
    Regenerate the filtered_activities.txt file to only include specified activities.
    """
    from collections import defaultdict
    import re
    
    # Parse allowed activities
    allowed_activities = set(allowed_activities.split(','))
    print(f"Filtering for activities: {', '.join(sorted(allowed_activities))}")
    
    # Find all AR directories
    domains = []
    for item in os.listdir(doppler_dir):
        item_path = os.path.join(doppler_dir, item)
        if os.path.isdir(item_path) and item.startswith('AR'):
            domains.append(item)
    
    print(f"Processing domains: {', '.join(domains)}")
    
    # Dictionary to store the selected activity directories for each domain
    selected_dirs = {}
    
    # Process each domain
    for domain in domains:
        domain_path = os.path.join(doppler_dir, domain)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory not found: {domain_path}")
            continue
        
        # Dictionary to store activities found in this domain
        domain_activities = defaultdict(list)
        
        # Find all activity directories
        activity_pattern = re.compile(f'{domain}_(\\w+)')
        for item in os.listdir(domain_path):
            item_path = os.path.join(domain_path, item)
            if not os.path.isdir(item_path):
                continue
                
            match = activity_pattern.match(item)
            if not match:
                continue
                
            activity_with_num = match.group(1)
            # Extract base activity (first letter)
            base_activity = activity_with_num[0]
            
            # Skip if not in allowed activities
            if base_activity not in allowed_activities:
                continue
            
            # Store activity directory with its relative path
            domain_activities[base_activity].append(os.path.join(domain, item))
        
        # Select one instance of each activity type
        domain_selected = {}
        for activity, dirs in domain_activities.items():
            # Sort directories so we always select the same one (e.g., E1 before E2)
            dirs.sort()
            if dirs:
                # Select first directory for each activity type
                domain_selected[activity] = dirs[0]
                print(f"Selected for {domain} activity {activity}: {os.path.basename(dirs[0])}")
        
        selected_dirs[domain] = domain_selected
    
    # Count total selected directories
    total_selected = sum(len(activities) for activities in selected_dirs.values())
    print(f"Total selected activity directories: {total_selected}")
    
    # Write selected directories to output file
    with open(output_file, 'w') as f:
        for domain, activities in selected_dirs.items():
            for activity, dir_path in activities.items():
                f.write(f"{dir_path}\n")
    
    print(f"Filtered activity directories written to: {output_file}")
    
    # Also generate an easy-to-parse summary
    summary_file = output_file + '.summary'
    with open(summary_file, 'w') as f:
        for domain, activities in selected_dirs.items():
            f.write(f"{domain}:")
            for activity in sorted(activities.keys()):
                f.write(f" {activity}")
            f.write("\n")
    
    print(f"Summary written to: {summary_file}")
    return total_selected > 0

def fix_model(model_path, num_classes=4, save_new=True):
    """
    Fix a model to have exactly the specified number of output classes.
    """
    try:
        # Load the original model
        original_model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Original model output shape: {original_model.output_shape}")
        
        # Get the original output layer
        output_layer = original_model.layers[-1]
        original_num_classes = output_layer.output_shape[-1]
        
        if original_num_classes == num_classes:
            print(f"Model already has {num_classes} output classes, no changes needed.")
            return True, model_path
        
        # Find the layer before the output layer
        penultimate_layer = None
        for layer in original_model.layers:
            if layer.name != output_layer.name and len(layer.outbound_nodes) > 0:
                for node in layer.outbound_nodes:
                    if node.outbound_layer.name == output_layer.name:
                        penultimate_layer = layer
                        break
        
        if penultimate_layer is None:
            print("Could not find the layer before the output layer.")
            return False, None
        
        # Create a new model with the correct number of output classes
        x = penultimate_layer.output
        new_output = Dense(num_classes, activation=None, name='new_output')(x)
        new_model = Model(inputs=original_model.input, outputs=new_output)
        
        # Compile the new model with the same optimizer, loss, and metrics
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True'),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        print(f"Created new model with {num_classes} output classes")
        print(f"New model output shape: {new_model.output_shape}")
        
        # Save the new model
        if save_new:
            new_model_path = model_path.replace('.h5', f'_fixed_{num_classes}classes.h5')
            new_model.save(new_model_path)
            print(f"Saved fixed model to {new_model_path}")
            return True, new_model_path
        
        return True, new_model
        
    except Exception as e:
        print(f"Error fixing model: {e}")
        return False, None

def check_dataset_classes(doppler_dir):
    """
    Check how many classes are in the dataset files.
    """
    label_files = []
    for root, dirs, files in os.walk(doppler_dir):
        for file in files:
            if "labels_" in file and file.endswith(".txt"):
                label_files.append(os.path.join(root, file))
    
    if not label_files:
        print("No label files found in dataset directory")
        return {}
    
    print(f"Found {len(label_files)} label files")
    
    # Store activity counts per domain
    activity_counts = {}
    
    for label_file in label_files:
        try:
            with open(label_file, "rb") as fp:
                labels = pickle.load(fp)
            
            unique_labels = np.unique(labels)
            
            # Extract domain from file path
            parts = os.path.dirname(label_file).split(os.sep)
            domain = next((p for p in parts if p.startswith('AR')), "unknown")
            
            if domain not in activity_counts:
                activity_counts[domain] = set()
            
            activity_counts[domain].update(unique_labels)
            
            print(f"\nFile: {os.path.basename(label_file)}")
            print(f"  Domain: {domain}")
            print(f"  Unique labels: {unique_labels}")
            print(f"  Number of unique labels: {len(unique_labels)}")
            print(f"  Label range: {np.min(unique_labels)} to {np.max(unique_labels)}")
            
        except Exception as e:
            print(f"Error loading label file {label_file}: {e}")
    
    # Print summary
    print("\nActivity classes summary by domain:")
    for domain, activities in activity_counts.items():
        print(f"  {domain}: {sorted(activities)} ({len(activities)} classes)")
    
    return activity_counts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doppler-dir', default='./doppler_traces/', help='Directory containing doppler data')
    parser.add_argument('--model', help='Path to the model file to fix')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes to use (default: 4)')
    parser.add_argument('--activities', default='E,J,L,W', help='Comma-separated list of allowed activities')
    parser.add_argument('--output-file', default='filtered_activities.txt',
                      help='Output file for filtered activities list')
    parser.add_argument('--skip-model-fix', action='store_true', help='Skip the model fixing step')
    parser.add_argument('--skip-dataset-check', action='store_true', help='Skip dataset checking')
    args = parser.parse_args()
    
    # Step 1: Check and regenerate the filtered activities file
    if args.doppler_dir and os.path.exists(args.doppler_dir):
        print("\n===== Step 1: Checking and fixing activity filtering =====")
        success = fix_activities_file(args.doppler_dir, args.output_file, args.activities)
        if not success:
            print("No valid activity directories found. Please check your doppler directory path.")
            return 1
            
        # Make a backup of the current filtered_activities.txt before replacing it
        if os.path.exists("filtered_activities.txt") and args.output_file != "filtered_activities.txt":
            backup_name = "filtered_activities.txt.bak"
            shutil.copy2("filtered_activities.txt", backup_name)
            print(f"Backed up existing filtered_activities.txt to {backup_name}")
            # Replace the current file with our new one
            shutil.copy2(args.output_file, "filtered_activities.txt")
            print("Updated filtered_activities.txt with our new version")
    else:
        if args.doppler_dir:
            print(f"Doppler directory not found: {args.doppler_dir}")
        print("Skipping activity filtering step")
    
    # Step 2: Check dataset classes
    if args.doppler_dir and os.path.exists(args.doppler_dir) and not args.skip_dataset_check:
        print("\n===== Step 2: Checking dataset classes =====")
        activity_counts = check_dataset_classes(args.doppler_dir)
        
        # Check if any domain has more than the expected number of classes
        domains_with_issues = []
        for domain, activities in activity_counts.items():
            if len(activities) > args.num_classes:
                domains_with_issues.append(domain)
        
        if domains_with_issues:
            print(f"\nWARNING: The following domains have more than {args.num_classes} classes:")
            for domain in domains_with_issues:
                print(f"  {domain}: {sorted(activity_counts[domain])}")
            print("\nYou should regenerate your dataset with the correct activities.")
            print("Use these commands to fix the issue:")
            print(f"  make filter_activities")
            print(f"  make datasets")
    
    # Step 3: Fix the model
    if args.model and os.path.exists(args.model) and not args.skip_model_fix:
        print(f"\n===== Step 3: Fixing model to have {args.num_classes} output classes =====")
        success, new_model = fix_model(args.model, args.num_classes)
        
        if success:
            print("\nModel fixed successfully!")
            print("To use the fixed model:")
            if isinstance(new_model, str):
                print(f"1. Rename {new_model} to replace your original model file")
                print(f"2. Rerun your tests with the fixed model")
            else:
                print("1. The model was modified in memory but not saved")
        else:
            print("\nFailed to fix the model. Try these manual steps:")
            print("1. Regenerate your dataset with the correct activities")
            print("2. Retrain your model with the correct dataset")
    else:
        if args.model and not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
        elif not args.skip_model_fix:
            print("No model file specified, skipping model fix")
    
    print("\n===== Summary =====")
    print(f"1. Filtered activities: {args.activities} ({args.num_classes} classes)")
    if args.doppler_dir and os.path.exists(args.doppler_dir):
        print(f"2. Dataset checked: {args.doppler_dir}")
    if args.model and os.path.exists(args.model) and not args.skip_model_fix:
        print(f"3. Model fixed: {args.model}")
    
    print("\nRecommended next steps:")
    print("1. Run 'make filter_activities' to regenerate the filtered_activities.txt file")
    print("2. Run 'make datasets' to recreate the datasets with the correct activities")
    print("3. Run 'make train' to retrain your model with the correct number of classes")
    print("4. Run 'make test' to test your model")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 