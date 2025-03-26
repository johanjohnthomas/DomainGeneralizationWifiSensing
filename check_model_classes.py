#!/usr/bin/env python
"""
Check model output classes and dataset classes to diagnose class mismatch issues.
This script helps verify that models and datasets use the correct number of classes based on
the specified activities (default: E,J,L,W).
"""

import os
import sys
import numpy as np
import argparse
import pickle
import tensorflow as tf
import glob

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Path to the model file (.h5)', required=False)
    parser.add_argument('--dataset', help='Path to the dataset directory', required=False) 
    parser.add_argument('--metrics', help='Path to a metrics file', required=False)
    parser.add_argument('--activities', default='E,J,L,W', help='Comma-separated list of activities to check for')
    args = parser.parse_args()
    
    # Calculate expected number of classes
    activities = args.activities.split(',')
    expected_classes = len(activities)
    
    print(f"Checking for {expected_classes} classes based on activities: {args.activities}")

    # Check model if provided
    if args.model and os.path.exists(args.model):
        print(f"\n===== Checking model: {args.model} =====")
        try:
            model = tf.keras.models.load_model(args.model)
            print("Model loaded successfully")
            
            # Get the output shape from the last layer
            output_layer = model.layers[-1]
            num_classes = output_layer.output_shape[-1]
            print(f"Number of output classes in the model: {num_classes}")
            
            # Print model summary
            print("\nModel Summary:")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        if args.model:
            print(f"Model file not found: {args.model}")
        
        # Try to find models automatically
        models = glob.glob("*.h5")
        if models:
            print(f"\nFound {len(models)} model files. Use --model to specify one:")
            for model in models:
                print(f"  {model}")
    
    # Check dataset if provided
    if args.dataset and os.path.exists(args.dataset):
        print(f"\n===== Checking dataset directory: {args.dataset} =====")
        
        label_files = []
        for root, dirs, files in os.walk(args.dataset):
            for file in files:
                if "labels_" in file and file.endswith(".txt"):
                    label_files.append(os.path.join(root, file))
        
        if label_files:
            print(f"Found {len(label_files)} label files")
            for label_file in label_files[:5]:  # Show only first 5
                try:
                    with open(label_file, "rb") as fp:
                        labels = pickle.load(fp)
                    unique_labels = np.unique(labels)
                    print(f"\nFile: {os.path.basename(label_file)}")
                    print(f"  Unique labels: {unique_labels}")
                    print(f"  Number of unique labels: {len(unique_labels)}")
                    print(f"  Label range: {np.min(unique_labels)} to {np.max(unique_labels)}")
                except Exception as e:
                    print(f"Error loading label file {label_file}: {e}")
        else:
            print("No label files found in dataset directory")
    else:
        if args.dataset:
            print(f"Dataset directory not found: {args.dataset}")
    
    # Check metrics file if provided
    if args.metrics and os.path.exists(args.metrics):
        print(f"\n===== Checking metrics file: {args.metrics} =====")
        try:
            with open(args.metrics, "rb") as fp:
                metrics_dict = pickle.load(fp)
            
            # Check confusion matrix dimensions
            if 'conf_matrix' in metrics_dict:
                conf_matrix = metrics_dict['conf_matrix']
                print(f"Confusion matrix shape: {conf_matrix.shape}")
                print(f"Number of classes (based on confusion matrix): {conf_matrix.shape[0]}")
            
            if 'conf_matrix_max_merge' in metrics_dict:
                conf_matrix_merge = metrics_dict['conf_matrix_max_merge']
                print(f"Merged confusion matrix shape: {conf_matrix_merge.shape}")
            
            # Check precision/recall vectors
            if 'precision_single' in metrics_dict:
                precision = metrics_dict['precision_single']
                print(f"Precision vector length: {len(precision)}")
            
        except Exception as e:
            print(f"Error loading metrics file: {e}")
    else:
        if args.metrics:
            print(f"Metrics file not found: {args.metrics}")
        
        # Try to find metrics files automatically
        metrics_files = glob.glob("./results/complete_different_*.txt")
        if not metrics_files:
            metrics_files = glob.glob("./outputs/complete_different_*.txt")
        
        if metrics_files:
            print(f"\nFound {len(metrics_files)} metrics files. Use --metrics to specify one:")
            for metrics_file in metrics_files[:5]:  # Show only first 5
                print(f"  {metrics_file}")

    print("\n===== Done =====")

if __name__ == "__main__":
    main() 