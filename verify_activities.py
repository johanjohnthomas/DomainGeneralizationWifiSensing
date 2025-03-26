#!/usr/bin/env python
"""
Verify that a model has the expected number of output classes based on the specified activities
and that dataset files only contain the corresponding labels.

Default is to check for 4 classes (E,J,L,W) with labels 0-3.
"""

import os
import sys
import glob
import numpy as np
import argparse
import pickle
import tensorflow as tf

def check_model(model_path, expected_classes=4):
    """Check that a model has the expected number of output classes"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Successfully loaded model: {model_path}")
        
        # Get output shape from the last layer
        output_layer = model.layers[-1]
        num_classes = output_layer.output_shape[-1]
        print(f"  Number of output classes: {num_classes}")
        
        if num_classes == expected_classes:
            print(f"✓ Model has the correct number of classes ({expected_classes})")
            return True
        else:
            print(f"✗ Model has {num_classes} classes instead of {expected_classes}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def check_dataset_labels(dataset_dir, expected_max_label=3):
    """Check that dataset files only contain the expected labels"""
    max_found_label = -1
    unique_labels = set()
    
    label_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.startswith("labels_") and file.endswith(".txt"):
                label_files.append(os.path.join(root, file))
    
    if not label_files:
        print(f"✗ No label files found in {dataset_dir}")
        return False
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        try:
            with open(label_file, "rb") as fp:
                labels = pickle.load(fp)
            
            file_unique_labels = set(np.unique(labels))
            unique_labels.update(file_unique_labels)
            
            file_max_label = np.max(labels)
            if file_max_label > max_found_label:
                max_found_label = file_max_label
        except Exception as e:
            print(f"✗ Error reading label file {label_file}: {e}")
    
    if max_found_label == expected_max_label:
        print(f"✓ Maximum label found is {expected_max_label} (as expected for {expected_max_label+1} classes)")
        print(f"  Unique labels found: {sorted(unique_labels)}")
        return True
    else:
        print(f"✗ Maximum label found is {max_found_label} (expected {expected_max_label} for {expected_max_label+1} classes)")
        print(f"  Unique labels found: {sorted(unique_labels)}")
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Path to the model file (.h5)")
    parser.add_argument("--dataset", help="Path to the dataset directory")
    parser.add_argument("--activities", default="E,J,L,W", 
                      help="Comma-separated list of activities to verify (default: E,J,L,W)")
    args = parser.parse_args()
    
    # Calculate expected number of classes from activities
    activities = args.activities.split(',')
    expected_classes = len(activities)
    expected_max_label = expected_classes - 1
    
    print(f"Verifying model for activities: {args.activities}")
    print(f"Expected number of classes: {expected_classes}")
    print(f"Expected maximum label value: {expected_max_label}")
    
    success = True
    
    # Check for models if none specified
    if not args.model:
        model_files = glob.glob("*.h5")
        if model_files:
            print(f"Found {len(model_files)} model files. You can specify one with --model")
            for i, model_file in enumerate(model_files):
                print(f"Model {i+1}: {model_file}")
                if check_model(model_file, expected_classes) == False:
                    success = False
        else:
            print("No model files found. Specify one with --model")
    else:
        if not check_model(args.model, expected_classes):
            success = False
    
    # Check dataset if provided
    if args.dataset:
        print(f"\nChecking dataset directory: {args.dataset}")
        if not check_dataset_labels(args.dataset, expected_max_label):
            success = False
    
    if success:
        print(f"\n✓ All checks passed. Your model and datasets appear to be using the correct {expected_classes} activities.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 