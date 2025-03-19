"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
from dataset_utility import (create_dataset, create_dataset_randomized_antennas, 
                              create_dataset_single, create_dataset_multi_channel,
                              expand_antennas, convert_to_number, convert_to_grouped_number, 
                              get_label_mappings)
from network_utility import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import glob
import gc
import shutil
import hashlib
import sys
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import datetime
import traceback
import json
import re

# Added imports for class imbalance handling
from imblearn.over_sampling import SMOTE

"""
IMPORTANT ARCHITECTURAL NOTE:

This script has been updated to use a multi-channel approach for processing CSI data.
The key architectural changes are:

1. The model now takes inputs with shape (340, 100, 4), where:
   - 340: Feature dimension (height)
   - 100: Time dimension (width)
   - 4: All 4 antennas processed together as channels

2. The data loading pipeline has been modified to:
   - Process all 4 antennas together for each sample
   - Transpose the data from (4, 100, 340) to (340, 100, 4)
   - This allows the model to leverage cross-antenna correlations

3. Evaluation is performed on original samples, not expanded by antenna

This approach significantly improves classification accuracy by utilizing
all antenna data simultaneously instead of processing each antenna separately.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # For reproducibility
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better GPU mem management
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# Now import TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def compute_class_weights(labels, num_classes):
    """
    Compute class weights for imbalanced dataset using inverse frequency with smoothing.
    This approach puts more weight on underrepresented classes to help balance training.
    
    Args:
        labels: Array of class labels
        num_classes: Total number of possible classes
        
    Returns:
        dict: Dictionary mapping class indices to their weights
    """
    # Get the unique classes that actually appear in the data
    unique_classes = np.unique(labels)
    
    # Use inverse frequency with smoothing
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Ensure class_counts has exactly num_classes elements
    if len(class_counts) < num_classes:
        # Pad with zeros if necessary
        class_counts = np.pad(class_counts, (0, num_classes - len(class_counts)), 'constant')
    elif len(class_counts) > num_classes:
        # Truncate if necessary (should not happen with proper minlength)
        class_counts = class_counts[:num_classes]
        
    # Calculate weights with smoothing factor to prevent division by zero
    weights = 1. / (class_counts + 0.1 * np.max(class_counts))
    
    # Normalize weights to prevent extremely large values
    weights = weights / np.sum(weights) * num_classes
    
    # Create a dictionary to store the weights
    class_weights = {i: weights[i] for i in range(num_classes)}
    
    # Log the weights and counts for transparency
    print("Class weights based on inverse frequency with smoothing:")
    for idx in range(num_classes):
        if idx in unique_classes:
            print(f"  Class {idx}: {class_weights[idx]:.4f} (count: {class_counts[idx]})")
        else:
            # Add default weight for any missing class
            class_weights[idx] = 1.0
            print(f"  Class {idx}: {class_weights[idx]:.4f} (default - class not found in training data)")
            
    return class_weights

def create_model(input_shape=(100, 100, 4), num_classes=6):
    """Create the CSI network model."""
    input_network = tf.keras.layers.Input(shape=input_shape)
    
    # Add L2 regularization to all convolutional layers - increased from 0.001 to 0.01
    regularizer = tf.keras.regularizers.l2(0.01)
    
    # Add initial dropout layer to reduce overfitting
    x = tf.keras.layers.Dropout(0.3, name='input_dropout')(input_network)
    
    # First branch - 3x3 convolutions
    conv3_1 = tf.keras.layers.Conv2D(3, (3, 3), padding='same', 
                                    kernel_regularizer=regularizer,
                                    name='1stconv3_1_res_a')(x)  # Now use x instead of input_network
    conv3_1 = tf.keras.layers.BatchNormalization()(conv3_1)
    conv3_1 = tf.keras.layers.Activation('relu', name='activation_1')(conv3_1)
    conv3_2 = tf.keras.layers.Conv2D(6, (3, 3), padding='same', 
                                    kernel_regularizer=regularizer,
                                    name='1stconv3_2_res_a')(conv3_1)
    conv3_2 = tf.keras.layers.BatchNormalization()(conv3_2)
    conv3_2 = tf.keras.layers.Activation('relu', name='activation_2')(conv3_2)
    conv3_3 = tf.keras.layers.Conv2D(9, (3, 3), strides=(2, 2), padding='same', 
                                    kernel_regularizer=regularizer,
                                    name='1stconv3_3_res_a')(conv3_2)
    conv3_3 = tf.keras.layers.BatchNormalization()(conv3_3)
    conv3_3 = tf.keras.layers.Activation('relu', name='activation_3')(conv3_3)
    
    # Second branch - 2x2 convolutions
    conv2_1 = tf.keras.layers.Conv2D(5, (2, 2), strides=(2, 2), padding='same', 
                                    kernel_regularizer=regularizer,
                                    name='1stconv2_1_res_a')(x)  # Now use x instead of input_network
    conv2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
    conv2_1 = tf.keras.layers.Activation('relu', name='activation')(conv2_1)
    
    # Third branch - max pooling
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)  # Now use x instead of input_network
    
    # Concatenate all branches
    concat = tf.keras.layers.Concatenate(name='concatenate')([pool1, conv2_1, conv3_3])
    
    # Additional convolution
    conv4 = tf.keras.layers.Conv2D(3, (1, 1), 
                                 kernel_regularizer=regularizer,
                                 name='conv4')(concat)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu', name='activation_4')(conv4)
    
    # Flatten and dense layers
    flat = tf.keras.layers.Flatten(name='flatten')(conv4)
    
    # Increase dropout rate from 0.6 to 0.7 for better regularization
    drop = tf.keras.layers.Dropout(0.7, name='dropout')(flat)
    
    # Add regularization to the final dense layer
    dense2 = tf.keras.layers.Dense(num_classes, 
                                 kernel_regularizer=regularizer,
                                 name='dense2')(drop)
    
    # Create model
    model = tf.keras.Model(inputs=input_network, outputs=dense2, name='csi_model')
    return model

# Add a function to validate labels and files for each domain
def validate_labels_files(dir_path, subdirs, activity_list, suffix='.txt'):
    """
    Validate that the labels and files in each domain match properly.
    
    Args:
        dir_path: Base directory path
        subdirs: List of subdirectory names
        activity_list: String with activity names
        suffix: File suffix for the labels and files lists
        
    Returns:
        Dictionary with validation results for each domain
    """
    validation_results = {}
    
    for subdir in subdirs.split(','):
        subdir_results = {'train': {}, 'val': {}, 'test': {}}
        
        # Check training files and labels
        label_path = f"{dir_path}{subdir}/labels_train_antennas_{activity_list}{suffix}"
        file_path = f"{dir_path}{subdir}/files_train_antennas_{activity_list}{suffix}"
        
        try:
            with open(label_path, "rb") as fp:
                train_labels = pickle.load(fp)
            with open(file_path, "rb") as fp:
                train_files = pickle.load(fp)
                
            subdir_results['train'] = {
                'labels_count': len(train_labels),
                'files_count': len(train_files),
                'match': len(train_labels) == len(train_files) or len(train_labels) == 1,
                'unique_labels': np.unique(train_labels).tolist()
            }
        except Exception as e:
            subdir_results['train'] = {'error': str(e)}
            
        # Check validation files and labels
        label_path = f"{dir_path}{subdir}/labels_val_antennas_{activity_list}{suffix}"
        file_path = f"{dir_path}{subdir}/files_val_antennas_{activity_list}{suffix}"
        
        try:
            with open(label_path, "rb") as fp:
                val_labels = pickle.load(fp)
            with open(file_path, "rb") as fp:
                val_files = pickle.load(fp)
                
            subdir_results['val'] = {
                'labels_count': len(val_labels),
                'files_count': len(val_files),
                'match': len(val_labels) == len(val_files) or len(val_labels) == 1,
                'unique_labels': np.unique(val_labels).tolist()
            }
        except Exception as e:
            subdir_results['val'] = {'error': str(e)}
            
        # Check test files and labels
        label_path = f"{dir_path}{subdir}/labels_test_antennas_{activity_list}{suffix}"
        file_path = f"{dir_path}{subdir}/files_test_antennas_{activity_list}{suffix}"
        
        try:
            with open(label_path, "rb") as fp:
                test_labels = pickle.load(fp)
            with open(file_path, "rb") as fp:
                test_files = pickle.load(fp)
                
            subdir_results['test'] = {
                'labels_count': len(test_labels),
                'files_count': len(test_files),
                'match': len(test_labels) == len(test_files) or len(test_labels) == 1,
                'unique_labels': np.unique(test_labels).tolist()
            }
        except Exception as e:
            subdir_results['test'] = {'error': str(e)}
            
        validation_results[subdir] = subdir_results
    
    return validation_results

def verify_label_file_integrity(files, labels, dataset_name="Dataset"):
    """
    Verify the integrity of the file-label assignments.
    
    Args:
        files: List of file paths
        labels: List of corresponding labels
        dataset_name: Name of the dataset for reporting
        
    Returns:
        Boolean indicating if the integrity check passed
    """
    if len(files) != len(labels):
        print(f"ERROR: {dataset_name} has {len(files)} files but {len(labels)} labels")
        return False
    
    # Check for None or invalid values in files or labels
    none_files = sum(1 for f in files if f is None)
    none_labels = sum(1 for l in labels if l is None)
    
    if none_files > 0:
        print(f"WARNING: {dataset_name} has {none_files} None file paths")
    
    if none_labels > 0:
        print(f"WARNING: {dataset_name} has {none_labels} None labels")
    
    # Check if all files exist
    missing_files = sum(1 for f in files if not os.path.exists(f))
    if missing_files > 0:
        print(f"WARNING: {dataset_name} has {missing_files} missing files")
    
    return none_files == 0 and none_labels == 0 and missing_files == 0

# Focal Loss implementation for handling class imbalance
@tf.keras.saving.register_keras_serializable(package="CustomLosses")
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation for class imbalance with sparse categorical inputs.
    
    Args:
        gamma: Focusing parameter for modulating loss for hard-to-classify examples
        alpha: Weighting factor for rare class samples
    
    Returns:
        loss_fn: Focal loss function compatible with Keras
    """
    def sparse_categorical_focal_loss(y_true, y_pred):
        # Get the number of classes from the prediction shape
        num_classes = tf.shape(y_pred)[-1]
        
        # Convert predictions to probabilities if logits=True was used
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])  # Flatten to 1D
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        
        # Clip predictions to avoid numerical issues
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Apply the focusing term
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes and then take mean over batch
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    # Name the inner function to match what the serialization system expects
    sparse_categorical_focal_loss.__name__ = 'sparse_categorical_focal_loss'
    
    return sparse_categorical_focal_loss

# Function to apply SMOTE for training data resampling
def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to oversample minority classes in training data.
    
    Args:
        X: Training features (flattened for SMOTE)
        y: Training labels
        random_state: Random seed for reproducibility
    
    Returns:
        X_resampled, y_resampled: Resampled data with balanced class distribution
    """
    print("Applying SMOTE to balance class distribution...")
    try:
        # Check if we have any data
        if len(X) == 0 or len(y) == 0:
            print("ERROR: Empty dataset provided to SMOTE. Cannot proceed.")
            return X, y
            
        # Get unique classes to check if we have enough samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("ERROR: Need at least 2 classes for SMOTE. Only found:", unique_classes)
            return X, y
            
        # Count samples per class to ensure we have enough for SMOTE
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        min_samples = min(class_counts.values())
        if min_samples < 5:  # SMOTE typically needs at least a few samples per class
            print(f"WARNING: Not enough samples for some classes. Minimum is {min_samples}.")
            print("Class counts:", class_counts)
            # Try with k_neighbors=min_samples-1 if possible
            if min_samples >= 2:
                print(f"Attempting SMOTE with reduced k_neighbors={min_samples-1}")
                k = min(min_samples - 1, 5)  # Maximum of 5 neighbors
                smote = SMOTE(random_state=random_state, k_neighbors=k)
            else:
                print("Cannot apply SMOTE with only 1 sample per class.")
                return X, y
        else:
            # Standard SMOTE
            smote = SMOTE(random_state=random_state)
        
        # Get original shape information before SMOTE
        print(f"Original data shape: {X.shape}, labels shape: {y.shape}")
        
        # Print class distribution before SMOTE
        unique_labels, counts_before = np.unique(y, return_counts=True)
        print("Class distribution before SMOTE:")
        for label, count in zip(unique_labels, counts_before):
            print(f"  Class {label}: {count} samples")
        
        # Apply SMOTE - make sure y is squeezed to avoid dimension mismatch
        y = np.squeeze(y)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Print statistics about the resampling
        unique_labels_after, counts_after = np.unique(y_resampled, return_counts=True)
        print("Class distribution after SMOTE:")
        for label, count in zip(unique_labels_after, counts_after):
            print(f"  Class {label}: {count} samples")
        
        # Ensure y_resampled is the right shape for the model
        # If using sparse categorical crossentropy, we want (n_samples,)
        y_resampled = np.squeeze(y_resampled)
        
        print(f"Resampled data shape: {X_resampled.shape}, labels shape: {y_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        print(traceback.format_exc())
        print("Continuing with original data...")
        return X, y

# Function to create per-class metrics
def create_per_class_metrics(num_classes):
    """
    Create precision and recall metrics for each class.
    
    Args:
        num_classes: Number of classes in the dataset
    
    Returns:
        List of Keras metrics for tracking per-class performance
    """
    metrics = []
    
    # Add basic metrics that work with sparse categorical data
    metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy'))
    
    # Instead of using class_id which can cause issues with sparse data,
    # we'll use custom metrics that we can analyze after training
    metrics.append(tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy'))
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for training (DEPRECATED: use --train_subdirs, --val_subdirs, and --test_subdirs instead)')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int)
    parser.add_argument('channels', help='Number of channels', type=int)
    parser.add_argument('batch_size', help='Number of samples in a batch', type=int)
    parser.add_argument('num_tot', help='Number of antenna * number of spatial streams', type=int)
    parser.add_argument('name_base', help='Name base for the files')
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    parser.add_argument('--use_grouped_labels', help='Group activity labels by their base letter (e.g., E1, E2 -> E)', 
                       action='store_true', default=True, required=False)
    parser.add_argument('--undersample', help='Apply undersampling to balance class distribution', 
                       action='store_true', default=True, required=False)  # Changed default to True
    parser.add_argument('--no-undersample', help='Disable undersampling (overrides --undersample)',
                       action='store_false', dest='undersample', required=False)
    parser.add_argument('--undersample_ratio', help='Ratio for undersampling (1.0 = fully balanced, 0.0 = no balancing)',
                        default=1.0, required=False, type=float)
    parser.add_argument('--verbose', help='Enable verbose output', action='store_true', default=False, required=False)
    
    # Add new arguments for directory-level train/val/test splits to prevent data leakage
    parser.add_argument('--train_subdirs', help='Comma-separated list of subdirectories to use for training',
                       default=None, required=False)
    parser.add_argument('--val_subdirs', help='Comma-separated list of subdirectories to use for validation',
                       default=None, required=False)
    parser.add_argument('--test_subdirs', help='Comma-separated list of subdirectories to use for testing',
                       default=None, required=False)
    parser.add_argument('--split_mode', help='Mode for train/val/test splitting: "directory" (split at directory level) '
                                           'or "file" (split within directories, higher risk of data leakage)',
                       choices=['directory', 'file'], default='directory', required=False)
    parser.add_argument('--ignore_unseen_labels', help='Continue training even if test data contains labels not seen in training',
                       action='store_true', default=False, required=False)
    
    args = parser.parse_args()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    csi_label_dict = []
    for lab_act in csi_act.split(','):
        csi_label_dict.append(lab_act)

    # Print a message about the label grouping configuration
    if args.use_grouped_labels:
        print("\nUsing GROUPED activity labels: Activities with the same base letter (e.g., E, E1, E2) will be treated as the same class")
        print("For example, E1 and E2 will both be classified as E")
        
        # Get both the original and grouped mappings
        original_mapping, grouped_mapping = get_label_mappings(csi_label_dict)
        
        # Show the grouping that will occur
        print("\nActivity label grouping:")
        base_letters = set(label[0] for label in csi_label_dict)
        for base in sorted(base_letters):
            grouped_labels = [label for label in csi_label_dict if label.startswith(base)]
            print(f"  {base}: {', '.join(grouped_labels)}")
    else:
        print("\nUsing ORIGINAL activity labels: Each activity (E, E1, E2, etc.) will be treated as a separate class")
        
    activities = np.asarray(csi_label_dict)
    
    name_base = args.name_base
    cache_prefix = f"{name_base}_{csi_act.replace(',','_')}"
    print(f"Cleaning up previous cache files: {cache_prefix}*")
    # Delete cache files
    for f in glob.glob(f"{cache_prefix}_cache*"):
        try:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    # Delete lockfiles
    for lockfile in glob.glob("*.lockfile"):
        try:
            os.remove(lockfile)
        except Exception as e:
            print(f"Could not delete lockfile {lockfile}: {e}")

    # Configure TensorFlow caching
    #tf.data.experimental.enable_optimizations(False)

    # Handle the train/val/test split mode
    if args.split_mode == 'directory':
        print("\nUsing DIRECTORY-LEVEL splitting to prevent data leakage")
        print("This ensures different environments/subjects are used for train, validation, and test sets")
        
        # Check if the new arguments are provided
        if args.train_subdirs is None:
            print("WARNING: --train_subdirs not specified, using the deprecated 'subdirs' argument for training")
            args.train_subdirs = args.subdirs
            
        if args.val_subdirs is None:
            print("WARNING: --val_subdirs not specified, creating an empty validation set")
            args.val_subdirs = ""
            
        if args.test_subdirs is None:
            print("WARNING: --test_subdirs not specified, creating an empty test set")
            args.test_subdirs = ""
        
        # Use the directory-specific splits
        subdirs_training = args.train_subdirs
        subdirs_validation = args.val_subdirs
        subdirs_testing = args.test_subdirs
        
        print(f"Training subdirectories: {subdirs_training}")
        print(f"Validation subdirectories: {subdirs_validation}")
        print(f"Testing subdirectories: {subdirs_testing}")
    else:
        print("\nUsing FILE-LEVEL splitting within each directory")
        print("WARNING: This approach risks data leakage if the same subject/environment appears in multiple sets")
        
        # Use the old behavior where each subdirectory contains train/val/test
        subdirs_training = args.subdirs
        subdirs_validation = args.subdirs
        subdirs_testing = args.subdirs
        
        print(f"Using the same subdirectories for all splits: {subdirs_training}")
    
    labels_train = []
    all_files_train = []
    labels_val = []
    all_files_val = []
    labels_test = []
    all_files_test = []
    sample_length = args.sample_length
    feature_length = args.feature_length
    channels = args.channels
    num_antennas = args.num_tot
    input_shape = (num_antennas, sample_length, feature_length, channels)
    input_network = (sample_length, feature_length, channels)
    batch_size = args.batch_size
    output_shape = activities.shape[0]
    labels_considered = np.arange(output_shape)
    activities = activities[labels_considered]

    suffix = '.txt'

    # Run validation on the dataset structure
    if args.verbose:
        print("\nValidating label and file structure...")
        
        # Validate training directories
        if subdirs_training:
            print("\nValidating training subdirectories...")
            validation_results_train = validate_labels_files(args.dir, subdirs_training, str(csi_act), suffix)
            
            # Print validation results summary
            for subdir, results in validation_results_train.items():
                print(f"\nTraining domain: {subdir}")
                for split, split_results in results.items():
                    if split == 'train':  # Only show training split for training directories
                        if 'error' in split_results:
                            print(f"  Train: Error - {split_results['error']}")
                        else:
                            match_status = "✓ Match" if split_results['match'] else "✗ Mismatch"
                            print(f"  Train: {split_results['labels_count']} labels, {split_results['files_count']} files - {match_status}")
                            if not split_results['match']:
                                print(f"    WARNING: Labels and files count don't match in {subdir}/train")
                            print(f"    Unique labels: {split_results['unique_labels']}")
                            
        # Validate validation directories
        if subdirs_validation:
            print("\nValidating validation subdirectories...")
            validation_results_val = validate_labels_files(args.dir, subdirs_validation, str(csi_act), suffix)
            
            # Print validation results summary
            for subdir, results in validation_results_val.items():
                print(f"\nValidation domain: {subdir}")
                for split, split_results in results.items():
                    if args.split_mode == 'directory' and split == 'train' or split == 'val':  # Use train split for directory mode
                        split_name = "Validation" if args.split_mode == 'directory' else split.capitalize()
                        if 'error' in split_results:
                            print(f"  {split_name}: Error - {split_results['error']}")
                        else:
                            match_status = "✓ Match" if split_results['match'] else "✗ Mismatch"
                            print(f"  {split_name}: {split_results['labels_count']} labels, {split_results['files_count']} files - {match_status}")
                            if not split_results['match']:
                                print(f"    WARNING: Labels and files count don't match in {subdir}/{split}")
                            print(f"    Unique labels: {split_results['unique_labels']}")
                        
        # Validate test directories
        if subdirs_testing:
            print("\nValidating test subdirectories...")
            validation_results_test = validate_labels_files(args.dir, subdirs_testing, str(csi_act), suffix)
            
            # Print validation results summary
            for subdir, results in validation_results_test.items():
                print(f"\nTest domain: {subdir}")
                for split, split_results in results.items():
                    if args.split_mode == 'directory' and split == 'train' or split == 'test':  # Use train split for directory mode
                        split_name = "Test" if args.split_mode == 'directory' else split.capitalize()
                        if 'error' in split_results:
                            print(f"  {split_name}: Error - {split_results['error']}")
                        else:
                            match_status = "✓ Match" if split_results['match'] else "✗ Mismatch"
                            print(f"  {split_name}: {split_results['labels_count']} labels, {split_results['files_count']} files - {match_status}")
                            if not split_results['match']:
                                print(f"    WARNING: Labels and files count don't match in {subdir}/{split}")
                            print(f"    Unique labels: {split_results['unique_labels']}")
        
    # Load training data
    for sdir in subdirs_training.split(','):
        if not sdir:  # Skip empty entries
            continue
            
        exp_save_dir = args.dir + sdir + '/'
        
        if args.split_mode == 'directory':
            # In directory mode, use only train subfolder data from training directories
            dir_path = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
            name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
        else:
            # In file mode (old behavior), use the train subfolder for training
            dir_train = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
            name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
        
        with open(name_labels, "rb") as fp:  # Unpickling
            domain_labels = pickle.load(fp)
            
            # Apply label grouping if enabled
            if args.use_grouped_labels:
                # Convert loaded numeric labels back to their string representation
                # then apply grouping
                str_labels = []
                for label_num in domain_labels:
                    # Look up the original activity label for this numeric index
                    for idx, act_label in enumerate(csi_label_dict):
                        if idx == label_num:
                            # Apply grouping by extracting the base letter
                            base_letter = act_label[0]
                            # Find the index of the first activity with this base letter
                            for base_idx, activity in enumerate(csi_label_dict):
                                if activity.startswith(base_letter):
                                    str_labels.append(base_idx)
                                    break
                            break
                domain_labels = str_labels
        name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            domain_files = pickle.load(fp)
            # Check if we have appropriate number of labels for files
            if len(domain_labels) == len(domain_files):
                # Use one-to-one mapping between files and labels
                labels_train.extend(domain_labels)
            elif len(domain_labels) == 1:
                # Replicate the single label for each file in this domain
                domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
                labels_train.extend(domain_labels_expanded)
            else:
                # Log warning about label mismatch and use available labels
                print(f"WARNING: Label-file count mismatch in {sdir} training set: {len(domain_labels)} labels for {len(domain_files)} files")
                # Use as many labels as available and replicate the last one if needed
                if len(domain_labels) > 0:
                    labels_train.extend(domain_labels[:min(len(domain_labels), len(domain_files))])
                    if len(domain_labels) < len(domain_files):
                        # Replicate the last label for remaining files
                        additional_labels = [domain_labels[-1]] * (len(domain_files) - len(domain_labels))
                        labels_train.extend(additional_labels)
            all_files_train.extend(domain_files)

    # Load validation data
    if args.split_mode == 'directory':
        # In directory mode, load val data from validation directories (train subfolders)
        for sdir in subdirs_validation.split(','):
            if not sdir:  # Skip empty entries
                continue
                
            dir_path = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
            name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
            
            with open(name_labels, "rb") as fp:  # Unpickling
                domain_labels = pickle.load(fp)
                
                # Apply label grouping if enabled
                if args.use_grouped_labels:
                    # Convert loaded numeric labels back to their string representation
                    # then apply grouping
                    str_labels = []
                    for label_num in domain_labels:
                        # Look up the original activity label for this numeric index
                        for idx, act_label in enumerate(csi_label_dict):
                            if idx == label_num:
                                # Apply grouping by extracting the base letter
                                base_letter = act_label[0]
                                # Find the index of the first activity with this base letter
                                for base_idx, activity in enumerate(csi_label_dict):
                                    if activity.startswith(base_letter):
                                        str_labels.append(base_idx)
                                        break
                                break
                    domain_labels = str_labels
                    
            with open(name_f, "rb") as fp:  # Unpickling
                domain_files = pickle.load(fp)
                # Check if we have appropriate number of labels for files
                if len(domain_labels) == len(domain_files):
                    # Use one-to-one mapping between files and labels
                    labels_val.extend(domain_labels)
                elif len(domain_labels) == 1:
                    # Replicate the single label for each file in this domain
                    domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
                    labels_val.extend(domain_labels_expanded)
                else:
                    # Log warning about label mismatch and use available labels
                    print(f"WARNING: Label-file count mismatch in {sdir} validation set: {len(domain_labels)} labels for {len(domain_files)} files")
                    # Use as many labels as available and replicate the last one if needed
                    if len(domain_labels) > 0:
                        labels_val.extend(domain_labels[:min(len(domain_labels), len(domain_files))])
                        if len(domain_labels) < len(domain_files):
                            # Replicate the last label for remaining files
                            additional_labels = [domain_labels[-1]] * (len(domain_files) - len(domain_labels))
                            labels_val.extend(additional_labels)
                all_files_val.extend(domain_files)
    else:
        # In file mode (old behavior), load validation data from val subfolders
        for sdir in subdirs_validation.split(','):
            if not sdir:  # Skip empty entries
                continue
                
            dir_val = args.dir + sdir + '/val_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_val_antennas_' + str(csi_act) + suffix
            with open(name_labels, "rb") as fp:  # Unpickling
                domain_labels = pickle.load(fp)
                
                # Apply label grouping if enabled
                if args.use_grouped_labels:
                    # Convert loaded numeric labels back to their string representation
                    # then apply grouping
                    str_labels = []
                    for label_num in domain_labels:
                        # Look up the original activity label for this numeric index
                        for idx, act_label in enumerate(csi_label_dict):
                            if idx == label_num:
                                # Apply grouping by extracting the base letter
                                base_letter = act_label[0]
                                # Find the index of the first activity with this base letter
                                for base_idx, activity in enumerate(csi_label_dict):
                                    if activity.startswith(base_letter):
                                        str_labels.append(base_idx)
                                        break
                                break
                    domain_labels = str_labels
            name_f = args.dir + sdir + '/files_val_antennas_' + str(csi_act) + suffix
            with open(name_f, "rb") as fp:  # Unpickling
                domain_files = pickle.load(fp)
                # Check if we have appropriate number of labels for files
                if len(domain_labels) == len(domain_files):
                    # Use one-to-one mapping between files and labels
                    labels_val.extend(domain_labels)
                elif len(domain_labels) == 1:
                    # Replicate the single label for each file in this domain
                    domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
                    labels_val.extend(domain_labels_expanded)
                else:
                    # Log warning about label mismatch and use available labels
                    print(f"WARNING: Label-file count mismatch in {sdir} validation set: {len(domain_labels)} labels for {len(domain_files)} files")
                    # Use as many labels as available and replicate the last one if needed
                    if len(domain_labels) > 0:
                        labels_val.extend(domain_labels[:min(len(domain_labels), len(domain_files))])
                        if len(domain_labels) < len(domain_files):
                            # Replicate the last label for remaining files
                            additional_labels = [domain_labels[-1]] * (len(domain_files) - len(domain_labels))
                            labels_val.extend(additional_labels)
                all_files_val.extend(domain_files)

    # Load test data
    if args.split_mode == 'directory':
        # In directory mode, load test data from test directories (train subfolders)
        for sdir in subdirs_testing.split(','):
            if not sdir:  # Skip empty entries
                continue
                
            dir_path = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
            name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
            
            with open(name_labels, "rb") as fp:  # Unpickling
                domain_labels = pickle.load(fp)
                
                # Apply label grouping if enabled
                if args.use_grouped_labels:
                    # Convert loaded numeric labels back to their string representation
                    # then apply grouping
                    str_labels = []
                    for label_num in domain_labels:
                        # Look up the original activity label for this numeric index
                        for idx, act_label in enumerate(csi_label_dict):
                            if idx == label_num:
                                # Apply grouping by extracting the base letter
                                base_letter = act_label[0]
                                # Find the index of the first activity with this base letter
                                for base_idx, activity in enumerate(csi_label_dict):
                                    if activity.startswith(base_letter):
                                        str_labels.append(base_idx)
                                        break
                                break
                    domain_labels = str_labels
                    
            with open(name_f, "rb") as fp:  # Unpickling
                domain_files = pickle.load(fp)
                # Check if we have appropriate number of labels for files
                if len(domain_labels) == len(domain_files):
                    # Use one-to-one mapping between files and labels
                    labels_test.extend(domain_labels)
                elif len(domain_labels) == 1:
                    # Replicate the single label for each file in this domain
                    domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
                    labels_test.extend(domain_labels_expanded)
                else:
                    # Log warning about label mismatch and use available labels
                    print(f"WARNING: Label-file count mismatch in {sdir} test set: {len(domain_labels)} labels for {len(domain_files)} files")
                    # Use as many labels as available and replicate the last one if needed
                    if len(domain_labels) > 0:
                        labels_test.extend(domain_labels[:min(len(domain_labels), len(domain_files))])
                        if len(domain_labels) < len(domain_files):
                            # Replicate the last label for remaining files
                            additional_labels = [domain_labels[-1]] * (len(domain_files) - len(domain_labels))
                            labels_test.extend(additional_labels)
                all_files_test.extend(domain_files)
    else:
        # In file mode (old behavior), load test data from test subfolders
        for sdir in subdirs_testing.split(','):
            if not sdir:  # Skip empty entries
                continue
                
            dir_test = args.dir + sdir + '/test_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_test_antennas_' + str(csi_act) + suffix
            with open(name_labels, "rb") as fp:  # Unpickling
                domain_labels = pickle.load(fp)
                
                # Apply label grouping if enabled
                if args.use_grouped_labels:
                    # Convert loaded numeric labels back to their string representation
                    # then apply grouping
                    str_labels = []
                    for label_num in domain_labels:
                        # Look up the original activity label for this numeric index
                        for idx, act_label in enumerate(csi_label_dict):
                            if idx == label_num:
                                # Apply grouping by extracting the base letter
                                base_letter = act_label[0]
                                # Find the index of the first activity with this base letter
                                for base_idx, activity in enumerate(csi_label_dict):
                                    if activity.startswith(base_letter):
                                        str_labels.append(base_idx)
                                        break
                                break
                    domain_labels = str_labels
            name_f = args.dir + sdir + '/files_test_antennas_' + str(csi_act) + suffix
            with open(name_f, "rb") as fp:  # Unpickling
                domain_files = pickle.load(fp)
                # Check if we have appropriate number of labels for files
                if len(domain_labels) == len(domain_files):
                    # Use one-to-one mapping between files and labels
                    labels_test.extend(domain_labels)
                elif len(domain_labels) == 1:
                    # Replicate the single label for each file in this domain
                    domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
                    labels_test.extend(domain_labels_expanded)
                else:
                    # Log warning about label mismatch and use available labels
                    print(f"WARNING: Label-file count mismatch in {sdir} test set: {len(domain_labels)} labels for {len(domain_files)} files")
                    # Use as many labels as available and replicate the last one if needed
                    if len(domain_labels) > 0:
                        labels_test.extend(domain_labels[:min(len(domain_labels), len(domain_files))])
                        if len(domain_labels) < len(domain_files):
                            # Replicate the last label for remaining files
                            additional_labels = [domain_labels[-1]] * (len(domain_files) - len(domain_labels))
                            labels_test.extend(additional_labels)
                all_files_test.extend(domain_files)

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                           labels_considered]
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                             labels_considered]

    # No expansion needed - use original files and labels
    file_train_selected_expanded = file_train_selected
    labels_train_selected_expanded = labels_train_selected
    
    print("Sample labels:", labels_train_selected_expanded[:5])
    print("Label shapes:", np.array(labels_train_selected_expanded).shape)
    
    # Also use original validation and test data without expansion
    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]

    file_val_selected_expanded = file_val_selected
    labels_val_selected_expanded = labels_val_selected
        
    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                         labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                           labels_considered]

    file_test_selected_expanded = file_test_selected
    labels_test_selected_expanded = labels_test_selected
    
    # Verify data integrity after loading
    if args.verbose:
        print("\nVerifying data integrity after loading...")
        train_integrity = verify_label_file_integrity(file_train_selected_expanded, labels_train_selected_expanded, "Training set")
        val_integrity = verify_label_file_integrity(file_val_selected_expanded, labels_val_selected_expanded, "Validation set")
        test_integrity = verify_label_file_integrity(file_test_selected_expanded, labels_test_selected_expanded, "Test set")
        
        if train_integrity and val_integrity and test_integrity:
            print("Data integrity check passed for all datasets")
        else:
            print("WARNING: Data integrity issues detected. Check the logs for details.")
    
    # Create a custom data generator for training instead of using TensorFlow's dataset API
    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, file_names, labels, input_shape=(100, 100, 4), batch_size=16, shuffle=True, label_mapping=None, undersample=False, undersample_ratio=1.0):
            self.file_names = file_names if file_names is not None else []
            self.labels = labels if labels is not None else []
            self.input_shape = input_shape
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.label_mapping = label_mapping  # Map from class indices to original labels
            self.undersample = undersample  # Whether to apply undersampling to balance classes
            self.undersample_ratio = undersample_ratio  # Ratio for undersampling (1.0 = fully balanced)
            
            # For tracking samples across epochs when undersampling
            self.epoch_counter = 0
            self.class_indices = None  # Will be initialized if undersampling is enabled
            
            # Check if we have valid data
            if len(self.file_names) == 0 or len(self.labels) == 0:
                print("WARNING: Empty file list or labels provided to CustomDataGenerator.")
                self.empty_generator = True
                # Initialize with dummy indices
                self.all_indices = np.array([0])
                self.indices = np.array([0])
                self.sample_usage_count = np.zeros(1)
            else:
                self.empty_generator = False
                # Track how often each sample is used
                self.sample_usage_count = np.zeros(len(file_names))
                # Store original indices for reference
                self.all_indices = np.arange(len(file_names))
            
            # Memory mapping cache for large files
            self.mmap_cache = {}
            
            # Skip file validation for empty generators
            if not self.empty_generator:
                # Validate file paths first
                self.validate_files()
                
                # Apply undersampling if requested
                if self.undersample:
                    print(f"Applying undersampling with ratio {self.undersample_ratio} to balance class distribution...")
                    # Store class indices for reuse
                    self._precompute_class_indices()
                    self.indices = self._create_balanced_indices()
                else:
                    self.indices = self.all_indices.copy()
                    
                # Shuffle if needed
                if self.shuffle:
                    np.random.shuffle(self.indices)

        def validate_files(self):
            """Validate that files exist and can be loaded."""
            # Check if file_names is empty
            if len(self.file_names) == 0:
                print("WARNING: Empty file_names list in validate_files. Cannot validate files.")
                self.empty_generator = True
                return
                
            invalid_indices = []
            
            # First check that files and labels are the same length
            if len(self.file_names) != len(self.labels):
                print(f"ERROR: Mismatch between file names ({len(self.file_names)}) and labels ({len(self.labels)})")
                # Create dummy indices to avoid breaking the generator
                self.indices = list(range(min(len(self.file_names), len(self.labels))))
                return
                
            # Check each file
            for i, file_path in enumerate(self.file_names):
                if not os.path.exists(file_path):
                    print(f"WARNING: File not found: {file_path}")
                    invalid_indices.append(i)
                else:
                    try:
                        # For .npy files, we can use np.load with mmap_mode to avoid loading the whole file
                        if file_path.endswith('.npy'):
                            # Just verify the file can be opened
                            with open(file_path, 'rb') as f:
                                # Check if it's a valid numpy file by reading the header
                                pass
                        # Handle .npz files (created by savez_compressed)
                        elif file_path.endswith('.npz'):
                            try:
                                # Load the npz file - no memory mapping for npz files
                                loaded = np.load(file_path, allow_pickle=True)
                                # Get the data array from the npz file
                                data = loaded['data']
                                
                                # Check if transpose needed based on input shape
                                if data.shape != self.input_shape:
                                    if len(data.shape) == 3 and data.shape[0] == 4:
                                        # Likely needs transpose from (4, 100, 340) to (340, 100, 4)
                                        data = np.transpose(data, (2, 1, 0))
                            except Exception as e:
                                print(f"ERROR loading NPZ file {file_path}: {str(e)}")
                                dummy_data = np.zeros(self.input_shape)
                                data = dummy_data
                                # Actually mark file as invalid
                                invalid_indices.append(i)
                                continue
                        # Handle our custom binary format
                        elif file_path.endswith('.bin'):
                            try:
                                # Just check if we can open and read the file
                                with open(file_path, 'rb') as f:
                                    # Try to read the shape
                                    shape = np.fromfile(f, dtype=np.int32, count=3)
                                    # If we got a valid shape, the file is probably good
                                    if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
                                        print(f"WARNING: Invalid shape in file {file_path}: {shape}")
                                        invalid_indices.append(i)
                                        continue
                            except Exception as e:
                                print(f"ERROR validating binary file {file_path}: {str(e)}")
                                invalid_indices.append(i)
                                continue
                        else:
                            # Regular pickle file loading
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)  # Shape (4, 100, 340)
                            data = np.transpose(data, (2, 1, 0))  # Transpose to (340, 100, 4)
                    except Exception as e:
                        print(f"ERROR: Cannot load file {file_path}: {e}")
                        invalid_indices.append(i)
            
            if invalid_indices:
                # Remove problematic files from both file_names and labels lists
                valid_indices = [i for i in range(len(self.file_names)) if i not in invalid_indices]
                self.file_names = [self.file_names[i] for i in valid_indices]
                self.labels = [self.labels[i] for i in valid_indices]
                print(f"Removed {len(invalid_indices)} invalid files. Remaining: {len(self.file_names)}")
            
            # Initialize indices list after validation
            self.indices = list(range(len(self.file_names)))
            if self.shuffle:
                np.random.shuffle(self.indices)

        def _precompute_class_indices(self):
            """Precompute indices for each class to avoid recomputing every epoch"""
            # Get all unique classes
            unique_labels = np.unique(self.labels)
            
            # Create a dictionary to store indices for each class
            self.class_indices = {}
            for label in unique_labels:
                self.class_indices[label] = np.where(np.array(self.labels) == label)[0]
                
            # Calculate min count and target counts per class (used for sampling)
            counts = [len(indices) for indices in self.class_indices.values()]
            self.min_count = np.min(counts)
            
            # Precompute target counts for each class
            self.target_counts = {}
            for label, indices in self.class_indices.items():
                current_count = len(indices)
                if current_count <= self.min_count:
                    # For minority classes, keep all samples
                    self.target_counts[label] = current_count
                else:
                    # For majority classes, undersample based on the ratio
                    self.target_counts[label] = int(current_count - self.undersample_ratio * (current_count - self.min_count))
            
        def _create_balanced_indices(self):
            """Create indices using stratified sampling that maintains relative class proportions
            while ensuring all samples have a fair chance of being selected"""
            # If class indices haven't been precomputed, do it now
            if self.class_indices is None:
                self._precompute_class_indices()
                
            balanced_indices = []
            
            # Calculate total target count based on batch size and class ratios
            total_samples_needed = self.batch_size * (len(self) + 1)  # Ensure enough samples for all batches
            
            # Safety check - ensure our file_names length is greater than 0
            if len(self.file_names) == 0:
                print("WARNING: Empty file_names list detected, returning empty indices")
                return np.array([])
            
            # Get class distribution
            unique_labels, class_counts = np.unique(self.labels, return_counts=True)
            total_samples = len(self.labels)
            
            # Calculate target counts for each class based on original distribution
            # but with a minimum threshold to prevent extreme underrepresentation
            class_ratios = class_counts / total_samples
            min_ratio = 0.5 * np.min(class_ratios)  # Set minimum ratio to half of the smallest class
            
            # Adjust class ratios to ensure minimum representation
            adjusted_ratios = np.maximum(class_ratios, min_ratio)
            adjusted_ratios = adjusted_ratios / np.sum(adjusted_ratios)  # Normalize to sum to 1
            
            # Calculate target counts for each class with stratified approach
            target_counts = {label: int(total_samples_needed * adjusted_ratios[i]) 
                            for i, label in enumerate(unique_labels)}
            
            print("Stratified sampling target counts:")
            for label, count in target_counts.items():
                original_count = len(self.class_indices[label])
                print(f"  Class {label}: {count} samples (from {original_count})")
            
            for label, indices in self.class_indices.items():
                current_count = len(indices)
                target_count = target_counts[label]
                
                # Get sample usage counts for this class
                class_usage = self.sample_usage_count[indices]
                
                # Sort indices by usage count (prioritize less used samples)
                sorted_by_usage = indices[np.argsort(class_usage)]
                
                # If target count exceeds available samples, use weighted sampling with replacement
                if target_count > current_count:
                    # Weight inversely proportional to usage count to prefer less-used samples
                    weights = 1.0 / (class_usage + 1.0)
                    weights = weights / np.sum(weights)
                    
                    # Sample with replacement using weights
                    sampled_indices = np.random.choice(
                        indices, 
                        size=target_count, 
                        replace=True, 
                        p=weights
                    )
                else:
                    # If we have enough samples, prioritize less used samples
                    # but occasionally mix in some randomness
                    if np.random.random() < 0.1:  # 10% chance of random sampling
                        sampled_indices = np.random.choice(indices, target_count, replace=False)
                    else:
                        sampled_indices = sorted_by_usage[:target_count]
                
                balanced_indices.extend(sampled_indices)
                
                # Update usage count for selected samples
                for idx in sampled_indices:
                    self.sample_usage_count[idx] += 1
            
            # Shuffle the indices to avoid batches with same class
            np.random.shuffle(balanced_indices)
            
            return np.array(balanced_indices)
            
        def __len__(self):
            # Safety check to ensure we don't return a negative or zero value
            if not hasattr(self, 'indices') or len(self.indices) == 0:
                print("WARNING: Empty indices detected in __len__, returning 1")
                return 1
            return max(1, int(np.ceil(len(self.indices) / self.batch_size)))

        def __getitem__(self, idx):
            # Early return with dummy data if file_names is empty
            if len(self.file_names) == 0:
                print("WARNING: Empty file_names list. Returning dummy batch.")
                # Create dummy batch with the right shapes
                dummy_x = np.zeros((self.batch_size,) + self.input_shape)
                dummy_y = np.zeros(self.batch_size)
                return dummy_x, dummy_y
                
            batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
            
            # Safety check - ensure all indices are within range
            valid_indices = []
            for i in batch_indices:
                if i < len(self.file_names):
                    valid_indices.append(i)
                else:
                    # If out of range, replace with a valid index by sampling from existing valid indices
                    print(f"Warning: Index {i} out of range for file_names length {len(self.file_names)}. Using replacement.")
                    if len(valid_indices) > 0:
                        # Use an already validated index if available
                        valid_indices.append(valid_indices[0])
                    else:
                        # Otherwise use index 0 as fallback if possible
                        if len(self.file_names) > 0:
                            valid_indices.append(0)
            
            # If we couldn't get any valid indices, return dummy data
            if len(valid_indices) == 0:
                print("WARNING: No valid indices found. Returning dummy batch.")
                # Create dummy batch with the right shapes
                dummy_x = np.zeros((self.batch_size,) + self.input_shape)
                dummy_y = np.zeros(self.batch_size)
                return dummy_x, dummy_y
            
            # Update batch_indices with only valid indices
            batch_indices = valid_indices
            
            # Ensure we have the correct batch size
            if len(batch_indices) < self.batch_size:
                # If we don't have enough indices, sample with replacement to reach batch size
                additional_needed = self.batch_size - len(batch_indices)
                if len(batch_indices) > 0:  # If we have at least one valid index
                    additional_indices = np.random.choice(batch_indices, size=additional_needed, replace=True)
                    batch_indices.extend(additional_indices)
                else:  # Worst case: no valid indices at all
                    batch_indices = [0] * self.batch_size
            
            batch_files = [self.file_names[i] for i in batch_indices]
            batch_labels = [self.labels[i] for i in batch_indices]
            
            batch_x = []
            for file_path in batch_files:
                try:
                    # Use memory mapping for large .npy files
                    if file_path.endswith('.npy'):
                        if file_path not in self.mmap_cache:
                            self.mmap_cache[file_path] = np.load(file_path, mmap_mode='r')
                        data = self.mmap_cache[file_path][...]
                        
                        # Check if transpose needed based on input shape
                        if data.shape != self.input_shape:
                            if len(data.shape) == 3 and data.shape[0] == 4:
                                # Likely needs transpose from (4, 100, 340) to (340, 100, 4)
                                data = np.transpose(data, (2, 1, 0))
                    # Handle .npz files (created by savez_compressed)
                    elif file_path.endswith('.npz'):
                        try:
                            # Load the npz file - no memory mapping for npz files
                            loaded = np.load(file_path, allow_pickle=True)
                            # Get the data array from the npz file
                            data = loaded['data']
                            
                            # Check if transpose needed based on input shape
                            if data.shape != self.input_shape:
                                if len(data.shape) == 3 and data.shape[0] == 4:
                                    # Likely needs transpose from (4, 100, 340) to (340, 100, 4)
                                    data = np.transpose(data, (2, 1, 0))
                        except Exception as e:
                            print(f"ERROR loading NPZ file {file_path}: {str(e)}")
                            dummy_data = np.zeros(self.input_shape)
                            data = dummy_data
                    # Handle our custom binary format
                    elif file_path.endswith('.bin'):
                        try:
                            with open(file_path, 'rb') as f:
                                # Read the shape information
                                shape = np.fromfile(f, dtype=np.int32, count=3)
                                # Read the data
                                data = np.fromfile(f, dtype=np.float32)
                                # Verify the data is complete and has the expected size
                                expected_size = np.prod(shape)
                                if len(data) != expected_size:
                                    print(f"WARNING: Binary file {file_path} has incomplete data. Expected {expected_size} elements, got {len(data)}")
                                    # Use zeros as fallback
                                    data = np.zeros(shape, dtype=np.float32)
                                else:
                                    # Reshape according to the stored shape
                                    data = data.reshape(shape)
                        except Exception as e:
                            print(f"ERROR loading binary file {file_path}: {str(e)}")
                            dummy_data = np.zeros(self.input_shape)
                            data = dummy_data
                    else:
                        # Regular pickle file loading
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)  # Shape (4, 100, 340)
                        data = np.transpose(data, (2, 1, 0))  # Transpose to (340, 100, 4)
                    
                    batch_x.append(data)
                except FileNotFoundError:
                    print(f"ERROR: File not found during batch loading: {file_path}")
                    # Create a dummy data point with the right shape filled with zeros
                    dummy_data = np.zeros(self.input_shape)
                    batch_x.append(dummy_data)
                except Exception as e:
                    print(f"ERROR loading file {file_path}: {str(e)}")
                    dummy_data = np.zeros(self.input_shape)
                    batch_x.append(dummy_data)
            
            return np.array(batch_x), np.array(batch_labels)
            
        def on_epoch_end(self):
            """Method called at the end of every epoch to reshuffle the data and ensure balanced sampling."""
            # Skip processing for empty generators
            if hasattr(self, 'empty_generator') and self.empty_generator:
                return
                
            # Skip if file_names is empty
            if len(self.file_names) == 0:
                print("WARNING: Empty file_names list in on_epoch_end. Skipping.")
                return
                
            # Increment epoch counter
            self.epoch_counter += 1
            
            # If undersampling is enabled, create a new balanced set for each epoch
            if self.undersample:
                self.indices = self._create_balanced_indices()
                
                # Every N epochs (where N is the number of classes), reset the usage counts
                # This helps avoid getting stuck in specific sampling patterns
                if self.class_indices and len(self.class_indices) > 0:
                    num_classes = len(self.class_indices)
                    if self.epoch_counter % (num_classes * 2) == 0:
                        print("Resetting sample usage tracking for fresh sampling patterns")
                        self.sample_usage_count = np.zeros(len(self.file_names))
            
            # Shuffle indices if needed
            if self.shuffle and len(self.indices) > 0:
                np.random.shuffle(self.indices)
                
            # Calculate class distribution in the shuffled data
            all_labels = [self.labels[i] for i in self.indices]
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            
            # Print class distribution
            print("\n----- Class Distribution in Shuffled Data -----")
            print(f"Total samples: {len(all_labels)}")
            
            # Create a dictionary to store class distribution
            class_distribution = {}
            for label, count in zip(unique_labels, counts):
                percentage = (count / len(all_labels)) * 100
                class_distribution[int(label)] = {"count": int(count), "percentage": percentage}
            
            # Try to get access to the reverse mapping if available
            try:
                # First try to use the label mapping provided in constructor
                if self.label_mapping is not None:
                    for class_idx in sorted(class_distribution.keys()):
                        stats = class_distribution[class_idx]
                        original_label = f" (Original: {self.label_mapping.get(class_idx, 'Unknown')})"
                        print(f"Class {class_idx}{original_label}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                # Fall back to global variables if available
                elif 'index_to_label' in globals():
                    for class_idx in sorted(class_distribution.keys()):
                        stats = class_distribution[class_idx]
                        original_label = f" (Original: {globals()['index_to_label'].get(class_idx, 'Unknown')})"
                        print(f"Class {class_idx}{original_label}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                else:
                    # No mapping available, just print indices
                    for class_idx in sorted(class_distribution.keys()):
                        stats = class_distribution[class_idx]
                        print(f"Class {class_idx}: {stats['count']} samples ({stats['percentage']:.2f}%)")
            except Exception as e:
                # Fall back to simpler output if there's an error
                print(f"Error accessing label mapping: {e}")
                for class_idx in sorted(class_distribution.keys()):
                    stats = class_distribution[class_idx]
                    print(f"Class {class_idx}: {stats['count']} samples ({stats['percentage']:.2f}%)")
                
            print("------------------------------------------------\n")

    # Create label mapping
    unique_labels = np.unique(np.concatenate([labels_train_selected_expanded, 
                                            labels_val_selected_expanded,
                                            labels_test_selected_expanded]))
    
    print(f"Found {len(unique_labels)} unique labels across all datasets: {unique_labels}")
    
    # IMPORTANT: We are simplifying the label mapping to only include 
    # classes that actually exist in the training data
    # This prevents issues with extraneous classes like C1, C2, E2, etc.
    training_labels_only = np.unique(labels_train_selected_expanded)
    print(f"Found {len(training_labels_only)} unique labels in TRAINING data: {training_labels_only}")
    print("Using ONLY training labels for mapping to ensure model outputs match actual classes")
    
    # Ensure all labels are numeric for consistency
    numeric_labels = []
    for label in training_labels_only:  # Only use training labels
        if isinstance(label, (int, np.integer)):
            numeric_labels.append(int(label))
        elif isinstance(label, str) and label.isdigit():
            numeric_labels.append(int(label))
        else:
            # For non-numeric labels, we'll assign a unique numeric ID
            print(f"Warning: Non-numeric label '{label}' found, using numeric placeholder")
            numeric_labels.append(hash(str(label)) % 1000)  # Use hash for unique numeric ID
    
    # Create the mapping with consistent numeric types - using ONLY training labels
    label_to_index = {label: idx for idx, label in enumerate(sorted(numeric_labels))}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    # Print the simplified mapping
    print("\nSimplified label mapping (using only training labels):")
    for label, idx in label_to_index.items():
        print(f"  Original label {label} → Index {idx}")
    
    # Check that all validation and test labels have mappings
    missing_val_labels = [lbl for lbl in np.unique(labels_val_selected_expanded) if lbl not in label_to_index]
    missing_test_labels = [lbl for lbl in np.unique(labels_test_selected_expanded) if lbl not in label_to_index]
    
    # Add strict validation
    missing_labels = set(np.unique(labels_test_selected_expanded)) - set(label_to_index.keys())
    if missing_labels:
        print(f"Critical Error: Test contains {len(missing_labels)} labels not seen in training")
        print(f"Missing labels: {missing_labels}")
        if not args.ignore_unseen_labels:
            raise ValueError("Test data contains unseen labels")
        else:
            print("Warning: Continuing despite unseen labels because --ignore_unseen_labels is set")
    
    if missing_val_labels or missing_test_labels:
        print("\nWARNING: Some validation/test labels are not in training data!")
        print(f"  Missing validation labels: {missing_val_labels}")
        print(f"  Missing test labels: {missing_test_labels}")
        print("  These samples will be excluded from evaluation.")
    
    # Save the simplified mappings for easier loading in test scripts
    mapping_data = {
        'label_to_index': label_to_index,
        'index_to_label': index_to_label,
        'activities': [str(a) for a in activities.tolist()]  # Ensure activities are strings
    }
    
    # Save in multiple formats and locations for better accessibility
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_to_index, f)  # Original format for backward compatibility
        
    with open(f'{name_base}_label_mapping.pkl', 'wb') as f:
        pickle.dump(mapping_data, f)  # Enhanced format
    
    print(f"Simplified label mapping saved to 'label_mapping.pkl' and '{name_base}_label_mapping.pkl'")
    
    # Convert labels to indices for Keras training
    train_labels_continuous = np.array([label_to_index[label] for label in labels_train_selected_expanded])
    
    # Calculate the number of samples in each set
    num_samples_train = len(file_train_selected_expanded)
    num_samples_val = len(file_val_selected_expanded)
    num_samples_test = len(file_test_selected_expanded)
    
    # Initialize validation and test labels
    val_labels_continuous = np.array([])
    test_labels_continuous = np.array([])
    
    # Only process validation and test labels if we have data
    if num_samples_val > 0:
        val_labels_continuous = np.array([label_to_index[label] for label in labels_val_selected_expanded])
    
    if num_samples_test > 0:
        # First, filter test samples to include only those with labels in training data
        # First, identify test samples with valid labels (those that exist in the training set)
        valid_test_indices = [i for i, label in enumerate(labels_test_selected_expanded) if label in label_to_index]
        
        if valid_test_indices:
            # Filter test samples to include only those with valid labels
            labels_test_selected_filtered = [labels_test_selected[i] for i in valid_test_indices]
            labels_test_selected_expanded_filtered = [labels_test_selected_expanded[i] for i in valid_test_indices]
            file_test_selected_filtered = [file_test_selected[i] for i in valid_test_indices]
            file_test_selected_expanded_filtered = [file_test_selected_expanded[i] for i in valid_test_indices]
            
            # Update our test data to use only the filtered samples
            labels_test_selected = labels_test_selected_filtered
            labels_test_selected_expanded = labels_test_selected_expanded_filtered
            file_test_selected = file_test_selected_filtered
            file_test_selected_expanded = file_test_selected_expanded_filtered
            
            # Update sample count
            num_samples_test = len(file_test_selected_expanded)
            
            # Now safely convert the filtered labels to indices
            test_labels_continuous = np.array([label_to_index[label] for label in labels_test_selected_expanded])
            print(f"Filtered test set to {num_samples_test} samples after removing samples with unseen labels")
        else:
            # No valid test samples left
            print("WARNING: All test samples contained unseen labels and were filtered out!")
            num_samples_test = 0

    # Find the actual number of unique classes in the training data
    actual_unique_classes = len(np.unique(train_labels_continuous))
    print(f"Number of unique classes in training data: {actual_unique_classes}")

    # Calculate class weights based on unique classes in training data, not all datasets
    # This ensures correct mapping between class indices and weights
    # NOTE: It's critical to use actual_unique_classes here instead of len(unique_labels)
    # because class weights must align with the actual classes present in the training data.
    # Using len(unique_labels) can cause misaligned indices if some classes exist in val/test
    # but not in training data, leading to incorrect weight assignments.
    class_weights_raw = compute_class_weights(train_labels_continuous, actual_unique_classes)

    # Keras requires class_weight to be a dictionary with keys from 0 to num_classes-1
    print("\nAdjusting class weights for Keras (requires consecutive indices from 0 to num_classes-1):")

    # Map from our indices to consecutive indices (0 to num_classes-1)
    # Using only the indices that actually appear in the training data
    unique_indices = sorted(list(set(train_labels_continuous)))
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

    # Create a new class weights dictionary with proper consecutive indices
    class_weights = {}
    for i in range(actual_unique_classes):
        if i < len(unique_indices):
            old_idx = unique_indices[i]
            class_weights[i] = class_weights_raw.get(old_idx, 1.0)
        else:
            class_weights[i] = 1.0  # Default weight for any missing class
    
    # Map continuous indices to Keras continuous indices (0 to num_classes-1)
    train_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in train_labels_continuous])
    
    # Initialize validation and test labels for Keras
    val_labels_continuous_keras = np.array([])
    test_labels_continuous_keras = np.array([])
    
    # Only map validation and test labels if we have data
    if num_samples_val > 0:
        val_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in val_labels_continuous])
    
    if num_samples_test > 0:
        test_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in test_labels_continuous])

    # Create data generators with class balance parameters
    print("\nCreating data generators...")
    
    # Apply SMOTE for minority class oversampling combined with controlled undersampling
    use_smote_with_undersampling = True  # Set to True to enable SMOTE with undersampling
    
    if use_smote_with_undersampling and num_samples_train > 0:
        print("\nApplying combined SMOTE and controlled undersampling strategy...")
        
        # First, we need to load the training data to apply SMOTE
        # This is a memory-intensive operation but provides better class balance
        X_train_loaded = []
        
        print(f"Loading {len(file_train_selected_expanded)} files for SMOTE processing...")
        for i, file_path in enumerate(file_train_selected_expanded):
            if i % 100 == 0:
                print(f"  Loading file {i}/{len(file_train_selected_expanded)}...")
            try:
                # Load the file and reshape for SMOTE
                data = np.load(file_path, allow_pickle=True)
                X_train_loaded.append(data.reshape(1, -1))  # Flatten for SMOTE
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                # Use a zero array as placeholder for corrupted files
                X_train_loaded.append(np.zeros((1, np.prod((340, 100, 4)))))
        
        # Convert to numpy array and reshape for SMOTE
        X_train_loaded = np.vstack(X_train_loaded)
        print(f"Loaded data shape: {X_train_loaded.shape}")
        
        # Get class distribution before resampling
        unique_labels, counts_before = np.unique(train_labels_continuous_keras, return_counts=True)
        min_samples = np.min(counts_before)
        max_samples = np.max(counts_before)
        
        # Calculate target counts based on controlled undersampling of majority classes
        # and moderate oversampling of minority classes
        mean_samples = np.mean(counts_before)
        
        # Control the degree of undersampling with this ratio (0.5 = meet halfway between min and max)
        undersampling_strength = 0.7  # Higher = less undersampling
        
        # Calculate target samples for each class
        sampling_strategy = {}
        print("Class distribution before resampling:")
        for label, count in zip(unique_labels, counts_before):
            print(f"  Class {label}: {count} samples")
            if count < mean_samples:
                # Oversample minority classes to approach the mean
                target = int(count + (mean_samples - count) * 0.8)  # 80% of the way to the mean
                # Ensure we're not requesting fewer samples than original for over-sampling
                target = max(target, count)
            else:
                # Undersample majority classes based on strength parameter
                target = int(mean_samples + (count - mean_samples) * undersampling_strength)
                # For majority classes, ensure target is at least 1 but can be less than original
                target = max(1, target)
            
            sampling_strategy[label] = target
            print(f"  Class {label}: {count} original samples -> {target} target samples")
        
        # Apply combined over/undersampling with SMOTE for minority classes
        try:
            from imblearn.combine import SMOTETomek
            print("Using SMOTETomek for combined over/undersampling...")
            
            # Modify the sampling strategy for SMOTETomek
            # For any class where we're requesting fewer samples than original, adjust to match original
            for label, count in zip(unique_labels, counts_before):
                if sampling_strategy[label] < count:
                    print(f"  Adjusting class {label}: target {sampling_strategy[label]} -> {count} (original count)")
                    sampling_strategy[label] = count
            
            # Create SMOTETomek resampler with custom sampling strategy
            resampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
            
            # Apply resampling
            X_resampled, y_resampled = resampler.fit_resample(X_train_loaded, train_labels_continuous_keras)
            
            # Print statistics about the resampling
            unique_labels_after, counts_after = np.unique(y_resampled, return_counts=True)
            print("Class distribution after combined SMOTE and undersampling:")
            for label, count in zip(unique_labels_after, counts_after):
                print(f"  Class {label}: {count} samples")
            
            # At this point, we need to convert the resampled data back to files
            # This is necessary because our CustomDataGenerator works with file paths
            print(f"Reconstructing {len(X_resampled)} samples into temporary files...")
            
            # Create a temp directory for storing the resampled data
            import tempfile
            import os
            
            temp_dir = tempfile.mkdtemp(prefix="resampled_data_")
            print(f"Created temporary directory: {temp_dir}")
            
            # Save the resampled data to temporary files
            temp_file_paths = []
            for i, (x, y) in enumerate(zip(X_resampled, y_resampled)):
                if i % 500 == 0:
                    print(f"  Saving sample {i}/{len(X_resampled)}...")
                
                # Reshape back to original dimensions
                try:
                    # Primary target shape should be (100, 100, 4) to match model input
                    x_reshaped = x.reshape(100, 100, 4)
                except ValueError:
                    # If the reshape fails, calculate the proper dimensions
                    total_elements = x.size
                    print(f"  Warning: Cannot reshape array of size {total_elements} into shape (100,100,4)")
                    
                    # Try alternative shapes if needed
                    if total_elements == 136000:  # For 340×100×4 case
                        print(f"  Adjusting reshape dimensions for sample {i} to (340,100,4)")
                        x_reshaped = x.reshape(340, 100, 4)
                    else:
                        # For other sizes, try to infer dimensions
                        print(f"  Warning: Unexpected array size: {total_elements}")
                        # A general approach: keep the last two dimensions if possible
                        height = total_elements // (100 * 4)
                        if height * 100 * 4 == total_elements:
                            x_reshaped = x.reshape(height, 100, 4)
                        else:
                            # Fall back to a simple reshaping that preserves total elements
                            x_reshaped = x.reshape(-1, 4)
                
                # Create a unique filename - use direct binary format instead of npz
                temp_file = os.path.join(temp_dir, f"resampled_sample_{i}_class_{y}.bin")
                
                # Save the array directly to a binary file
                # This avoids all pickle-related issues
                try:
                    # Ensure data is in native byte order to maximize compatibility
                    x_reshaped = x_reshaped.astype(np.float32).copy(order='C')
                    with open(temp_file, 'wb') as f:
                        # Write shape information first
                        np.array(x_reshaped.shape, dtype=np.int32).tofile(f)
                        # Then write the raw data
                        x_reshaped.tofile(f)
                        # Force write to disk with flush and fsync
                        f.flush()
                        os.fsync(f.fileno())
                    temp_file_paths.append(temp_file)
                except Exception as e:
                    print(f"ERROR saving sample {i}: {e}")
                    continue
            
            print(f"Saved {len(temp_file_paths)} resampled samples to disk")
            
            # Update the file paths and labels for the data generator
            file_train_selected_expanded = temp_file_paths
            train_labels_continuous_keras = y_resampled
            
            # Since we've already balanced the dataset, we can disable the generator's undersampling
            args.undersample = False
            print("Disabled generator's undersampling since data is already balanced")
            
        except ImportError:
            print("Could not import SMOTETomek. Falling back to standard SMOTE...")
            # Fall back to standard SMOTE
            X_resampled, y_resampled = apply_smote(X_train_loaded, train_labels_continuous_keras)
        except Exception as e:
            print(f"Error during resampling: {e}")
            print(traceback.format_exc())
            print("Continuing with original data...")
    
    try:
        # Check if we have any training data available
        if len(file_train_selected_expanded) == 0:
            print("WARNING: No training files are available. Creating dummy generator.")
            # Create a dummy generator that will return empty batches
            train_generator = CustomDataGenerator(
                [], [], 
                input_shape=(100, 100, 4),
                batch_size=batch_size,
                shuffle=True,
                label_mapping=index_to_label
            )
        else:
            train_generator = CustomDataGenerator(
                file_train_selected_expanded, 
                train_labels_continuous_keras,
                input_shape=(100, 100, 4),
                batch_size=batch_size,
                shuffle=True,
                label_mapping=index_to_label,
                undersample=args.undersample,
                undersample_ratio=args.undersample_ratio
            )
        
        # Only create validation generator if we have validation data
        val_generator = None
        if num_samples_val > 0:
            try:
                val_generator = CustomDataGenerator(
                    file_val_selected_expanded, 
                    val_labels_continuous_keras,
                    input_shape=(100, 100, 4),
                    batch_size=batch_size,
                    shuffle=False,
                    label_mapping=index_to_label,
                    undersample=False  # No undersampling for validation
                )
                # Update num_samples_val based on validated files
                num_samples_val = len(val_generator.file_names)
                
                if num_samples_val == 0:
                    print("WARNING: All validation files were invalid! Will use validation_split instead.")
                    val_generator = None
            except ValueError as e:
                print(f"WARNING: Could not create validation generator: {e}")
                print("Will use validation_split instead.")
                val_generator = None
                num_samples_val = 0
    except ValueError as e:
        print(f"ERROR: Failed to create data generators: {e}")
        sys.exit(1)
    
    # Print information about undersampling if enabled
    if args.undersample:
        print("\nUndersampling is enabled:")
        print(f"  Ratio: {args.undersample_ratio} (1.0 = fully balanced, 0.0 = no balancing)")
        print("  Note: Undersampling reduces the number of training samples, which affects:")
        print("    1. The effective number of steps per epoch will be lower")
        print("    2. Each epoch will use a different random subset of majority classes")
        print("    3. This helps combat class imbalance but may require more epochs for convergence")
        print(f"    4. Consider increasing the number of epochs or decreasing the undersampling ratio")
        print("    5. Class weights are still applied in addition to undersampling\n")
    
    # Save the index_mapping for later use in prediction
    with open(f'{name_base}_index_mapping.pkl', 'wb') as f:
        pickle.dump({
            'index_mapping': index_mapping,
            'reverse_mapping': {new_idx: old_idx for old_idx, new_idx in index_mapping.items()}
        }, f)
    
    print(f"\nIndex mapping for Keras compatibility (saved to {name_base}_index_mapping.pkl):")
    for old_idx, new_idx in index_mapping.items():
        print(f"  Original index {old_idx} → Keras index {new_idx}")
    
    # Print Keras-compatible class weights
    print("\nKeras-compatible class weights:")
    for idx in range(actual_unique_classes):
        # Ensure we have weights for all classes 0 to actual_unique_classes-1
        if idx not in class_weights:
            class_weights[idx] = 1.0
            print(f"  Class {idx}: {class_weights[idx]:.4f} (added default weight)")
        else:
            print(f"  Class {idx}: {class_weights[idx]:.4f}")
            
    # Verify class weights have exactly the right keys
    class_weight_keys = set(class_weights.keys())
    expected_keys = set(range(actual_unique_classes))
    if class_weight_keys != expected_keys:
        print("Adjusting class weights to match expected keys exactly...")
        # Create a new dictionary with exactly the right keys
        adjusted_weights = {idx: class_weights.get(idx, 1.0) for idx in range(actual_unique_classes)}
        class_weights = adjusted_weights
        print(f"Final class weights: {class_weights}")

    # Print original mapping info for reference
    print("\nLabel to index mapping (original):")
    for label, idx in label_to_index.items():
        print(f"  Original label {label} -> Index {idx}")

    # Safely print sample batch shape for verification
    try:
        x_sample, y_sample = train_generator[0]
        print(f"Training batch shape: {x_sample.shape}, labels shape: {y_sample.shape}")
        print(f"Sample labels: {y_sample[:5]}")
    except Exception as e:
        print(f"WARNING: Could not get sample batch: {e}")
        print("Continuing with training regardless...")
    
    # Verify that shuffling is working by showing the first few indices
    print("\nVerifying data shuffling:")
    print(f"First 10 indices of training data: {train_generator.indices[:10]}")
    # Get a second batch to verify different indices are used
    x_sample2, y_sample2 = train_generator[1]
    print(f"Second batch sample labels: {y_sample2[:5]}")
    # Simulate an epoch end to trigger reshuffling
    train_generator.on_epoch_end()
    print(f"After reshuffling, first 10 indices: {train_generator.indices[:10]}")
    
    # Also verify that validation and test data are not being shuffled
    if val_generator is not None:
        print(f"First 10 indices of validation data (should be ordered 0-9): {val_generator.indices[:10]}")
    else:
        print("No validation generator available - will use validation_split from training data")

    # Create custom model with inception-resnet-v2 style modules
    # with the number of classes equal to the unique labels in training data
    print(f"Creating model with {actual_unique_classes} output classes and all 4 antennas as channels")
    csi_model = create_model(input_shape=(100, 100, 4), num_classes=actual_unique_classes)
    csi_model.summary()
    
    # Freeze BatchNormalization layers to prevent issues with small batch sizes
    # Use a more robust approach that can find nested BN layers
    def freeze_bn_layers(model):
        """Recursively freeze all BatchNormalization layers in the model"""
        count = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                count += 1
            
            # If this layer has sub-layers (nested model), recursively freeze those too
            if hasattr(layer, 'layers') and layer.layers:
                count += freeze_bn_layers(layer)
        return count
    
    batch_norm_count = freeze_bn_layers(csi_model)
    print(f"Froze {batch_norm_count} BatchNormalization layers to improve stability with small batches")

    # Define optimizer and loss function with improved learning rate schedule
    # As recommended by expert reviewer
    initial_learning_rate = 0.001
    
    # Add learning rate decay schedule with new parameters
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)
    
    # Use the schedule with Adam optimizer
    optimiz = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True  # Enable AMSGrad variant for better convergence
    )
    
    # Use sparse categorical crossentropy loss with from_logits=True
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Check if focal loss should be used
    use_focal_loss = False  # Set to True to use focal loss
    if use_focal_loss:
        print("Using Focal Loss for handling class imbalance...")
        loss = focal_loss(gamma=2.0, alpha=0.25)
    
    # Get the number of classes from the labels
    num_classes = len(np.unique(labels_train_selected_expanded))
    print(f"Number of classes detected: {num_classes}")
    
    # Calculate class weights to handle imbalance
    use_class_weights = True  # Set to True to enable class weights
    class_weight_dict = None
    
    if use_class_weights:
        print("Computing class weights to handle class imbalance...")
        class_weight_dict = compute_class_weights(labels_train_selected_expanded, num_classes)
        
        # Print the class weights for reference
        print("Class weights that will be applied during training:")
        for class_idx, weight in class_weight_dict.items():
            print(f"  Class {class_idx}: {weight:.4f}")
    
    # Create metrics
    metrics = create_per_class_metrics(num_classes)
    
    # Compile the model with our improved settings
    csi_model.compile(optimizer=optimiz, loss=loss, metrics=metrics)

    # Dataset statistics - use the previously defined sample count variables
    lab, count = np.unique(labels_train_selected_expanded, return_counts=True)
    
    # Check if validation and test sets are empty
    if num_samples_val > 0:
        lab_val, count_val = np.unique(labels_val_selected_expanded, return_counts=True)
    else:
        lab_val, count_val = np.array([]), np.array([])
        print("WARNING: No validation samples available! Will use a subset of training data for validation.")
        
    if num_samples_test > 0:
        lab_test, count_test = np.unique(labels_test_selected_expanded, return_counts=True)
    else:
        lab_test, count_test = np.array([]), np.array([])
        print("WARNING: No test samples available! Evaluation will be limited to training data.")
    
    # Display dataset information 
    print(f"Training samples: {num_samples_train}")
    print(f"Validation samples: {num_samples_val}")
    print(f"Test samples: {num_samples_test}")
    
    print("\nDetailed training label distribution:")
    label_counts = {}
    for lbl in np.unique(labels_train_selected_expanded):
        count = np.sum(np.array(labels_train_selected_expanded) == lbl)
        label_counts[int(lbl)] = count
        print(f"  Label {lbl}: {count} samples")
    
    print("\nDetailed validation label distribution:")
    for lbl in np.unique(labels_val_selected_expanded):
        count = np.sum(np.array(labels_val_selected_expanded) == lbl)
        print(f"  Label {lbl}: {count} samples")
    
    print("\nDetailed test label distribution:")
    for lbl in np.unique(labels_test_selected_expanded):
        count = np.sum(np.array(labels_test_selected_expanded) == lbl)
        print(f"  Label {lbl}: {count} samples")
    
    # Fix the lab/count iterable error - more robust type checking
    print(f"\nTraining labels and counts: ")
    print(f"  lab type: {type(lab)}")
    print(f"  count type: {type(count)}")
    
    try:
        if hasattr(lab, '__iter__') and hasattr(count, '__iter__'):
            # Both are iterable
            for l, c in zip(lab, count):
                print(f"  Label {l}: {c} samples")
        elif isinstance(lab, (int, np.integer)) and isinstance(count, (int, np.integer)):
            # Both are single integers
            print(f"  Label {lab}: {count} samples")
        else:
            # Mixed types or other cases
            print(f"  Unable to print label counts due to incompatible types")
    except Exception as e:
        print(f"  Error processing label counts: {e}")
    
    # Data loading diagnostics
    print("\nData loading diagnostics:")
    print(f"  Number of subdir folders: {len(subdirs_training.split(','))}")
    print(f"  Subdirs: {subdirs_training}")
    
    # Debug raw data counts before filtering/expansion
    print("\nRaw data counts before filtering/expansion:")
    print(f"  Raw training files: {len(all_files_train)}")
    print(f"  Raw training labels: {len(labels_train)}")
    
    # Check first few raw training files
    if len(all_files_train) > 0:
        print(f"\nFirst 5 raw training files:")
        for i in range(min(5, len(all_files_train))):
            print(f"  {i}: {all_files_train[i]}")
    
    # Check first few raw training labels
    if len(labels_train) > 0:
        print(f"\nFirst 10 raw training labels:")
        for i in range(min(10, len(labels_train))):
            print(f"  {i}: {labels_train[i]}")
    
    # Check file filtering process
    print("\nChecking the file filtering process:")
    print(f"  Labels considered shape: {labels_considered.shape}")
    print(f"  Labels considered: {labels_considered}")
    
    # Count files that match the labels_considered criteria
    matching_count = sum(1 for label in labels_train if label in labels_considered)
    print(f"  Files with matching labels: {matching_count} out of {len(labels_train)}")
    
    # Debug the file selection process
    print("\nFile selection process:")
    print(f"  Selected files (after filtering): {len(file_train_selected)}")
    print(f"  Selected labels (after filtering): {len(labels_train_selected)}")
    
    # Check the expansion process
    print("\nExpansion process:")
    print(f"  Pre-expansion files: {len(file_train_selected)}")
    print(f"  num_antennas: {num_antennas}")
    print(f"  Expected post-expansion: {len(file_train_selected) * num_antennas}")
    print(f"  Actual post-expansion: {len(file_train_selected_expanded)}")
    
    # Print a sample of the expanded data
    if len(file_train_selected_expanded) > 0:
        print(f"\nSample of files (first 5):")
        for i in range(min(5, len(file_train_selected_expanded))):
            print(f"  {i}: File={file_train_selected_expanded[i]}, Label={labels_train_selected_expanded[i]}")
    
    # Examine if we're only loading one file per domain
    print("\nChecking for potential domain-specific loading patterns:")
    for domain_idx, sdir in enumerate(subdirs_training.split(',')):
        try:
            # Load the original files and check how many are selected
            name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix
            name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
            
            with open(name_f, "rb") as fp:
                domain_files = pickle.load(fp)
            
            with open(name_labels, "rb") as fp:
                domain_labels = pickle.load(fp)
            
            # Count how many files from this domain are selected
            domain_files_in_selected = sum(1 for f in file_train_selected if any(f == df for df in domain_files))
            
            print(f"  Domain {sdir}: {len(domain_files)} total files, {domain_files_in_selected} selected")
            
            # Check if only one file per domain is selected
            if domain_files_in_selected == 1:
                print(f"    WARNING: Only one file selected from domain {sdir}!")
                # Find which file it is
                for i, f in enumerate(file_train_selected):
                    if any(f == df for df in domain_files):
                        print(f"    Selected file: {f}")
                        print(f"    Label: {labels_train_selected[i]}")
                        break
            
        except Exception as e:
            print(f"  Error analyzing domain {sdir}: {e}")
    
    # Safety check to prevent training with empty datasets
    if num_samples_train == 0:
        print("Error: No training samples found. Cannot proceed with training.")
        exit(1)
    
    # Define improved callbacks for better training
    # 1. Early stopping with increased patience and monitoring validation accuracy
    callback_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Increased from 3 to give model more time to converge
        min_delta=0.001,  # Minimum change to qualify as improvement
        verbose=1,
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )
    
    # 2. Model checkpoint to save the best model
    name_model = name_base + '_' + str(csi_act) + '_network.keras'
    callback_save = tf.keras.callbacks.ModelCheckpoint(
        name_model,
        save_freq='epoch',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
    
    # 3. Reduce learning rate when validation metrics plateau
    callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Reduce learning rate by factor of 0.2 when plateau is detected
        patience=3,
        min_delta=0.001,
        verbose=1,
        min_lr=0.000001  # Minimum learning rate
    )
    
    # 4. TensorBoard logging
    log_dir = f"logs/{name_base}_{str(csi_act)}_{int(time.time())}"
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )

    # Check that the generators are working
    print("Checking data generators...")
    try:
        test_batch = train_generator[0]
        print(f"Training batch shape: {test_batch[0].shape}, labels shape: {test_batch[1].shape}")
    except Exception as e:
        print(f"Error testing data generator: {e}")
        sys.exit(1)
    
    # Determine the number of epochs based on undersampling
    if args.undersample:
        # Increase epochs when undersampling to compensate for seeing fewer samples per epoch
        num_epochs = 4  # Increased from 30 to give model more training iterations with undersampled data
        print(f"Using {num_epochs} epochs (increased from default 30) due to undersampling")
    else:
        num_epochs = 3  # Default value
    
    # If validation set is empty, create a validation split from training data
    if num_samples_val == 0 or val_generator is None:
        print("Creating validation split from training data since no validation set is available...")
        
        # Manually split the training data - use 20% for validation
        val_split_ratio = 0.2
        
        # Compute the number of samples for validation
        n_val = int(len(train_generator.file_names) * val_split_ratio)
        
        # Make a copy of the training file names and labels
        all_train_files = train_generator.file_names.copy()
        all_train_labels = train_generator.labels.copy()
        
        # Shuffle the indices to ensure a random split
        split_indices = np.arange(len(all_train_files))
        np.random.shuffle(split_indices)
        
        # Split the indices
        val_indices = split_indices[:n_val]
        train_indices = split_indices[n_val:]
        
        # Create new file and label lists for validation
        val_files = [all_train_files[i] for i in val_indices]
        val_labels = [all_train_labels[i] for i in val_indices]
        
        # Create new file and label lists for training
        new_train_files = [all_train_files[i] for i in train_indices]
        new_train_labels = [all_train_labels[i] for i in train_indices]
        
        # Create a new training generator with the reduced dataset
        train_generator = CustomDataGenerator(
            new_train_files,
            new_train_labels,
            input_shape=(100, 100, 4),
            batch_size=batch_size,
            shuffle=True,
            label_mapping=index_to_label,
            undersample=args.undersample,
            undersample_ratio=args.undersample_ratio
        )
        
        # Create a validation generator with the held-out data
        val_generator = CustomDataGenerator(
            val_files,
            val_labels,
            input_shape=(100, 100, 4),
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            label_mapping=index_to_label,
            undersample=False  # No undersampling for validation
        )
        
        print(f"Manual split created: {len(new_train_files)} training samples, {len(val_files)} validation samples")
        
        # Apply SMOTE to balance classes if enabled
        use_smote = True  # Set to True to use SMOTE
        
        # We need to adapt the data generator to work with SMOTE
        if use_smote:
            print("\nApplying SMOTE to balance the training data...")
            
            # Load all training data into memory for SMOTE processing
            X_train = []
            y_train = []
            
            # Load all data from files
            for i, file_path in enumerate(train_generator.file_names):
                try:
                    # Different loading approach based on file extension
                    if file_path.endswith('.bin'):
                        # Handle our custom binary format
                        with open(file_path, 'rb') as f:
                            # Read the shape information
                            shape = np.fromfile(f, dtype=np.int32, count=3)
                            # Read the data
                            data = np.fromfile(f, dtype=np.float32)
                            # Reshape according to the stored shape
                            data = data.reshape(shape)
                    else:
                        # Regular pickle file loading
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        # Transpose to (340, 100, 4)
                        data = np.transpose(data, (2, 1, 0))  

                    # Reshape data for SMOTE (which requires 2D input)
                    X_train.append(data.reshape(1, -1)[0])  # Flatten to 1D array
                    y_train.append(train_generator.labels[i])
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    # Continue with remaining files instead of failing completely
                    continue
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            if len(X_train) > 0:
                # Check if we have enough samples from each class for SMOTE
                unique_classes, class_counts = np.unique(y_train, return_counts=True)
                min_samples_per_class = np.min(class_counts)
                
                if len(unique_classes) >= 2 and min_samples_per_class >= 2:
                    # Apply SMOTE to create balanced dataset
                    X_resampled, y_resampled = apply_smote(X_train, y_train)
                    
                    # Check the shapes to ensure they're compatible
                    print(f"SMOTE X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")
                    
                    # Reshape back to original dimensions
                    orig_shape = (100, 100, 4)  # This should match your model's input shape
                    X_resampled_reshaped = np.array([x.reshape(orig_shape) for x in X_resampled])
                    
                    # Ensure labels are in the right format for sparse categorical crossentropy
                    # Should be (n_samples,) and not (n_samples, 1)
                    if len(y_resampled.shape) > 1 and y_resampled.shape[1] == 1:
                        y_resampled = y_resampled.flatten()
                    
                    print(f"Resampled data shape: {X_resampled.shape}, labels shape: {y_resampled.shape}")
                    
                    # Reshape back to original dimensions
                    orig_shape = (100, 100, 4)  # This should match your model's input shape
                    X_resampled_reshaped = np.array([x.reshape(orig_shape) for x in X_resampled])
                    
                    # Use the resampled data
                    # Train with SMOTE-balanced data
                    results = csi_model.fit(
                        X_resampled_reshaped,
                        y_resampled,
                        epochs=num_epochs,
                        validation_data=val_generator,
                        callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
                        batch_size=batch_size,
                        verbose=1,
                        class_weight=class_weight_dict if use_class_weights else None
                    )
                else:
                    print(f"Error: Not enough samples for SMOTE. Found {len(unique_classes)} classes with minimum {min_samples_per_class} samples per class.")
                    print("Falling back to using the generator without SMOTE.")
                    # Fall back to using the generator without SMOTE
                    results = csi_model.fit(
                        train_generator,
                        epochs=num_epochs,
                        validation_data=val_generator,
                        callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
                        class_weight=class_weight_dict if use_class_weights else None,
                        verbose=1
                    )
            else:
                print("Error: No training data available for SMOTE")
                # Fall back to using the generator without SMOTE
                results = csi_model.fit(
                    train_generator,
                    epochs=num_epochs,
                    validation_data=val_generator,
                    callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
                    class_weight=class_weight_dict if use_class_weights else None,
                    verbose=1
                )
        else:
            # Original fit approach without SMOTE but with class weights
            results = csi_model.fit(
                train_generator,
                epochs=num_epochs,
                validation_data=val_generator,
                callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
                class_weight=class_weight_dict if use_class_weights else None,
                verbose=1
            )
    else:
        # Train with class weights and all callbacks using separate validation data
        results = csi_model.fit(
            train_generator,
            epochs=num_epochs,
            validation_data=val_generator,
            callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
            class_weight=class_weight_dict if use_class_weights else None,
            verbose=1
        )

    # Verify label mapping consistency
    print("\nVerifying label mapping consistency:")
    # Check that all continuous labels are in the expected range
    try:
        max_label_idx = max(index_to_label.keys())
        unique_train_labels = np.unique(train_labels_continuous)
        unique_val_labels = np.unique(val_labels_continuous)
        unique_test_labels = np.unique(test_labels_continuous)
        
        print(f"Max label index in mapping: {max_label_idx}")
        print(f"Unique continuous labels in training data: {unique_train_labels}")
        print(f"Unique continuous labels in validation data: {unique_val_labels}")
        print(f"Unique continuous labels in test data: {unique_test_labels}")
        
        # Check for labels that might be missing in the mapping
        missing_train = [label for label in unique_train_labels if label not in index_to_label]
        if missing_train:
            print(f"Warning: Some training labels are missing in index_to_label mapping: {missing_train}")
            # Add missing labels with numeric values for consistency
            for label in missing_train:
                # Use the label value itself as the label to maintain numeric consistency
                index_to_label[label] = int(label) if isinstance(label, (int, np.integer)) else 0
                print(f"  Added missing label {label} → Value {index_to_label[label]}")
        
        # Verify that all original labels can be mapped back correctly
        original_train_labels = []
        for idx in unique_train_labels:
            if idx in index_to_label:
                label = index_to_label[idx]
                # Ensure numeric labels
                if isinstance(label, (int, np.integer)):
                    original_train_labels.append(int(label))
                elif isinstance(label, str) and label.isdigit():
                    original_train_labels.append(int(label))
                else:
                    original_train_labels.append(0)  # Default numeric placeholder
            else:
                print(f"Warning: Label index {idx} not found in mapping, using placeholder")
                original_train_labels.append(0)  # Use numeric placeholder
                
        print(f"Original training labels after mapping and unmapping: {original_train_labels}")
    except Exception as e:
        print(f"Warning: Error during label mapping verification: {e}")
        print("This is non-critical and training will continue.")
    
    # For inference, create a model that includes the softmax
    inference_model = tf.keras.Sequential([
        csi_model,
        tf.keras.layers.Softmax()
    ])

    # Save both models
    csi_model.save(name_model)
    inference_model.save(name_model.replace('.keras', '_inference.keras'))

    # Function to convert Keras indices back to original indices then to original labels
    def convert_predictions_to_original_labels(predicted_indices, index_to_label_map):
        """
        Convert predicted indices to their original labels using the mapping.
        Ensures all returned labels are numeric for consistency.
        """
        converted_predictions = []
        
        # First, convert Keras indices back to original indices
        reverse_mapping = {new_idx: old_idx for old_idx, new_idx in index_mapping.items()}
        
        # Ensure index_to_label dict has numeric values where possible
        for key in list(index_to_label_map.keys()):
            if not isinstance(index_to_label_map[key], (int, np.integer)):
                try:
                    index_to_label_map[key] = int(index_to_label_map[key])
                except:
                    pass  # Keep as is if not convertible
        
        for keras_idx in predicted_indices:
            # Step 1: Convert Keras index back to original index
            original_idx = reverse_mapping.get(int(keras_idx), keras_idx)
            
            # Step 2: Convert original index to original label
            if original_idx in index_to_label_map:
                label = index_to_label_map[original_idx]
                # Ensure numeric consistency
                if isinstance(label, (int, np.integer)):
                    converted_predictions.append(int(label))
                else:
                    # Try to convert to integer if it's a string representation of a number
                    try:
                        converted_predictions.append(int(label))
                    except:
                        # If it's a string like 'Unknown-5', extract the number
                        if isinstance(label, str) and label.startswith('Unknown-'):
                            try:
                                num = int(label.split('-')[1])
                                converted_predictions.append(num)
                            except:
                                converted_predictions.append(0)  # Default
                        else:
                            converted_predictions.append(0)  # Default for non-numeric
            else:
                # If index not found, use the original index as the label (numeric fallback)
                converted_predictions.append(int(original_idx))
                print(f"Warning: Index {original_idx} not found in mapping, using it as the label")
        
        return np.array(converted_predictions, dtype=np.int32)

    # EVALUATION SECTION
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print("NOTE: While we use undersampling during training for balanced class distribution,")
    print("      evaluation is performed on the complete dataset to get full performance metrics.")
    print("      This approach ensures training is balanced but evaluation is comprehensive.")
    print("="*80 + "\n")
    
    # Use inference model for predictions
    print("Evaluating on training data...")
    # Use the original labels (not expanded) since our data generator now processes all antennas together
    train_labels_original = np.array(labels_train_selected)
    # We don't need to remap these since we're comparing original labels directly
    
    # Create a separate evaluation generator without undersampling for prediction
    # This ensures we can evaluate on the entire dataset
    print("Creating evaluation generator without undersampling...")
    eval_generator = CustomDataGenerator(
        file_train_selected_expanded, 
        train_labels_continuous_keras,  # Use remapped indices
        input_shape=(100, 100, 4),
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        label_mapping=index_to_label,
        undersample=False  # No undersampling for evaluation
    )
    
    train_prediction_list = []
    for i in range(len(eval_generator)):
        batch_x, _ = eval_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        train_prediction_list.append(batch_pred)

    # Limit to the number of unique samples (original samples, not expanded)
    train_prediction_list = np.vstack(train_prediction_list)[:len(train_labels_original)]
    train_labels_pred_continuous = np.argmax(train_prediction_list, axis=1)
    # These are now in Keras index space, we need to convert back to original labels
    train_labels_pred = convert_predictions_to_original_labels(train_labels_pred_continuous, index_to_label)
    
    print(f"Train prediction shape: {train_prediction_list.shape}")
    print(f"Train labels shape: {train_labels_original.shape}")
    
    # Print types of labels to verify consistency
    print("\nCHECKING LABEL TYPES:")
    print(f"Train original labels type: {type(train_labels_original[0])} ({train_labels_original[0]})")
    print(f"Train predicted labels type: {type(train_labels_pred[0])} ({train_labels_pred[0]})")
    
    # Convert labels to numeric if they're not already - for confusion matrix calculation
    train_labels_original_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 for label in train_labels_original])
    train_labels_pred_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 for label in train_labels_pred])
    
    # Get all unique label values for consistent matrix dimensions
    all_train_labels = np.unique(np.concatenate([train_labels_original_numeric, train_labels_pred_numeric]))
    
    print(f"Train confusion matrix with {len(all_train_labels)} classes:")
    train_confusion_matrix = confusion_matrix(train_labels_original_numeric, train_labels_pred_numeric)
    print(train_confusion_matrix)
    print("\n")
    
    # Predict on validation data if available
    val_labels_original_numeric = None
    val_labels_pred_numeric = None
    
    if num_samples_val > 0:
        print("Evaluating on validation data...")
        # Use the original labels (not expanded)
        val_labels_original = np.array(labels_val_selected)
        # We don't need to remap these since we're comparing original labels directly
        
        try:
            # Create evaluation generator for validation data without undersampling
            val_eval_generator = CustomDataGenerator(
                file_val_selected_expanded,
                val_labels_continuous_keras,  # Use remapped indices
                input_shape=(100, 100, 4),
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle for evaluation
                label_mapping=index_to_label,
                undersample=False  # No undersampling for evaluation
            )
            
            # Check if we have any valid files after validation
            if len(val_eval_generator.file_names) == 0:
                print("No valid validation files found. Skipping validation evaluation.")
            else:
                val_prediction_list = []
                for i in range(len(val_eval_generator)):
                    batch_x, _ = val_eval_generator[i]
                    batch_pred = inference_model.predict(batch_x, verbose=0)
                    val_prediction_list.append(batch_pred)
                
                # Limit to the number of available validation samples
                available_val_samples = min(len(val_labels_original), len(val_eval_generator.file_names))
                
                if available_val_samples > 0 and len(val_prediction_list) > 0:
                    val_prediction_list = np.vstack(val_prediction_list)[:available_val_samples]
                    val_labels_pred_continuous = np.argmax(val_prediction_list, axis=1)
                    val_labels_pred = convert_predictions_to_original_labels(val_labels_pred_continuous, index_to_label)
                    
                    print(f"Val prediction shape: {val_prediction_list.shape}")
                    print(f"Val labels shape: {val_labels_original[:available_val_samples].shape}")
                    
                    # Same for validation - use only available samples
                    val_labels_original_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 
                                                          for label in val_labels_original[:available_val_samples]])
                    val_labels_pred_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 
                                                      for label in val_labels_pred])
                    
                    all_val_labels = np.unique(np.concatenate([val_labels_original_numeric, val_labels_pred_numeric]))
                    
                    print(f"Validation confusion matrix with {len(all_val_labels)} classes:")
                    val_confusion_matrix = confusion_matrix(val_labels_original_numeric, val_labels_pred_numeric)
                    print(val_confusion_matrix)
                    print("\n")
                else:
                    print("No predictions generated for validation data - skipping evaluation.")
        except Exception as e:
            print(f"Error during validation evaluation: {e}")
            print("Skipping validation evaluation.")
    else:
        print("Skipping validation evaluation - no validation samples available.")
    
    # Predict on test data if available
    test_labels_original_numeric = None
    test_labels_pred_numeric = None
    
    if num_samples_test > 0:
        print("Evaluating on test data...")
        # Use the original labels (not expanded)
        test_labels_original = np.array(labels_test_selected)
        # We don't need to remap these since we're comparing original labels directly
        
        try:
            # Create evaluation generator for test data without undersampling
            test_eval_generator = CustomDataGenerator(
                file_test_selected_expanded,
                test_labels_continuous_keras,  # Use remapped indices
                input_shape=(100, 100, 4),
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle for evaluation
                label_mapping=index_to_label,
                undersample=False  # No undersampling for evaluation
            )
            
            # Check if we have any valid files after validation
            if len(test_eval_generator.file_names) == 0:
                print("No valid test files found. Using training data for evaluation instead.")
                test_labels_original_numeric = train_labels_original_numeric
                test_labels_pred_numeric = train_labels_pred_numeric
            else:
                test_prediction_list = []
                for i in range(len(test_eval_generator)):
                    batch_x, _ = test_eval_generator[i]
                    batch_pred = inference_model.predict(batch_x, verbose=0)
                    test_prediction_list.append(batch_pred)
                
                # Limit to the number of available test samples
                available_test_samples = min(len(test_labels_original), len(test_eval_generator.file_names))
                
                if available_test_samples > 0 and len(test_prediction_list) > 0:
                    test_prediction_list = np.vstack(test_prediction_list)[:available_test_samples]
                    test_labels_pred_continuous = np.argmax(test_prediction_list, axis=1)
                    test_labels_pred = convert_predictions_to_original_labels(test_labels_pred_continuous, index_to_label)
                    
                    print(f"Test prediction shape: {test_prediction_list.shape}")
                    print(f"Test labels shape: {test_labels_original[:available_test_samples].shape}")

                    # Calculate metrics using original (non-expanded) labels
                    # Ensure both arrays are of the same type (numeric)
                    test_labels_original_numeric = np.array([int(l) if isinstance(l, (int, np.integer)) else 0 
                                                          for l in test_labels_original[:available_test_samples]])
                    test_labels_pred_numeric = np.array([int(l) if isinstance(l, (int, np.integer)) else 0 
                                                      for l in test_labels_pred])
                else:
                    print("No predictions generated for test data. Using training data for evaluation instead.")
                    test_labels_original_numeric = train_labels_original_numeric
                    test_labels_pred_numeric = train_labels_pred_numeric
        except Exception as e:
            print(f"Error during test evaluation: {e}")
            print("Using training data for final evaluation metrics.")
            test_labels_original_numeric = train_labels_original_numeric
            test_labels_pred_numeric = train_labels_pred_numeric
        
        # Calculate test metrics
        conf_matrix = confusion_matrix(test_labels_original_numeric, test_labels_pred_numeric)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            test_labels_original_numeric,
            test_labels_pred_numeric,
            labels=np.unique(np.concatenate([test_labels_original_numeric, test_labels_pred_numeric])),
            zero_division=0
        )
        accuracy = accuracy_score(test_labels_original_numeric, test_labels_pred_numeric)
    else:
        print("Skipping test evaluation - no test samples available.")
        print("Using training data for final evaluation metrics.")
        # If no test data is available, use training data for final metrics
        test_labels_original_numeric = train_labels_original_numeric
        test_labels_pred_numeric = train_labels_pred_numeric
        
        # Calculate metrics on training data
        conf_matrix = confusion_matrix(test_labels_original_numeric, test_labels_pred_numeric)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            test_labels_original_numeric,
            test_labels_pred_numeric,
            labels=np.unique(np.concatenate([test_labels_original_numeric, test_labels_pred_numeric])),
            zero_division=0
        )
        accuracy = accuracy_score(test_labels_original_numeric, test_labels_pred_numeric)
        print("NOTE: These metrics are based on training data and may overestimate model performance!")

    # Since we're now processing all antennas together, we don't need to merge antennas
    # as was done in the original code. The predictions already incorporate all antennas.
    labels_true_merge = test_labels_original_numeric
    pred_max_merge = test_labels_pred_numeric
    
    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=np.unique(np.concatenate([labels_true_merge, pred_max_merge])))
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=np.unique(np.concatenate([labels_true_merge, pred_max_merge])), zero_division=0)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                           'accuracy_single': accuracy,
                           'precision_single': precision,
                           'recall_single': recall,
                           'fscore_single': fscore,
                           'conf_matrix_max_merge': conf_matrix_max_merge,
                           'accuracy_max_merge': accuracy_max_merge,
                           'precision_max_merge': precision_max_merge,
                           'recall_max_merge': recall_max_merge,
                           'fscore_max_merge': fscore_max_merge}
    unique_id = hashlib.md5(f"{csi_act}_{subdirs_training}".encode()).hexdigest()[:8]
    name_file = f'./outputs/test_{unique_id}_b{bandwidth}_sb{sub_band}.txt'

    # name_file = './outputs/test_' + str(csi_act) + '_' + subdirs_training + '_band_' + str(bandwidth) + '_subband_' + \
    #             str(sub_band) + suffix
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)

    # Print final results
    print("\nModel Performance Summary:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Average F1-Score: {np.mean(fscore):.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Clean up resources
    tf.keras.backend.clear_session()
    gc.collect()
    
    # We need to reopen the file for writing since it was previously opened for binary writing
    name_file_txt = name_file.replace('.txt', '_metrics.txt')
    
    # Save the formatted metrics to a text file for easier reading
    with open(name_file_txt, 'w') as f:
        f.write(f"Test Metrics:\n")
        f.write(f"Average Precision: {np.mean(precision):.4f}\n")
        f.write(f"Average Recall: {np.mean(recall):.4f}\n")
        f.write(f"Average F1 Score: {np.mean(fscore):.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        
        # Add per-class metrics if available
        f.write(f"\nPer-class Metrics:\n")
        unique_labels = np.unique(np.concatenate([test_labels_original_numeric, test_labels_pred_numeric]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                f.write(f"Class {label}:\n")
                f.write(f"  Precision: {precision[i]:.4f}\n")
                f.write(f"  Recall: {recall[i]:.4f}\n")
                f.write(f"  F1 Score: {fscore[i]:.4f}\n")
                f.write(f"  Samples: {np.sum(test_labels_original_numeric == label)}\n")
                
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    # Print success message
    print(f"\nResults saved to {name_file_txt}")
    print("Training and evaluation complete!")

    
