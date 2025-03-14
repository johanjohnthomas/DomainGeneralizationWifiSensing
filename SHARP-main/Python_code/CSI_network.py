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
    """Compute class weights for imbalanced dataset using inverse frequency."""
    # Get unique labels that exist in the dataset
    unique_labels = np.unique(labels)
    
    # Use scikit-learn's compute_class_weight with 'balanced' setting
    # This automatically computes inverse frequency weights
    sk_weights = compute_class_weight(class_weight='balanced', 
                                     classes=unique_labels, 
                                     y=labels)
    
    # Create a dictionary to store the weights
    weights = {}
    for i, label_idx in enumerate(unique_labels):
        weights[label_idx] = sk_weights[i]
    
    # Add default weights for any classes not in the dataset
    for idx in range(num_classes):
        if idx not in weights:
            print(f"Warning: Class {idx} not found in dataset, using default weight 1.0")
            weights[idx] = 1.0
    
    # Log the weights
    print("Class weights based on inverse frequency:")
    print("  Class weights:", weights)
            
    return weights

def create_model(input_shape=(340, 100, 4), num_classes=6):
    """Create the CSI network model."""
    input_network = tf.keras.layers.Input(shape=input_shape)
    
    # Add L2 regularization to all convolutional layers
    regularizer = tf.keras.regularizers.l2(0.001)
    
    # First branch - 3x3 convolutions
    conv3_1 = tf.keras.layers.Conv2D(3, (3, 3), padding='same', 
                                    kernel_regularizer=regularizer,
                                    name='1stconv3_1_res_a')(input_network)
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
                                    name='1stconv2_1_res_a')(input_network)
    conv2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
    conv2_1 = tf.keras.layers.Activation('relu', name='activation')(conv2_1)
    
    # Third branch - max pooling
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d')(input_network)
    
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
    
    # Increase dropout rate from 0.5 to 0.6 for better regularization
    drop = tf.keras.layers.Dropout(0.6, name='dropout')(flat)
    
    # Add regularization to the final dense layer
    dense2 = tf.keras.layers.Dense(num_classes, 
                                 kernel_regularizer=regularizer,
                                 name='dense2')(drop)
    
    # Create model
    model = tf.keras.Model(inputs=input_network, outputs=dense2, name='csi_model')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for training')
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

    subdirs_training = args.subdirs  # string
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

    for sdir in subdirs_training.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_train = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix
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
            # Replicate the label for each file in this domain
            domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
            labels_train.extend(domain_labels_expanded)
            all_files_train.extend(domain_files)

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
            # Replicate the label for each file in this domain
            domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
            labels_val.extend(domain_labels_expanded)
            all_files_val.extend(domain_files)

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
            # Replicate the label for each file in this domain
            domain_labels_expanded = [domain_labels[0] for _ in range(len(domain_files))]
            labels_test.extend(domain_labels_expanded)
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
    
    # Create a custom data generator for training instead of using TensorFlow's dataset API
    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, file_names, labels, input_shape=(340, 100, 4), batch_size=16, shuffle=True):
            self.file_names = file_names
            self.labels = labels
            self.input_shape = input_shape
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(file_names))
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __len__(self):
            return int(np.ceil(len(self.file_names) / self.batch_size))

        def __getitem__(self, idx):
            batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
            batch_files = [self.file_names[i] for i in batch_indices]
            batch_labels = [self.labels[i] for i in batch_indices]
            
            batch_x = []
            for file_path in batch_files:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)  # Shape (4, 100, 340)
                data = np.transpose(data, (2, 1, 0))  # Transpose to (340, 100, 4)
                batch_x.append(data)
            
            return np.array(batch_x), np.array(batch_labels)

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
    
    # Convert labels to continuous indices
    train_labels_continuous = np.array([label_to_index[label] for label in labels_train_selected_expanded])
    
    # For validation and test, we need to filter out samples with labels not in the training set
    # First, create masks for valid samples
    valid_val_mask = np.array([label in label_to_index for label in labels_val_selected_expanded])
    valid_test_mask = np.array([label in label_to_index for label in labels_test_selected_expanded])
    
    # Filter validation samples
    if not all(valid_val_mask):
        print(f"Filtering out {np.sum(~valid_val_mask)} validation samples with labels not in training set")
        file_val_selected_expanded = [f for i, f in enumerate(file_val_selected_expanded) if valid_val_mask[i]]
        labels_val_selected_expanded = [l for i, l in enumerate(labels_val_selected_expanded) if valid_val_mask[i]]
    
    # Filter test samples
    if not all(valid_test_mask):
        print(f"Filtering out {np.sum(~valid_test_mask)} test samples with labels not in training set")
        file_test_selected_expanded = [f for i, f in enumerate(file_test_selected_expanded) if valid_test_mask[i]]
        labels_test_selected_expanded = [l for i, l in enumerate(labels_test_selected_expanded) if valid_test_mask[i]]
    
    # Now convert the filtered validation and test labels
    val_labels_continuous = np.array([label_to_index[label] for label in labels_val_selected_expanded])
    test_labels_continuous = np.array([label_to_index[label] for label in labels_test_selected_expanded])

    # Calculate class weights based on continuous indices
    num_classes = len(unique_labels)
    class_weights_raw = compute_class_weights(train_labels_continuous, num_classes)
    
    # Keras requires class_weight to be a dictionary with keys from 0 to num_classes-1
    print("\nAdjusting class weights for Keras (requires consecutive indices from 0 to num_classes-1):")
    
    # Find the actual number of unique classes in the training data
    actual_unique_classes = len(np.unique(train_labels_continuous))
    print(f"Number of unique classes in training data: {actual_unique_classes}")
    
    # Create a proper keras-compatible class weights dictionary
    # Map from our indices to consecutive indices (0 to num_classes-1)
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
    
    # We also need to remap the continuous labels to match this mapping
    train_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in train_labels_continuous])
    val_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in val_labels_continuous])
    test_labels_continuous_keras = np.array([index_mapping.get(idx, 0) for idx in test_labels_continuous])
    
    # Create training and validation generators
    train_generator = CustomDataGenerator(
        file_train_selected_expanded, 
        train_labels_continuous_keras,  # Use remapped indices
        input_shape=(340, 100, 4),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = CustomDataGenerator(
        file_val_selected_expanded,
        val_labels_continuous_keras,  # Use remapped indices
        input_shape=(340, 100, 4),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_generator = CustomDataGenerator(
        file_test_selected_expanded,
        test_labels_continuous_keras,  # Use remapped indices
        input_shape=(340, 100, 4),
        batch_size=batch_size,
        shuffle=False
    )
    
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

    # Print sample batch shape for verification
    x_sample, y_sample = train_generator[0]
    print(f"Training batch shape: {x_sample.shape}, labels shape: {y_sample.shape}")
    print(f"Sample labels: {y_sample[:5]}")
    
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
    print(f"First 10 indices of validation data (should be ordered 0-9): {val_generator.indices[:10]}")

    # Create the model with the number of classes equal to the number of unique indices in Keras space
    actual_unique_classes = len(np.unique(train_labels_continuous_keras))
    csi_model = create_model(input_shape=(340, 100, 4), num_classes=actual_unique_classes)
    csi_model.summary()

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
    
    # Add additional metrics for better performance monitoring
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy')
    ]
    
    # Compile the model with our improved settings
    csi_model.compile(optimizer=optimiz, loss=loss, metrics=metrics)

    # Dataset statistics
    num_samples_train = len(file_train_selected_expanded)
    num_samples_val = len(file_val_selected_expanded)
    num_samples_test = len(file_test_selected_expanded)
    lab, count = np.unique(labels_train_selected_expanded, return_counts=True)
    lab_val, count_val = np.unique(labels_val_selected_expanded, return_counts=True)
    lab_test, count_test = np.unique(labels_test_selected_expanded, return_counts=True)
    
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
    
    # Train with class weights and all callbacks
    results = csi_model.fit(
        train_generator,
        epochs=30,  # Increased from 25 to give model more time to converge with our regularization
        validation_data=val_generator,
        callbacks=[callback_save, callback_stop, callback_reduce_lr, callback_tensorboard],
        class_weight=class_weights,
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

    # Use inference model for predictions
    print("Evaluating on training data...")
    # Use the original labels (not expanded) since our data generator now processes all antennas together
    train_labels_original = np.array(labels_train_selected)
    # We don't need to remap these since we're comparing original labels directly
    
    train_prediction_list = []
    for i in range(len(train_generator)):
        batch_x, _ = train_generator[i]
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
    
    # Predict on validation data
    print("Evaluating on validation data...")
    # Use the original labels (not expanded)
    val_labels_original = np.array(labels_val_selected)
    # We don't need to remap these since we're comparing original labels directly
    
    val_prediction_list = []
    for i in range(len(val_generator)):
        batch_x, _ = val_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        val_prediction_list.append(batch_pred)
    
    # Limit to the number of unique samples (original samples, not expanded)
    val_prediction_list = np.vstack(val_prediction_list)[:len(val_labels_original)]
    val_labels_pred_continuous = np.argmax(val_prediction_list, axis=1)
    val_labels_pred = convert_predictions_to_original_labels(val_labels_pred_continuous, index_to_label)
    
    print(f"Val prediction shape: {val_prediction_list.shape}")
    print(f"Val labels shape: {val_labels_original.shape}")
    
    # Same for validation
    val_labels_original_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 for label in val_labels_original])
    val_labels_pred_numeric = np.array([int(label) if isinstance(label, (int, np.integer)) else 0 for label in val_labels_pred])
    
    all_val_labels = np.unique(np.concatenate([val_labels_original_numeric, val_labels_pred_numeric]))
    
    print(f"Validation confusion matrix with {len(all_val_labels)} classes:")
    val_confusion_matrix = confusion_matrix(val_labels_original_numeric, val_labels_pred_numeric)
    print(val_confusion_matrix)
    print("\n")
    
    # Predict on test data
    print("Evaluating on test data...")
    # Use the original labels (not expanded)
    test_labels_original = np.array(labels_test_selected)
    # We don't need to remap these since we're comparing original labels directly
    
    test_prediction_list = []
    for i in range(len(test_generator)):
        batch_x, _ = test_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        test_prediction_list.append(batch_pred)
    
    # Limit to the number of unique samples (original samples, not expanded)
    test_prediction_list = np.vstack(test_prediction_list)[:len(test_labels_original)]
    test_labels_pred_continuous = np.argmax(test_prediction_list, axis=1)
    test_labels_pred = convert_predictions_to_original_labels(test_labels_pred_continuous, index_to_label)
    
    print(f"Test prediction shape: {test_prediction_list.shape}")
    print(f"Test labels shape: {test_labels_original.shape}")

    # Calculate metrics using original (non-expanded) labels
    # Ensure both arrays are of the same type (numeric)
    test_labels_original_numeric = np.array([int(l) if isinstance(l, (int, np.integer)) else 0 for l in test_labels_original])
    test_labels_pred_numeric = np.array([int(l) if isinstance(l, (int, np.integer)) else 0 for l in test_labels_pred])
    
    # Now calculate metrics with consistent types
    conf_matrix = confusion_matrix(test_labels_original_numeric, test_labels_pred_numeric)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels_original_numeric,
        test_labels_pred_numeric,
        labels=np.unique(np.concatenate([test_labels_original_numeric, test_labels_pred_numeric])),
        zero_division=0
    )
    accuracy = accuracy_score(test_labels_original_numeric, test_labels_pred_numeric)

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

    
