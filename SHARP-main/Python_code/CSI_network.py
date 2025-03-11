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
from dataset_utility import create_dataset_single, expand_antennas
from network_utility import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import glob
import gc
import shutil
import hashlib
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # For reproducibility
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better GPU mem management
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# Now import TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def compute_class_weights(labels, num_classes):
    """Compute class weights for imbalanced dataset."""
    total_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = {}
    for idx in range(num_classes):
        if idx in unique_labels:
            weights[idx] = total_samples / (num_classes * counts[np.where(unique_labels == idx)[0][0]])
        else:
            weights[idx] = 1.0
    return weights

def create_model(input_shape=(340, 100, 1), num_classes=6):
    """Create the CSI network model."""
    input_network = tf.keras.layers.Input(shape=input_shape)
    
    # First branch - 3x3 convolutions
    conv3_1 = tf.keras.layers.Conv2D(3, (3, 3), padding='same', name='1stconv3_1_res_a')(input_network)
    conv3_1 = tf.keras.layers.Activation('relu', name='activation_1')(conv3_1)
    conv3_2 = tf.keras.layers.Conv2D(6, (3, 3), padding='same', name='1stconv3_2_res_a')(conv3_1)
    conv3_2 = tf.keras.layers.Activation('relu', name='activation_2')(conv3_2)
    conv3_3 = tf.keras.layers.Conv2D(9, (3, 3), strides=(2, 2), padding='same', name='1stconv3_3_res_a')(conv3_2)
    conv3_3 = tf.keras.layers.Activation('relu', name='activation_3')(conv3_3)
    
    # Second branch - 2x2 convolutions
    conv2_1 = tf.keras.layers.Conv2D(5, (2, 2), strides=(2, 2), padding='same', name='1stconv2_1_res_a')(input_network)
    conv2_1 = tf.keras.layers.Activation('relu', name='activation')(conv2_1)
    
    # Third branch - max pooling
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d')(input_network)
    
    # Concatenate all branches
    concat = tf.keras.layers.Concatenate(name='concatenate')([pool1, conv2_1, conv3_3])
    
    # Additional convolution
    conv4 = tf.keras.layers.Conv2D(3, (1, 1), name='conv4')(concat)
    conv4 = tf.keras.layers.Activation('relu', name='activation_4')(conv4)
    
    # Flatten and dense layers
    flat = tf.keras.layers.Flatten(name='flatten')(conv4)
    drop = tf.keras.layers.Dropout(0.5, name='dropout')(flat)
    dense2 = tf.keras.layers.Dense(num_classes, name='dense2')(drop)
    
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
    args = parser.parse_args()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    
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

    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas)
    
    print("Sample labels:", labels_train_selected_expanded[:5])
    print("Label shapes:", np.array(labels_train_selected_expanded).shape)
    
    # Also expand validation and test data
    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)
        
    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                         labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                           labels_considered]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)
    
    # Create a custom data generator for training instead of using TensorFlow's dataset API
    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, file_names, labels, stream_indices, input_shape=(340, 100, 1), batch_size=16):
            self.file_names = file_names
            self.labels = labels
            self.stream_indices = stream_indices
            self.input_shape = input_shape
            self.batch_size = batch_size
            self.num_samples = len(file_names)

        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size

        def __getitem__(self, idx):
            batch_x = np.zeros((self.batch_size,) + self.input_shape)
            batch_y = np.zeros(self.batch_size)
            
            for i in range(self.batch_size):
                sample_idx = idx * self.batch_size + i
                if sample_idx >= self.num_samples:
                    # Pad with zeros if we're at the end
                    continue
                    
                try:
                    # Load the data file
                    file_path = self.file_names[sample_idx]
                    stream_idx = self.stream_indices[sample_idx]
                    
                    # Load and preprocess the data
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        
                    # Extract the specific antenna stream and reshape
                    # data shape is (antennas, features, time)
                    data = data[stream_idx]  # Get specific antenna data
                    data = np.transpose(data, (1, 0))  # Reshape to (time, features)
                    data = np.expand_dims(data, axis=-1)  # Add channel dimension
                    
                    # Store in batch
                    batch_x[i] = data
                    batch_y[i] = self.labels[sample_idx]
                    
                except Exception as e:
                    print(f"Warning: Error loading file {file_path}: {str(e)}")
                    # Keep zeros for this sample
                    continue
            
            return batch_x, batch_y

    # Create label mapping
    unique_labels = np.unique(np.concatenate([labels_train_selected_expanded, 
                                            labels_val_selected_expanded,
                                            labels_test_selected_expanded]))
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_to_index, f)
    # Convert labels to continuous indices
    train_labels_continuous = np.array([label_to_index[label] for label in labels_train_selected_expanded])
    val_labels_continuous = np.array([label_to_index[label] for label in labels_val_selected_expanded])
    test_labels_continuous = np.array([label_to_index[label] for label in labels_test_selected_expanded])

    # Calculate class weights based on continuous indices
    num_classes = len(unique_labels)
    class_weights = compute_class_weights(train_labels_continuous, num_classes)

    print("\nLabel to index mapping:")
    for label, idx in label_to_index.items():
        print(f"  Original label {label} -> Index {idx}")

    print("\nClass weights:")
    for idx in range(num_classes):
        print(f"  Class {idx} (original label {index_to_label[idx]}): {class_weights[idx]:.4f}")

    # Create data generators with continuous indices
    train_generator = CustomDataGenerator(
        file_train_selected_expanded, 
        train_labels_continuous,
        stream_ant_train,
        input_shape=(340, 100, 1),
        batch_size=batch_size
    )

    val_generator = CustomDataGenerator(
        file_val_selected_expanded,
        val_labels_continuous,
        stream_ant_val,
        input_shape=(340, 100, 1),
        batch_size=batch_size
    )

    test_generator = CustomDataGenerator(
        file_test_selected_expanded,
        test_labels_continuous,
        stream_ant_test,
        input_shape=(340, 100, 1),
        batch_size=batch_size
    )

    # Print sample batch shape for verification
    x_sample, y_sample = train_generator[0]
    print(f"Training batch shape: {x_sample.shape}, labels shape: {y_sample.shape}")
    print(f"Sample labels: {y_sample[:5]}")

    # Create the model with the correct number of output classes
    csi_model = create_model(input_shape=(340, 100, 1), num_classes=num_classes)
    csi_model.summary()

    # Define optimizer and loss function
    optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    csi_model.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

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
        print(f"\nSample of expanded files (first 5):")
        for i in range(min(5, len(file_train_selected_expanded))):
            print(f"  {i}: File={file_train_selected_expanded[i]}, Label={labels_train_selected_expanded[i]}, Stream={stream_ant_train[i]}")
    
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
    
    # Define callbacks
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    name_model = name_base + '_' + str(csi_act) + '_network.keras'
    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                       monitor='val_sparse_categorical_accuracy')

    # Check that the generators are working
    print("Checking data generators...")
    try:
        test_batch = train_generator[0]
        print(f"Training batch shape: {test_batch[0].shape}, labels shape: {test_batch[1].shape}")
    except Exception as e:
        print(f"Error testing data generator: {e}")
        sys.exit(1)
    
    # Train with class weights
    results = csi_model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[callback_save, callback_stop],
        class_weight=class_weights
    )

    # For inference, create a model that includes the softmax
    inference_model = tf.keras.Sequential([
        csi_model,
        tf.keras.layers.Softmax()
    ])

    # Save both models
    csi_model.save(name_model)
    inference_model.save(name_model.replace('.keras', '_inference.keras'))

    # Function to convert continuous indices back to original labels
    def convert_predictions_to_original_labels(predictions):
        return np.array([index_to_label[idx] for idx in predictions])

    # Use inference model for predictions
    print("Evaluating on training data...")
    train_labels_true = np.array(labels_train_selected_expanded)
    train_prediction_list = []
    for i in range(len(train_generator)):
        batch_x, _ = train_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        train_prediction_list.append(batch_pred)

    train_prediction_list = np.vstack(train_prediction_list)[:train_labels_true.shape[0]]
    train_labels_pred_continuous = np.argmax(train_prediction_list, axis=1)
    train_labels_pred = convert_predictions_to_original_labels(train_labels_pred_continuous)
    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

    # Predict on validation data
    print("Evaluating on validation data...")
    val_labels_true = np.array(labels_val_selected_expanded)
    
    val_prediction_list = []
    for i in range(len(val_generator)):
        batch_x, _ = val_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        val_prediction_list.append(batch_pred)
    
    val_prediction_list = np.vstack(val_prediction_list)[:val_labels_true.shape[0]]
    val_labels_pred_continuous = np.argmax(val_prediction_list, axis=1)
    val_labels_pred = convert_predictions_to_original_labels(val_labels_pred_continuous)
    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

    # Predict on test data
    print("Evaluating on test data...")
    test_labels_true = np.array(labels_test_selected_expanded)
    
    test_prediction_list = []
    for i in range(len(test_generator)):
        batch_x, _ = test_generator[i]
        batch_pred = inference_model.predict(batch_x, verbose=0)
        test_prediction_list.append(batch_pred)
    
    test_prediction_list = np.vstack(test_prediction_list)[:test_labels_true.shape[0]]
    test_labels_pred_continuous = np.argmax(test_prediction_list, axis=1)
    test_labels_pred = convert_predictions_to_original_labels(test_labels_pred_continuous)

    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                   test_labels_pred,
                                                                   labels=labels_considered, zero_division=0)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)

    # merge antennas test
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    
    # Process predictions by antenna groups
    for i_lab in range(len(labels_test_selected)):
        # Get predictions for all antennas for this sample
        pred_antennas = test_prediction_list[i_lab*num_antennas:(i_lab+1)*num_antennas, :]
        # Sum predictions across antennas
        sum_pred = np.sum(pred_antennas, axis=0)
        lab_merge_max = np.argmax(sum_pred)

        # Get predicted classes for each antenna
        pred_max_antennas = test_labels_pred[i_lab*num_antennas:(i_lab+1)*num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        lab_max_merge = -1
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered,  zero_division=0)
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

    # impact of the number of antennas
    one_antenna = [[0], [1], [2], [3]]
    two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    four_antennas = [[0, 1, 2, 3]]
    seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    average_accuracy_change_num_ant = np.zeros((num_antennas,))
    average_fscore_change_num_ant = np.zeros((num_antennas,))
    labels_true_merge = np.array(labels_test_selected)
    for ant_n in range(num_antennas):
        seq_ant = seq_ant_list[ant_n]
        num_seq = len(seq_ant)
        for seq_n in range(num_seq):
            pred_max_merge = np.zeros((len(labels_test_selected),))
            ants_selected = seq_ant[seq_n]
            for i_lab in range(len(labels_test_selected)):
                pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
                pred_antennas = pred_antennas[ants_selected, :]

                lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

                pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
                pred_max_antennas = pred_max_antennas[ants_selected]
                lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
                lab_max_merge = -1
                if lab_unique.shape[0] > 1:
                    count_argsort = np.flip(np.argsort(count))
                    count_sort = count[count_argsort]
                    lab_unique_sort = lab_unique[count_argsort]
                    if count_sort[0] == count_sort[1] or lab_unique.shape[0] > ant_n - 1:  # ex aequo between two labels
                        lab_max_merge = lab_merge_max
                    else:
                        lab_max_merge = lab_unique_sort[0]
                else:
                    lab_max_merge = lab_unique[0]
                pred_max_merge[i_lab] = lab_max_merge

            _, _, fscore_max_merge, _ = precision_recall_fscore_support(labels_true_merge, pred_max_merge,
                                                                        labels=[0, 1, 2, 3, 4], zero_division=0)
            accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

            average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
            average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

        average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
        average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq

    metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
                           'average_fscore_change_num_ant': average_fscore_change_num_ant}
    unique_id = hashlib.md5(f"{csi_act}_{subdirs_training}".encode()).hexdigest()[:8]
    name_file = f'./outputs/change_number_antennas_test_{unique_id}_b{bandwidth}_sb{sub_band}.txt'
    # name_file = './outputs/change_number_antennas_test_' + str(csi_act) + '_' + subdirs_training + '_band_' + \
    #       
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)
    tf.keras.backend.clear_session()
    gc.collect()
    
