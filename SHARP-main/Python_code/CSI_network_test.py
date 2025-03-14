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
from dataset_utility import create_dataset_single, expand_antennas, create_dataset_multi_channel, load_data_multi_channel
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tensorflow as tf
import hashlib
import matplotlib.pyplot as plt
import sys
import gc

def find_label_mapping(name_base, model_dir=None):
    """
    Find the label mapping file by checking multiple possible locations.
    
    Args:
        name_base: Base name for the model/files
        model_dir: Optional directory where model files are stored
        
    Returns:
        dict: Label mapping dictionary
        str: Path to the label mapping file that was found
    """
    possible_paths = [
        f'single_ant.keras_label_mapping.pkl',  # Specifically mentioned in the task
        f'{name_base}_label_mapping.pkl',  # Original path
        'label_mapping.pkl',  # Generic name
        os.path.join(os.path.dirname(name_base), 'label_mapping.pkl'),  # Same directory as name_base
    ]
    
    if model_dir:
        possible_paths.extend([
            os.path.join(model_dir, 'single_ant.keras_label_mapping.pkl'),
            os.path.join(model_dir, f'{os.path.basename(name_base)}_label_mapping.pkl'),
            os.path.join(model_dir, 'label_mapping.pkl')
        ])
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    mapping = pickle.load(f)
                print(f"Successfully loaded label mapping from: {path}")
                
                # Handle both original and enhanced mapping formats
                if isinstance(mapping, dict) and any(key in mapping for key in ['label_to_index', 'index_to_label', 'activities']):
                    # Enhanced format with nested dictionaries
                    print("Detected enhanced mapping format with multiple components")
                    # Extract just the label_to_index mapping that this script expects
                    label_mapping = mapping['label_to_index']
                    
                    # Print additional information if available
                    print("\nRaw label to index mapping:")
                    for label, idx in label_mapping.items():
                        print(f"  Raw label {label} → Index {idx}")
                        
                    if 'index_to_label' in mapping:
                        print("\nIndex to label mapping:")
                        for idx, label in mapping['index_to_label'].items():
                            print(f"  Index {idx} → Raw label {label}")
                            
                    if 'activities' in mapping:
                        print(f"\nActivity names: {mapping['activities']}")
                else:
                    # Original format (simple dict)
                    print("Detected original mapping format (simple dictionary)")
                    label_mapping = mapping
                    
                    # Print the mapping for verification
                    print("\nRaw label to index mapping:")
                    for label, idx in label_mapping.items():
                        print(f"  Raw label {label} → Index {idx}")
                
                # Validation: check for sane number of classes
                if len(label_mapping) < 2:
                    print(f"WARNING: Label mapping has only {len(label_mapping)} classes, which is suspiciously low.")
                    print("The model might be misclassifying because it was trained with insufficient classes.")
                elif len(label_mapping) > 15:
                    print(f"WARNING: Label mapping has {len(label_mapping)} classes, which is suspiciously high.")
                    print("This might indicate extra, unused classes in the mapping that could cause confusion.")
                else:
                    print(f"Label mapping has {len(label_mapping)} classes, which seems reasonable.")
                
                # Validation: check for class index bias
                class_indices = list(label_mapping.values())
                if 5 in class_indices and class_indices.count(5) == 1:
                    # Get which raw label maps to index 5
                    label_5 = [k for k, v in label_mapping.items() if v == 5][0]
                    print(f"\nNOTE: Raw label {label_5} maps to index 5, which was previously identified with class bias.")
                    print("Monitor predictions carefully to ensure this class isn't being over-predicted.")
                
                return label_mapping, path
            except Exception as e:
                print(f"Error reading mapping from {path}: {e}")
                continue
                
    # If we get here, no valid mapping was found
    print(f"ERROR: Could not find valid label mapping file in any of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("Will use a fallback approach that may not work correctly.")
    
    # Create a simple fallback mapping
    default_mapping = {i: i for i in range(6)}  # Six classes as default
    print(f"Created default mapping: {default_mapping}")
    return default_mapping, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for testing')
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
    parser.add_argument('--model_dir', help='Directory containing the model and label mapping files', 
                       default=None, required=False)
    args = parser.parse_args()

    print("\n" + "="*80)
    print("IMPORTANT: This script requires a model trained with 4-channel (multi-antenna) input")
    print("The model input shape should be (340, 100, 4)")
    print("If you're using a model trained with the old single-channel architecture,")
    print("please retrain it using the updated CSI_network.py script.")
    print("="*80 + "\n")
    
    # Find and load label mapping
    label_to_index, mapping_path = find_label_mapping(args.name_base, args.model_dir)
    print(f"Using label mapping with {len(label_to_index)} classes: {label_to_index}")
    
    # Create reverse mapping for validation
    try:
        # Check if the values in label_to_index are simple hashable types (like int)
        is_simple_mapping = all(isinstance(v, (int, str, float, bool)) for v in label_to_index.values())
        
        if is_simple_mapping:
            # Standard case - create reverse mapping directly
            index_to_label = {v: k for k, v in label_to_index.items()}
        else:
            # Complex case - handle nested dictionaries or unhashable types
            print("Warning: Complex label mapping detected - creating simplified version")
            # Create a simple numeric mapping instead
            index_to_label = {}
            for i, (k, v) in enumerate(label_to_index.items()):
                index_to_label[i] = k
            # Update label_to_index to match this simpler mapping
            label_to_index = {k: i for i, (k, v) in enumerate(label_to_index.items())}
            print(f"Simplified label_to_index: {label_to_index}")
        
        print(f"Index to label mapping: {index_to_label}")
    except Exception as e:
        print(f"Error creating reverse mapping: {e}")
        print("Creating fallback mapping...")
        # Create simple sequential mappings as fallback
        unique_labels = list(label_to_index.keys())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        print(f"Fallback label_to_index: {label_to_index}")
        print(f"Fallback index_to_label: {index_to_label}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

    # Create a lookup table for real activity names
    activity_name_lookup = {}
    for i, activity in enumerate(activities):
        # Map the activity index to the actual activity name
        activity_name_lookup[i] = activity
    
    print("\nActivity labels from command line:")
    for i, activity in enumerate(activities):
        print(f"  Activity {i}: '{activity}'")

    suffix = '.txt'

    name_base = args.name_base
    # Clean up any existing cache files
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_complete.index'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_complete.index')
    
    subdirs_complete = args.subdirs  # string
    labels_complete = []
    all_files_complete = []
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

    # Load and validate labels
    for sdir in subdirs_complete.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_complete = args.dir + sdir + '/complete_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_complete_antennas_' + str(csi_act) + suffix
        
        if not os.path.exists(name_labels):
            print(f"Error: File not found - {name_labels}")
            sys.exit(1)
            
        name_f = args.dir + sdir + '/files_complete_antennas_' + str(csi_act) + suffix
        
        if not os.path.exists(name_f):
            print(f"Error: File not found - {name_f}")
            sys.exit(1)
            
        try:
            with open(name_labels, "rb") as fp:
                current_labels = pickle.load(fp)
                
                # Ensure all labels are simple types (int, str) for compatibility
                current_labels = [int(label) if isinstance(label, (int, float, np.integer)) 
                                else str(label) for label in current_labels]
                
                # Validate labels before adding them
                invalid_labels = set([l for l in current_labels if l not in label_to_index])
                if invalid_labels:
                    print(f"\nWarning: Found invalid labels in {name_labels}:")
                    print(f"Invalid labels: {invalid_labels}")
                    print(f"Valid labels are: {set(label_to_index.keys())}")
                    print("\nAdding these labels to the mapping with new indices...")
                    
                    # Add missing labels to the mapping
                    next_idx = max(v for v in label_to_index.values() if isinstance(v, (int, np.integer)))
                    for label in invalid_labels:
                        next_idx += 1
                        label_to_index[label] = next_idx
                        index_to_label[next_idx] = label
                        print(f"  Added missing label {label} → Index {next_idx}")
                    
                labels_complete.extend(current_labels)
            with open(name_f, "rb") as fp:
                all_files_complete.extend(pickle.load(fp))
        except Exception as e:
            print(f"Error loading data files: {e}")
            sys.exit(1)

    # Filter data
    file_complete_selected = [all_files_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                              labels_considered]
    
    # Map raw labels to model indices
    raw_labels_complete_selected = [labels_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                                labels_considered]
    
    # Create mapped labels
    labels_complete_selected = []
    invalid_labels = []
    for label in raw_labels_complete_selected:
        if label in label_to_index:
            labels_complete_selected.append(label_to_index[label])
        else:
            invalid_labels.append(label)
    
    if invalid_labels:
        print(f"Warning: Found {len(invalid_labels)} labels not in the mapping: {set(invalid_labels)}")
        print("These will be excluded from evaluation.")

    # Check if we have enough data
    if len(file_complete_selected) == 0:
        print("Error: No data files selected after filtering. Check your activities parameter.")
        sys.exit(1)

    # Verify file existence and content
    missing_files = [f for f in file_complete_selected if not os.path.exists(f)]
    if missing_files:
        print(f"Error: {len(missing_files)} data files are missing. First few: {missing_files[:5]}")
        sys.exit(1)
    else:
        print(f"All {len(file_complete_selected)} data files exist.")
        
        # Directly inspect a few files to verify content
        print(f"\nDirectly inspecting sample data files:")
        for i in range(min(2, len(file_complete_selected))):
            file_path = file_complete_selected[i]
            print(f"\nInspecting file {i+1}: {file_path}")
            file_stats = os.stat(file_path)
            print(f"  File size: {file_stats.st_size} bytes")
            
            try:
                with open(file_path, "rb") as fp:
                    data = pickle.load(fp)
                print(f"  Data type: {type(data)}")
                
                if isinstance(data, np.ndarray):
                    print(f"  Data shape: {data.shape}")
                    print(f"  Data sample (min/max/mean): {np.min(data):.6f}/{np.max(data):.6f}/{np.mean(data):.6f}")
                    print(f"  Contains zeros only: {np.all(data == 0)}")
                    print(f"  Contains NaNs: {np.any(np.isnan(data))}")
                else:
                    print(f"  Data is not a numpy array")
            except Exception as e:
                print(f"  Error inspecting file: {e}")

    # Check labels
    print(f"\nLabel distribution: {np.unique(labels_complete_selected, return_counts=True)}")
        
    # Create a custom testing function that bypasses TensorFlow dataset API issues
    print("\nCreating custom test process using direct data loading...")
    
    try:
        # Print a detailed explanation of the shape transformation we're performing
        print("\n" + "="*80)
        print("SHAPE TRANSFORMATION EXPLANATION:")
        print("="*80)
        print(f"Input data shape from files: (4, 100, 340) - 4 antennas, 100 time steps, 340 features")
        print(f"Required model input shape: (?, 340, 100, 4) - batch, height=340, width=100, channels=4")
        print("\nCorrect transformation process (all antennas as channels):")
        print("1. Transpose data to reorganize dimensions: (4, 100, 340) → (340, 100, 4)")
        print("2. Add batch dimension: (1, 340, 100, 4)")
        print("="*80 + "\n")
        
        # Load the model
        name_model = name_base + '_' + str(csi_act) + '_network.keras'
        if not os.path.exists(name_model):
            print(f"Error: Model file not found - {name_model}")
            sys.exit(1)
            
        csi_model = load_model(name_model)
        print(f"Model loaded: {name_model}")
        
        # Add debugging for model architecture and input shape
        print("\nModel Summary:")
        csi_model.summary()
        print(f"\nModel expected input shape: {csi_model.input_shape}")
        print(f"Model output shape: {csi_model.output_shape}")
        
        # Create a mapping from indices to activity names if possible
        activity_names = {}
        for label_val, idx in label_to_index.items():
            if isinstance(label_val, str) and len(label_val) == 1:
                # Single-letter activity codes like W, S, J
                activity_names[idx] = label_val
            else:
                # Get real activity name from the lookup table if possible
                # FIXED: Previously, we were using the idx value as an index into activity_name_lookup
                # but this is incorrect. Instead, we need to use the raw label value directly
                # to get the correct name from the activities list

                # Find the correct position in the activities array
                activity_index = None
                for i, act_name in enumerate(activities):
                    # Check if this is the activity that corresponds to raw label
                    if i == label_val:  # This maps the raw numeric label to the activity name
                        activity_index = i
                        break

                if activity_index is not None and activity_index < len(activity_name_lookup):
                    activity_names[idx] = activity_name_lookup[activity_index]
                else:
                    # Fallback: Use the original label as the activity name
                    activity_names[idx] = f"{index_to_label.get(idx, 'Unknown')}"
                
        print("\nActivity names for each class index:")
        for idx, name in activity_names.items():
            print(f"  Class {idx}: '{name}' (original label: {index_to_label[idx]})")
        
        # Process files directly
        all_predictions = []
        true_labels = []
        
        for i, file_path in enumerate(file_complete_selected):
            try:
                original_label = raw_labels_complete_selected[i]  # Use raw label
                
                # Ensure label is a simple type
                if isinstance(original_label, (np.integer, float)):
                    original_label = int(original_label)
                elif not isinstance(original_label, (int, str)):
                    original_label = str(original_label)
                
                # Check if the label exists in the mapping
                if original_label not in label_to_index:
                    print(f"\nWarning: Label {original_label} not found in label mapping.")
                    print("Adding it to the mapping with a new index to prevent errors.")
                    # Add it to the mapping with a new index
                    next_idx = max(v for v in label_to_index.values() if isinstance(v, (int, np.integer))) + 1
                    label_to_index[original_label] = next_idx
                    index_to_label[next_idx] = original_label
                    print(f"Mapped missing label {original_label} → Index {next_idx}")
                
                true_label_index = label_to_index[original_label]
                true_labels.append(true_label_index)
                print(f"\nProcessing file {i+1}/{len(file_complete_selected)}: {file_path}")
                print(f"  True label: {original_label} → Index {true_label_index} (Activity: '{activity_names.get(true_label_index, 'Unknown')}')")
                
                # Load data directly
                with open(file_path, "rb") as fp:
                    matrix_csi = pickle.load(fp)
                
                print(f"Loaded data shape: {matrix_csi.shape}")
                print(f"Data statistics - min: {np.min(matrix_csi)}, max: {np.max(matrix_csi)}, mean: {np.mean(matrix_csi)}")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
            
            # CRITICAL FIX: Process all 4 antennas together as channels
            print("\nProcessing all antennas together as channels:")
            
            # Original shape: (4, 100, 340) - 4 antennas, 100 time steps, 340 features
            print(f"Original data shape: {matrix_csi.shape}")
            
            # STEP 1: Transpose to get (340, 100, 4) - features as height, time as width, antennas as channels
            data = np.transpose(matrix_csi, (2, 1, 0))
            print(f"After transpose shape: {data.shape}")
            
            # STEP 2: Add batch dimension for model
            data = np.expand_dims(data, axis=0)  # Shape (1, 340, 100, 4)
            print(f"Final input shape with batch dimension: {data.shape}")
            print(f"Expected model input shape: {csi_model.input_shape}")
            
            # Verify data alignment with model
            if data.shape[1:] != csi_model.input_shape[1:]:
                print(f"WARNING: Input shape {data.shape[1:]} does not match model's expected shape {csi_model.input_shape[1:]}")
                print(f"This test script requires a model trained with input shape (340, 100, 4)")
                print(f"Please retrain the model with the multi-channel (4 antennas) architecture")
                sys.exit(1)
            
            # Make a single prediction with all antennas as channels
            pred = csi_model.predict(data, verbose=0)
            
            # Print prediction info for debugging
            print(f"\nPrediction shape: {pred.shape}")
            print(f"Prediction values: {pred[0]}")
            print(f"Predicted class: {np.argmax(pred[0])}")
            
            # Get predicted class
            pred_class = np.argmax(pred[0])
            
            activity_name = activity_names.get(pred_class, "Unknown")
            original_label_name = index_to_label.get(pred_class, "Unknown")
            print(f"Prediction: class {pred_class} (Activity: '{activity_name}', original label: {original_label_name})")
            print(f"True label: class {true_label_index} (Activity: '{activity_names.get(true_label_index, 'Unknown')}', original label: {original_label})")
            all_predictions.append(pred_class)
        
        # Calculate accuracy
        all_predictions = np.array(all_predictions)
        true_labels = np.array(true_labels)
        accuracy = np.mean(all_predictions == true_labels)
        
        # Check if all predictions are the same (which would indicate a problem)
        unique_predictions, counts = np.unique(all_predictions, return_counts=True)
        print(f"\nUnique predictions: {unique_predictions}")
        print(f"Prediction counts: {counts}")
        
        if len(unique_predictions) == 1:
            print("\n⚠️ WARNING: All predictions are the same class! This indicates the data shape transformation")
            print("might not be working correctly. Check the shape transformations above.")
        else:
            print("\n✓ SUCCESS: Predictions have multiple classes, suggesting the shape transformation is working.")
        
        # Calculate class distribution percentages
        total_preds = len(all_predictions)
        print("\nDetailed prediction distribution:")
        for i, (pred_class, count) in enumerate(zip(unique_predictions, counts)):
            percentage = (count / total_preds) * 100
            activity_name = activity_names.get(pred_class, "Unknown")
            original_label = index_to_label.get(pred_class, "Unknown")
            print(f"  Class {pred_class} (Activity: '{activity_name}', original label: {original_label}): {count} predictions ({percentage:.1f}%)")
        
        # Display true label distribution
        unique_true_labels, true_counts = np.unique(true_labels, return_counts=True)
        print("\nTrue label distribution:")
        for i, (true_class, count) in enumerate(zip(unique_true_labels, true_counts)):
            percentage = (count / len(true_labels)) * 100
            activity_name = activity_names.get(true_class, "Unknown")
            original_label = index_to_label.get(true_class, "Unknown")
            print(f"  Class {true_class} (Activity: '{activity_name}', original label: {original_label}): {count} instances ({percentage:.1f}%)")
        
        print(f"\nTest results with direct data loading:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, all_predictions)
        class_labels = [f"{i}({activity_names.get(i, '?')})" for i in range(max(np.max(true_labels), np.max(all_predictions)) + 1)]
        print(f"\nConfusion matrix (row: true, column: predicted):")
        print(f"Labels: {class_labels}")
        print(cm)
        
        # Calculate per-class accuracy
        print("\nPer-class accuracy:")
        per_class_accuracy = {}
        for i, label in enumerate(labels_considered):
            if label in unique_true_labels:
                idx = np.where(unique_true_labels == label)[0][0]
                class_accuracy = cm[idx, idx] / np.sum(cm[idx]) if np.sum(cm[idx]) > 0 else 0
                activity_name = activity_names.get(label, "Unknown")
                original_label = index_to_label.get(label, "Unknown")
                print(f"  Class {label} (Activity: '{activity_name}', original label: {original_label}): {class_accuracy * 100:.2f}%")
                per_class_accuracy[label] = class_accuracy
        
        # Calculate precision, recall, and f-score
        precision, recall, fscore, _ = precision_recall_fscore_support(
            true_labels, 
            all_predictions,
            labels=np.unique(np.concatenate([true_labels, all_predictions])),
            zero_division=0
        )
        
        print("\nPrecision, Recall, and F-Score by class:")
        for i, label in enumerate(np.unique(np.concatenate([true_labels, all_predictions]))):
            if i < len(precision):
                activity_name = activity_names.get(label, "Unknown")
                original_label = index_to_label.get(label, "Unknown")
                print(f"  Class {label} (Activity: '{activity_name}', original label: {original_label}):")
                print(f"    Precision: {precision[i]:.4f}")
                print(f"    Recall: {recall[i]:.4f}")
                print(f"    F-Score: {fscore[i]:.4f}")
        
        # Save metrics dictionary with enhanced information
        metrics_matrix_dict = {
            'conf_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'per_class_accuracy': per_class_accuracy,
            'activity_names': activity_names,
            'label_to_index': label_to_index,
            'index_to_label': index_to_label
        }
        
        # Exit successfully
        print("\nDirect data loading test completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"Error in custom test process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # If we get here, fallback to the standard dataset approach
    print("\nFalling back to the standard dataset approach with multi-channel data...")
    
    try:
        # Create multi-channel dataset directly (no need to expand antennas)
        print("Creating multi-channel dataset from all files...")
        
        # First map the original labels to indices for consistent evaluation
        labels_complete_selected_indices = [label_to_index[label] for label in labels_complete_selected 
                                          if label in label_to_index]
        
        # Filter files to only include those with valid labels
        valid_indices = [i for i, label in enumerate(labels_complete_selected) if label in label_to_index]
        file_complete_selected_filtered = [file_complete_selected[i] for i in valid_indices]
        
        if len(file_complete_selected_filtered) != len(file_complete_selected):
            print(f"Warning: Filtered out {len(file_complete_selected) - len(file_complete_selected_filtered)} files with invalid labels")
        
        dataset_csi_complete = create_dataset_multi_channel(
            file_complete_selected_filtered, 
            labels_complete_selected_indices,  # Use the mapped indices
            (feature_length, sample_length, num_antennas),  # Expected shape: (340, 100, 4)
            batch_size, 
            shuffle=False,
            cache_file=name_base + '_' + str(csi_act) + '_cache_complete_multi'
        )
    except Exception as e:
        print(f"Error creating multi-channel dataset: {e}")
        print("Trying original approach as last resort...")
        
        # Expand antennas for dataset creation (original approach as fallback)
        file_complete_selected_expanded, labels_complete_selected_expanded, stream_ant_complete = \
            expand_antennas(file_complete_selected, labels_complete_selected, num_antennas)
            
        print(f"Data files expanded to {len(file_complete_selected_expanded)} samples with {num_antennas} antennas each.")

        # Create dataset with original approach
        try:
            dataset_csi_complete = create_dataset_single(file_complete_selected_expanded, labels_complete_selected_expanded,
                                                        stream_ant_complete, input_network, batch_size, shuffle=False,
                                                        cache_file=name_base + '_' + str(csi_act) + '_cache_complete')
        except Exception as e:
            print(f"Error creating dataset: {e}")
            sys.exit(1)
            
        # After creating the dataset, also update the label information for the original approach
        num_samples_complete = len(file_complete_selected_expanded)
        lab_complete, count_complete = np.unique(labels_complete_selected_expanded, return_counts=True)
        complete_labels_true = np.array(labels_complete_selected_expanded)

        # Create a mapping from indices to activity names
        activity_names = {}
        for label_val, idx in label_to_index.items():
            if isinstance(label_val, str) and len(label_val) == 1:
                # Single-letter activity codes like W, S, J
                activity_names[idx] = label_val
            else:
                # Get real activity name from the lookup table if possible
                # FIXED: Previously, we were using the idx value as an index into activity_name_lookup
                # but this is incorrect. Instead, we need to use the raw label value directly
                # to get the correct name from the activities list

                # Find the correct position in the activities array
                activity_index = None
                for i, act_name in enumerate(activities):
                    # Check if this is the activity that corresponds to raw label
                    if i == label_val:  # This maps the raw numeric label to the activity name
                        activity_index = i
                        break

                if activity_index is not None and activity_index < len(activity_name_lookup):
                    activity_names[idx] = activity_name_lookup[activity_index]
                else:
                    # Fallback: Use the original label as the activity name
                    activity_names[idx] = f"{index_to_label.get(idx, 'Unknown')}"
        
        print("\nActivity names for each class index:")
        for idx, name in activity_names.items():
            print(f"  Class {idx}: '{name}' (original label: {index_to_label[idx]})")
        
    # Load model
    name_model = name_base + '_' + str(csi_act) + '_network.keras'
    if not os.path.exists(name_model):
        print(f"Error: Model file not found - {name_model}")
        sys.exit(1)
        
    try:
        csi_model = load_model(name_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Prepare for prediction
    complete_steps_per_epoch = int(np.ceil(num_samples_complete / batch_size))

    # Display dataset information
    print(f"Training samples: {num_samples_complete}")
    print(f"Training labels and counts: {list(zip(lab_complete, count_complete))}")
    print(f"Test steps per epoch: {complete_steps_per_epoch}")
    print("Checking datasets...")
    
    # Validate dataset
    for batch in dataset_csi_complete.take(1):
        print(f"Training batch shape: {batch[0].shape}, labels shape: {batch[1].shape}")

    # Make predictions
    try:
        complete_prediction_list = csi_model.predict(dataset_csi_complete,
                                                    steps=complete_steps_per_epoch)[:complete_labels_true.shape[0]]
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    # Check prediction distribution
    print("\nPrediction distribution check:")
    print(f"Prediction shape: {complete_prediction_list.shape}")
    print(f"Sample predictions (first 3 samples):")
    for i in range(min(3, complete_prediction_list.shape[0])):
        print(f"  Sample {i}: {complete_prediction_list[i]}")
    
    # Check if predictions are all similar (which would cause all the same class predictions)
    pred_variance = np.var(complete_prediction_list, axis=1)
    print(f"Prediction variance stats: min={pred_variance.min():.6f}, max={pred_variance.max():.6f}, mean={pred_variance.mean():.6f}")
    print(f"Zero variance predictions: {np.sum(pred_variance < 1e-6)} out of {pred_variance.shape[0]}")
    
    complete_labels_pred = np.argmax(complete_prediction_list, axis=1)
    
    # Check prediction distribution
    unique_preds, counts = np.unique(complete_labels_pred, return_counts=True)
    print(f"Predicted class distribution: {list(zip(unique_preds, counts))}")
    
    # Display prediction distribution with activity names
    print("\nDetailed prediction distribution:")
    for i, (pred_class, count) in enumerate(zip(unique_preds, counts)):
        percentage = (count / len(complete_labels_pred)) * 100
        activity_name = activity_names.get(pred_class, "Unknown")
        original_label = index_to_label.get(pred_class, "Unknown")
        print(f"  Class {pred_class} (Activity: '{activity_name}', original label: {original_label}): {count} predictions ({percentage:.1f}%)")
    
    # Display true label distribution
    unique_true_labels, true_counts = np.unique(complete_labels_true, return_counts=True)
    print("\nTrue label distribution:")
    for i, (true_class, count) in enumerate(zip(unique_true_labels, true_counts)):
        percentage = (count / len(complete_labels_true)) * 100
        activity_name = activity_names.get(true_class, "Unknown")
        original_label = index_to_label.get(true_class, "Unknown")
        print(f"  Class {true_class} (Activity: '{activity_name}', original label: {original_label}): {count} instances ({percentage:.1f}%)")
    
    # Check if ground truth and predictions match
    correct = np.sum(complete_labels_pred == complete_labels_true)
    print(f"Correct predictions: {correct} out of {complete_labels_true.shape[0]} ({100.0 * correct / complete_labels_true.shape[0]:.2f}%)")

    # Calculate metrics with zero_division=0 to avoid warnings
    conf_matrix = confusion_matrix(complete_labels_true, complete_labels_pred, labels=labels_considered)
    precision, recall, fscore, _ = precision_recall_fscore_support(complete_labels_true,
                                                                  complete_labels_pred,
                                                                  labels=labels_considered,
                                                                  zero_division=0)
    accuracy = accuracy_score(complete_labels_true, complete_labels_pred)

    # Display confusion matrix with activity names
    class_labels = [f"{i}({activity_names.get(i, '?')})" for i in labels_considered]
    print(f"\nConfusion matrix (row: true, column: predicted):")
    print(f"Labels: {class_labels}")
    print(conf_matrix)
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    per_class_accuracy = {}
    for i, label in enumerate(labels_considered):
        if label in unique_true_labels:
            idx = np.where(unique_true_labels == label)[0][0]
            class_accuracy = conf_matrix[idx, idx] / np.sum(conf_matrix[idx]) if np.sum(conf_matrix[idx]) > 0 else 0
            activity_name = activity_names.get(label, "Unknown")
            original_label = index_to_label.get(label, "Unknown")
            print(f"  Class {label} (Activity: '{activity_name}', original label: {original_label}): {class_accuracy * 100:.2f}%")
            per_class_accuracy[label] = class_accuracy

    # Save metrics dictionary with enhanced information
    metrics_matrix_dict = {
        'conf_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'per_class_accuracy': per_class_accuracy,
        'activity_names': activity_names,
        'label_to_index': label_to_index,
        'index_to_label': index_to_label
    }

    # Save results
    os.makedirs('./outputs', exist_ok=True)
    unique_id = hashlib.md5(f"{csi_act}_{subdirs_complete}".encode()).hexdigest()[:8]
    name_file = f'./outputs/complete_different__{unique_id}_b{bandwidth}_sb{sub_band}.txt'
    
    try:
        with open(name_file, "wb") as fp:
            pickle.dump(metrics_matrix_dict, fp)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Print final accuracy and metrics summary
    print("\nFinal Results Summary:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average F-Score: {np.mean(fscore):.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Clean up any other variables that might not be needed for antenna analysis
    # to ensure they don't cause errors later
    if 'accuracy_max_merge' not in locals():
        accuracy_max_merge = accuracy
    if 'fscore_max_merge' not in locals():
        fscore_max_merge = fscore
    if 'conf_matrix_max_merge' not in locals():
        conf_matrix_max_merge = conf_matrix

    # impact of the number of antennas
    # This section was designed for the expanded antenna approach, 
    # but we now use a multi-channel approach
    print("\nAnalyzing impact of using different numbers of antennas...")
    
    # Define variables to make the code below work regardless of which path was taken
    if 'all_predictions' in locals() and 'complete_prediction_list' not in locals():
        # Direct loading path
        print("Using prediction data from direct loading approach")
        complete_prediction_list = np.array([])  # Not used in this case
        complete_labels_pred = all_predictions
        labels_true_merge = true_labels
    else:
        # Dataset path
        print("Using prediction data from dataset approach")
        labels_true_merge = np.array(labels_complete_selected)

    # Start the antenna analysis
    one_antenna = [[0], [1], [2], [3]]
    two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    four_antennas = [[0, 1, 2, 3]]
    seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    average_accuracy_change_num_ant = np.zeros((num_antennas, ))
    average_fscore_change_num_ant = np.zeros((num_antennas, ))
    
    # Skip this section if using direct loading approach since we already combine antennas
    if 'all_predictions' in locals() and 'complete_prediction_list' not in locals():
        print("Skipping detailed antenna analysis since we're already using the multi-channel approach")
        
        # Create simpler output for consistency
        average_accuracy_change_num_ant = np.array([0.0, 0.0, 0.0, accuracy])
        average_fscore_change_num_ant = np.array([0.0, 0.0, 0.0, np.mean(fscore)])
        
        # Skip the rest of the detailed analysis
    else:
        # Original antenna analysis code
        for ant_n in range(num_antennas):
            seq_ant = seq_ant_list[ant_n]
            num_seq = len(seq_ant)
            for seq_n in range(num_seq):
                pred_max_merge = np.zeros((len(labels_complete_selected), ))
                ants_selected = seq_ant[seq_n]
                for i_lab in range(len(labels_complete_selected)):
                    pred_antennas = complete_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
                    pred_antennas = pred_antennas[ants_selected, :]

                    lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

                    pred_max_antennas = complete_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
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
                                                                           labels=labels_considered, zero_division=0)
                accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

                average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
                average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

            average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
            average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq
    
    # Print antenna analysis results
    print("\nAccuracy by number of antennas:")
    for i in range(num_antennas):
        print(f"  {i+1} antenna(s): {average_accuracy_change_num_ant[i] * 100:.2f}%")
    
    print("\nF-Score by number of antennas:")
    for i in range(num_antennas):
        print(f"  {i+1} antenna(s): {average_fscore_change_num_ant[i]:.4f}")

    # Save antenna metrics
    metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
                           'average_fscore_change_num_ant': average_fscore_change_num_ant}
    unique_id = hashlib.md5(f"{csi_act}_{subdirs_complete}".encode()).hexdigest()[:8]
    name_file = f'./outputs/change_number_antennas_complete_different_{unique_id}_b{bandwidth}_sb{sub_band}.txt'

    try:
        with open(name_file, "wb") as fp:
            pickle.dump(metrics_matrix_dict, fp)
    except Exception as e:
        print(f"Error saving antenna results: {e}")
        
    # Clean up TensorFlow resources to prevent memory leaks and errors
    try:
        # Delete model explicitly
        del csi_model
        # Delete datasets explicitly
        del dataset_csi_complete
        # Clear backend session
        tf.keras.backend.clear_session()
        # Garbage collect
        gc.collect()
    except Exception as cleanup_error:
        print(f"Warning during resource cleanup: {cleanup_error}")
        
    print("\nTest completed successfully.")
