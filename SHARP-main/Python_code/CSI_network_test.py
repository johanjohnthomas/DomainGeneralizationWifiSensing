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
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tensorflow as tf
import hashlib
import matplotlib.pyplot as plt
import sys

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
        f'{name_base}_label_mapping.pkl',  # Original path
        'label_mapping.pkl',  # Generic name
        os.path.join(os.path.dirname(name_base), 'label_mapping.pkl'),  # Same directory as name_base
    ]
    
    if model_dir:
        possible_paths.extend([
            os.path.join(model_dir, f'{os.path.basename(name_base)}_label_mapping.pkl'),
            os.path.join(model_dir, 'label_mapping.pkl')
        ])
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    mapping = pickle.load(f)
                print(f"Successfully loaded label mapping from: {path}")
                return mapping, path
            except Exception as e:
                print(f"Warning: Found {path} but couldn't load it: {e}")
                continue
    
    # If we get here, no valid mapping file was found
    print("\nERROR: Label mapping file not found. Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nTo fix this:")
    print("1. Ensure you have trained the model first")
    print("2. Check if label_mapping.pkl exists in any of the above locations")
    print("3. You can manually specify the model directory using the --model_dir argument")
    sys.exit(1)

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

    # Find and load label mapping
    label_to_index, mapping_path = find_label_mapping(args.name_base, args.model_dir)
    print(f"Using label mapping with {len(label_to_index)} classes: {label_to_index}")
    
    # Create reverse mapping for validation
    index_to_label = {v: k for k, v in label_to_index.items()}
    print(f"Index to label mapping: {index_to_label}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

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
                # Validate labels before adding them
                invalid_labels = set([l for l in current_labels if l not in label_to_index])
                if invalid_labels:
                    print(f"\nERROR: Found invalid labels in {name_labels}:")
                    print(f"Invalid labels: {invalid_labels}")
                    print(f"Valid labels are: {set(label_to_index.keys())}")
                    print("\nThis usually means either:")
                    print("1. You're using a label mapping file from a different experiment")
                    print("2. The data contains labels that weren't in the training set")
                    print("3. The labels need to be remapped to match the training labels")
                    sys.exit(1)
                labels_complete.extend(current_labels)
            with open(name_f, "rb") as fp:
                all_files_complete.extend(pickle.load(fp))
        except Exception as e:
            print(f"Error loading data files: {e}")
            sys.exit(1)

    # Filter data
    file_complete_selected = [all_files_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                              labels_considered]
    labels_complete_selected = [label_to_index[labels_complete[idx]] for idx in range(len(labels_complete)) if labels_complete[idx] in
                                labels_considered ] 


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
        # Load the model
        name_model = name_base + '_' + str(csi_act) + '_network.keras'
        if not os.path.exists(name_model):
            print(f"Error: Model file not found - {name_model}")
            sys.exit(1)
            
        csi_model = load_model(name_model)
        print(f"Model loaded: {name_model}")
        
        # Process files directly
        all_predictions = []
        true_labels = []
        
        for i, file_path in enumerate(file_complete_selected):
            original_label = labels_complete_selected[i]
            if original_label not in label_to_index:
                print(f"Warning: Label {original_label} not found in label mapping.")
                continue
            true_label_index = label_to_index[original_label]  
            true_labels.append(true_label_index)
            print(f"\nProcessing file {i+1}/{len(file_complete_selected)}: {file_path}")
            
            # Load data directly
            with open(file_path, "rb") as fp:
                matrix_csi = pickle.load(fp)
                
            print(f"Loaded data shape: {matrix_csi.shape}")
            
            # Make predictions for each antenna
            file_predictions = []
            for antenna_idx in range(num_antennas):
                # Process data the same way as in the dataset pipeline
                data = matrix_csi[antenna_idx, ...].T
                if len(data.shape) < 3:
                    data = np.expand_dims(data, axis=-1)
                
                # Add batch dimension for model
                data = np.expand_dims(data, axis=0)
                
                # Make prediction
                pred = csi_model.predict(data, verbose=0)
                file_predictions.append(pred[0])
                
            # Combine predictions from all antennas
            combined_pred = np.mean(file_predictions, axis=0)
            pred_class = np.argmax(combined_pred)
            
            print(f"Prediction: class {pred_class} (true label: {original_label})")
            all_predictions.append(pred_class)
        
        # Calculate accuracy
        all_predictions = np.array(all_predictions)
        true_labels = np.array(true_labels)
        accuracy = np.mean(all_predictions == true_labels)
        
        print(f"\nTest results with direct data loading:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Predictions: {all_predictions}")
        print(f"True labels: {true_labels}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, all_predictions)
        print(f"Confusion matrix:\n{cm}")
        
        # Exit successfully
        print("\nDirect data loading test completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"Error in custom test process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # If we get here, fallback to the standard dataset approach
    # Expand antennas for dataset creation
    file_complete_selected_expanded, labels_complete_selected_expanded, stream_ant_complete = \
        expand_antennas(file_complete_selected, labels_complete_selected, num_antennas)
        
    print(f"Data files expanded to {len(file_complete_selected_expanded)} samples with {num_antennas} antennas each.")

    # Create dataset
    try:
        dataset_csi_complete = create_dataset_single(file_complete_selected_expanded, labels_complete_selected_expanded,
                                                    stream_ant_complete, input_network, batch_size, shuffle=False,
                                                    cache_file=name_base + '_' + str(csi_act) + '_cache_complete')
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)

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
    num_samples_complete = len(file_complete_selected_expanded)
    lab_complete, count_complete = np.unique(labels_complete_selected_expanded, return_counts=True)
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
    complete_labels_true = np.array(labels_complete_selected_expanded)
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

    # merge antennas
    labels_true_merge = np.array(labels_complete_selected)
    pred_max_merge = np.zeros_like(labels_complete_selected)
    for i_lab in range(len(labels_complete_selected)):
        pred_antennas = complete_prediction_list[i_lab*num_antennas:(i_lab+1)*num_antennas, :]
        sum_pred = np.sum(pred_antennas, axis=0)
        lab_merge_max = np.argmax(sum_pred)

        pred_max_antennas = complete_labels_pred[i_lab*num_antennas:(i_lab+1)*num_antennas]
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
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered, zero_division=0)
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

    # Save results
    os.makedirs('./outputs', exist_ok=True)
    unique_id = hashlib.md5(f"{csi_act}_{subdirs_complete}".encode()).hexdigest()[:8]
    name_file = f'./outputs/complete_different__{unique_id}_b{bandwidth}_sb{sub_band}.txt'
    
    try:
        with open(name_file, "wb") as fp:
            pickle.dump(metrics_matrix_dict, fp)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print('accuracy', accuracy_max_merge)
    print('fscore', fscore_max_merge)
    print(conf_matrix_max_merge)

    # impact of the number of antennas
    one_antenna = [[0], [1], [2], [3]]
    two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    four_antennas = [[0, 1, 2, 3]]
    seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    average_accuracy_change_num_ant = np.zeros((num_antennas, ))
    average_fscore_change_num_ant = np.zeros((num_antennas, ))
    labels_true_merge = np.array(labels_complete_selected)
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
        import gc
        gc.collect()
    except Exception as cleanup_error:
        print(f"Warning during resource cleanup: {cleanup_error}")
        
    print("\nTest completed successfully.")
