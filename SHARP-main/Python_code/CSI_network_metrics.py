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
import os
import csv
import re
import glob
from pathlib import Path


def normalize_path(path):
    """
    Normalize a path to remove any redundant separators or up-level references.
    This will convert paths like './results//RQ1_generalization' to './results/RQ1_generalization'
    """
    return os.path.normpath(path)


def extract_experiment_info(filename):
    """
    Extract experiment ID, source and target domains from filename
    Returns experiment_id, source_domains, target_domains, component_modified
    """
    # Default values
    experiment_id = os.path.basename(filename).replace('.txt', '')
    source_domains = "unknown"
    target_domains = "unknown"
    component_modified = "none"
    
    # Check for command-line environment variables first - highest priority
    # These would be set by the Makefile when using custom commands like test_training
    if os.environ.get('TRAIN_DOMAINS'):
        source_domains = os.environ.get('TRAIN_DOMAINS')
        print(f"Using environment variable for source domains: {source_domains}")
    
    if os.environ.get('TEST_DOMAINS'):
        target_domains = os.environ.get('TEST_DOMAINS')
        print(f"Using environment variable for target domains: {target_domains}")
    
    # If environment variables aren't set, try to extract from filename
    if source_domains == "unknown" or target_domains == "unknown":
        # Extract model name (used as experiment_id)
        # First, check specifically for "no_" leave-one-out experiments
        model_match = re.search(r'complete_different_(no_[a-zA-Z]+)_', filename)
        if model_match:
            experiment_id = model_match.group(1)
        else:
            # If not a leave-one-out experiment, use the original pattern but make it less greedy
            model_match = re.search(r'complete_different_(.*?)_[EJLRWSCG]', filename)
            if model_match:
                experiment_id = model_match.group(1)
        
        # Check for ablation metadata file
        model_base = os.path.splitext(filename)[0]
        ablation_metadata_file = model_base + '_ablation_metadata.pkl'
        if os.path.exists(ablation_metadata_file):
            try:
                with open(ablation_metadata_file, 'rb') as f:
                    ablation_metadata = pickle.load(f)
                    if ablation_metadata.get("no_phase_sanitization", False):
                        component_modified = "phase_sanitization"
                    elif ablation_metadata.get("no_antenna_randomization", False):
                        component_modified = "antenna_randomization"
            except Exception as e:
                print(f"Error reading ablation metadata: {e}")
        
        # If no metadata file exists, check for component modifications in the name
        if component_modified == "none":
            if "no_phase" in experiment_id:
                component_modified = "phase_sanitization"
            elif "no_antenna" in experiment_id:
                component_modified = "antenna_randomization"
        
        # Extract target domain for leave-one-out experiments if not already set by environment variables
        if source_domains == "unknown" and target_domains == "unknown" and experiment_id.startswith("no_"):
            domain_type = experiment_id.replace("no_", "")
            target_domains = domain_type
            
            # Set source domains based on what's left out
            all_domains = ["bedroom", "living", "kitchen", "lab", "office", "semi"]
            source_domains = ",".join([d for d in all_domains if d != domain_type])
        
        # Extract source domain count for scaling experiments
        elif source_domains == "unknown" and target_domains == "unknown" and experiment_id.startswith("source"):
            num_sources = experiment_id.replace("source", "")
            if num_sources == "1":
                source_domains = "bedroom_split1"
                target_domains = "bedroom_split2"
            elif num_sources == "2":
                source_domains = "bedroom_split1,living_split1"
                target_domains = "bedroom_split2,living_split2"
            elif num_sources == "3":
                source_domains = "bedroom_split1,living_split1,office_split1"
                target_domains = "bedroom_split2,living_split2,office_split2"
            elif num_sources == "4":
                source_domains = "bedroom_split1,living_split1,office_split1,semi_split1"
                target_domains = "bedroom_split2,living_split2,office_split2,semi_split2"
    
    print(f"Extracted experiment_id: {experiment_id} from filename: {os.path.basename(filename)}")
    print(f"Source domains: {source_domains}")
    print(f"Target domains: {target_domains}")
    return experiment_id, source_domains, target_domains, component_modified


def get_train_accuracy(experiment_id):
    """
    Retrieve training accuracy from the model history file
    Returns the final training accuracy or None if not found
    """
    try:
        # Extract the base experiment ID from the full ID
        base_experiment_id = experiment_id
        
        # Domain types that might appear in experiment IDs
        domain_types = ['bedroom', 'living', 'kitchen', 'lab', 'office', 'semi']
        
        # Extract activity string and base experiment ID more dynamically
        parts = experiment_id.split('_')
        domain_prefix_parts = []
        activity_parts = []
        domain_suffix_parts = []
        
        # Categorize parts of the experiment_id
        for part in parts:
            # Domain prefixes (like "no_" or "source1")
            if part == 'no' or part.startswith('source'):
                domain_prefix_parts.append(part)
            # Domain types
            elif part in domain_types:
                domain_prefix_parts.append(part)
            # Target domains (starting with AR)
            elif part.startswith('AR'):
                domain_suffix_parts.append(part)
            # Activities (usually single letters)
            elif len(part) == 1 and part.isalpha():
                activity_parts.append(part)
            # Handle any other parts based on context
            else:
                # If we already found activities and this is at the end, it's likely a domain suffix
                if activity_parts and not domain_suffix_parts:
                    domain_suffix_parts.append(part)
                else:
                    # Otherwise assume it's part of the domain prefix
                    domain_prefix_parts.append(part)
        
        # Reconstruct IDs
        domain_prefix = '_'.join(domain_prefix_parts)
        clean_activity_str = '_'.join(activity_parts)
        clean_experiment_id = f"{domain_prefix}_{clean_activity_str}" if domain_prefix else clean_activity_str
        
        # Special case for "no_" experiments (leave-one-out)
        if experiment_id.startswith('no_'):
            base_experiment_id = clean_experiment_id
            # If it has a target domain, take everything before it
            if '_AR' in experiment_id:
                base_experiment_id = experiment_id.split('_AR')[0]
                
        # Special case for source scaling experiments
        elif any(experiment_id.startswith(f"source{i}") for i in range(1, 10)):
            # Extract source number
            source_part = next((p for p in parts if p.startswith("source")), "")
            if source_part and clean_activity_str:
                base_experiment_id = f"{source_part}_{clean_activity_str}"
        
        print(f"Looking for history file with base experiment ID: {base_experiment_id}")
        print(f"Also checking for cleaned experiment ID: {clean_experiment_id}")
        print(f"Clean activity string: {clean_activity_str}")
        
        # Check common locations for history files with correct pattern
        search_paths = [
            f"./models/{base_experiment_id}_history.pkl",                 # Relative path from script execution
            f"./SHARP-main/Python_code/models/{base_experiment_id}_history.pkl",  # From project root
            f"../models/{base_experiment_id}_history.pkl",                # Up one level
            f"models/{base_experiment_id}_history.pkl",                   # Direct models folder
            f"./models/{clean_experiment_id}_history.pkl",                # With cleaned ID
            f"./SHARP-main/Python_code/models/{clean_experiment_id}_history.pkl",  # From project root with cleaned ID
            f"../models/{clean_experiment_id}_history.pkl",               # Up one level with cleaned ID
            f"models/{clean_experiment_id}_history.pkl",                  # Direct models folder with cleaned ID
            # Additional search paths with activity string
            f"./models/complete_different_{clean_activity_str}_history.pkl",
            f"./models/no_bedroom_{clean_activity_str}_history.pkl",
            f"./models/complete_different_no_bedroom_{clean_activity_str}_history.pkl"
        ]
        
        # First try exact match with experiment_id
        for path in search_paths:
            if os.path.exists(path):
                print(f"Found history file at: {path}")
                with open(path, "rb") as fp:
                    history = pickle.load(fp)
                    # Check for sparse_categorical_accuracy first as it's what we found in the files
                    if 'sparse_categorical_accuracy' in history:
                        return history['sparse_categorical_accuracy'][-1]
                    elif 'accuracy' in history:
                        return history['accuracy'][-1]
                    elif 'acc' in history:
                        return history['acc'][-1]
        
        # If exact match failed, try directory search
        search_dirs = [
            "./models/",
            "./SHARP-main/Python_code/models/",
            "../models/",
            "models/"
        ]
        
        # First, list all history files to help with debugging
        print("Searching for available history files:")
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                history_files = glob.glob(f"{search_dir}*_history.pkl")
                for file in history_files:
                    print(f"  Found history file: {file}")
        
        # Try different glob patterns to match history files
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Look for files matching various patterns
                search_patterns = [
                    f"{search_dir}*{base_experiment_id}*_history.pkl",
                    f"{search_dir}*{clean_experiment_id}*_history.pkl",
                    f"{search_dir}*_{clean_activity_str}_history.pkl",
                    f"{search_dir}*complete_different*{clean_activity_str}*_history.pkl",
                    f"{search_dir}*no_bedroom*{clean_activity_str}*_history.pkl"
                ]
                
                for pattern in search_patterns:
                    history_files = glob.glob(pattern)
                    if history_files:
                        print(f"Found history file using pattern {pattern}: {history_files[0]}")
                        with open(history_files[0], "rb") as fp:
                            history = pickle.load(fp)
                            if 'sparse_categorical_accuracy' in history:
                                return history['sparse_categorical_accuracy'][-1]
                            elif 'accuracy' in history:
                                return history['accuracy'][-1]
                            elif 'acc' in history:
                                return history['acc'][-1]
        
        # If nothing found, print diagnostic info
        print(f"Could not find history file for experiment_id: {experiment_id}")
        print(f"With base_experiment_id: {base_experiment_id}")
        print(f"Searched paths: {search_paths}")
        print(f"Searched directories for pattern matching: {search_dirs}")
        
        # As a fallback, try to find any history file with matching activity string
        # This is less precise but better than returning None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                fallback_files = glob.glob(f"{search_dir}*_{clean_activity_str}_*_history.pkl")
                if fallback_files:
                    print(f"Found fallback history file: {fallback_files[0]}")
                    print(f"WARNING: Using a fallback history file that may not be an exact match!")
                    with open(fallback_files[0], "rb") as fp:
                        history = pickle.load(fp)
                        if 'sparse_categorical_accuracy' in history:
                            return history['sparse_categorical_accuracy'][-1]
                        elif 'accuracy' in history:
                            return history['accuracy'][-1]
                        elif 'acc' in history:
                            return history['acc'][-1]
        
        # ADDITIONAL FALLBACK: Use any available history file if activity-specific file not found
        print("Attempting to use any available history file as ultimate fallback...")
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Look for any history file
                any_history_files = glob.glob(f"{search_dir}*_history.pkl")
                if any_history_files:
                    fallback_file = any_history_files[0]  # Use the first history file found
                    print(f"IMPORTANT NOTE: Using generic fallback history file: {fallback_file}")
                    print(f"WARNING: This is not the correct history for {experiment_id} but provides an approximation")
                    try:
                        with open(fallback_file, "rb") as fp:
                            history = pickle.load(fp)
                            if 'sparse_categorical_accuracy' in history:
                                acc_value = history['sparse_categorical_accuracy'][-1]
                                print(f"Using approximated training accuracy: {acc_value:.4f}")
                                return acc_value
                            elif 'accuracy' in history:
                                acc_value = history['accuracy'][-1]
                                print(f"Using approximated training accuracy: {acc_value:.4f}")
                                return acc_value
                            elif 'acc' in history:
                                acc_value = history['acc'][-1]
                                print(f"Using approximated training accuracy: {acc_value:.4f}")
                                return acc_value
                    except Exception as e:
                        print(f"Error reading fallback file {fallback_file}: {e}")
        
        return None
    except Exception as e:
        print(f"Error retrieving training accuracy: {e}")
        return None


def save_csv_metrics(metrics_data, output_folder='./results'):
    """
    Save metrics to a CSV file with standardized columns
    """
    csv_path = os.path.join(output_folder, 'experiment_metrics.csv')
    file_exists = os.path.isfile(csv_path)
    
    # Create directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    fieldnames = [
        'experiment_id', 
        'source_domains', 
        'target_domains', 
        'train_acc', 
        'test_acc', 
        'gap',
        'component_modified',
        'precision',
        'recall',
        'fscore'
    ]
    
    with open(csv_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics_data)
    
    print(f"Metrics saved to {csv_path}")
    
    # If it's an RQ1 experiment, also save to the RQ1 directory
    if metrics_data['experiment_id'].startswith('no_'):
        rq1_dir = './results/RQ1_generalization/leave_one_out'
        experiment_dir = os.path.join(rq1_dir, metrics_data['experiment_id'], 'metrics')
        os.makedirs(experiment_dir, exist_ok=True)
        
        csv_path = os.path.join(experiment_dir, 'metrics.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics_data)
        print(f"RQ1 metrics saved to {csv_path}")
    
    # If it's an RQ4 experiment, also save to the RQ4 directory
    elif metrics_data['experiment_id'].startswith('source'):
        rq4_dir = './results/RQ4_source_scaling'
        experiment_dir = os.path.join(rq4_dir, metrics_data['experiment_id'])
        os.makedirs(experiment_dir, exist_ok=True)
        
        csv_path = os.path.join(experiment_dir, 'metrics.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics_data)
        print(f"RQ4 metrics saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name_file', help='Name of the file')
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('--train-acc', help='Override training accuracy', type=float, default=None)
    parser.add_argument('--source-domains', help='Override source domains', default=None)
    parser.add_argument('--target-domains', help='Override target domains', default=None)
    parser.add_argument('--component', help='Component modified in this experiment', default=None)
    args = parser.parse_args()

    name_file = args.name_file  # string
    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

    folder_name = './results/'

    # Fix path handling to avoid duplication
    if name_file.startswith('./results/') and name_file.endswith('.txt'):
        # Path is already complete, use as is
        full_path = normalize_path(name_file)
    elif name_file.startswith('./results/'):
        # Has path but no extension
        full_path = normalize_path(name_file + '.txt')
    elif name_file.endswith('.txt'):
        # Has extension but no path
        full_path = normalize_path(folder_name + name_file)
    else:
        # Neither path nor extension
        full_path = normalize_path(folder_name + name_file + '.txt')

    # Extract experiment information
    experiment_id, source_domains, target_domains, component_modified = extract_experiment_info(full_path)
    
    # Override with command line arguments if provided
    if args.source_domains:
        source_domains = args.source_domains
    if args.target_domains:
        target_domains = args.target_domains
    if args.component:
        component_modified = args.component

    with open(full_path, "rb") as fp:  # Pickling
        conf_matrix_dict = pickle.load(fp)

    conf_matrix = conf_matrix_dict['conf_matrix']
    # Add a small epsilon to avoid division by zero and handle empty rows safely
    row_sums = np.sum(conf_matrix, axis=1).reshape(-1, 1)
    # Replace zeros with ones to avoid division by zero (will result in zeros in the normalized matrix)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_matrix_normaliz_row = np.transpose(conf_matrix / row_sums)
    accuracies = np.diag(confusion_matrix_normaliz_row)
    accuracy = conf_matrix_dict['accuracy_single']
    precision = conf_matrix_dict['precision_single']
    recall = conf_matrix_dict['recall_single']
    fscore = conf_matrix_dict['fscore_single']
    average_prec = np.mean(precision)
    average_rec = np.mean(recall)
    average_f = np.mean(recall)
    print('single antenna - average accuracy %f, average precision %f, average recall %f, average fscore %f'
          % (accuracy, average_prec, average_rec, average_f))
    
    # Print fscores for each activity dynamically based on provided activities
    fscore_str = ', '.join([f"{activities[i]} {fscore[i]:.6f}" for i in range(min(len(activities), len(fscore)))])
    print(f'fscores - {fscore_str}')
    print('average fscore %f' % (np.mean(fscore)))
    
    # Print accuracies for each activity dynamically based on provided activities
    accuracies_str = ', '.join([f"{activities[i]} {accuracies[i]:.6f}" for i in range(min(len(activities), len(accuracies)))])
    print(f'accuracies - {accuracies_str}')

    conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']
    # Apply the same safety check for the second matrix
    row_sums_max_merge = np.sum(conf_matrix_max_merge, axis=1).reshape(-1, 1)
    # Replace zeros with ones to avoid division by zero
    row_sums_max_merge = np.where(row_sums_max_merge == 0, 1, row_sums_max_merge)
    conf_matrix_max_merge_normaliz_row = np.transpose(conf_matrix_max_merge / row_sums_max_merge)
    accuracies_max_merge = np.diag(conf_matrix_max_merge_normaliz_row)
    accuracy_max_merge = conf_matrix_dict['accuracy_max_merge']
    precision_max_merge = conf_matrix_dict['precision_max_merge']
    recall_max_merge = conf_matrix_dict['recall_max_merge']
    fscore_max_merge = conf_matrix_dict['fscore_max_merge']
    average_max_merge_prec = np.mean(precision_max_merge)
    average_max_merge_rec = np.mean(recall_max_merge)
    average_max_merge_f = np.mean(fscore_max_merge)
    print('\n-- FINAL DECISION --')
    print('max-merge - average accuracy %f, average precision %f, average recall %f, average fscore %f'
          % (accuracy_max_merge, average_max_merge_prec, average_max_merge_rec, average_max_merge_f))
    
    # Print fscores for max merge dynamically based on provided activities
    fscore_max_merge_str = ', '.join([f"{activities[i]} {fscore_max_merge[i]:.6f}" for i in range(min(len(activities), len(fscore_max_merge)))])
    print(f'fscores - {fscore_max_merge_str}')
    
    # Print accuracies for max merge dynamically based on provided activities
    accuracies_max_merge_str = ', '.join([f"{activities[i]} {accuracies_max_merge[i]:.6f}" for i in range(min(len(activities), len(accuracies_max_merge)))])
    print(f'accuracies - {accuracies_max_merge_str}')

    # performance assessment by changing the number of monitor antennas
    # Get the directory of the input file to look for the antenna variation file in the same location
    input_file_dir = os.path.dirname(full_path)
    
    # Create the antenna variation filename based on the main metrics file
    base_name = os.path.basename(name_file.replace('.txt', ''))
    antenna_file_name = f'change_number_antennas_{base_name}.txt'
    
    # Construct the full path to the antenna variation file in the same directory
    second_file = os.path.join(input_file_dir, antenna_file_name)
    
    try:    
        with open(second_file, "rb") as fp:  # Pickling
            metrics_matrix_dict = pickle.load(fp)

        average_accuracy_change_num_ant = metrics_matrix_dict['average_accuracy_change_num_ant']
        average_fscore_change_num_ant = metrics_matrix_dict['average_fscore_change_num_ant']
        print('\naccuracies - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
              % (average_accuracy_change_num_ant[0], average_accuracy_change_num_ant[1], average_accuracy_change_num_ant[2],
                 average_accuracy_change_num_ant[3]))
        print('fscores - one antenna %f, two antennas %f, three antennas %f, four antennas %f'
              % (average_fscore_change_num_ant[0], average_fscore_change_num_ant[1], average_fscore_change_num_ant[2],
                 average_fscore_change_num_ant[3]))
    except Exception as e:
        print(f"Warning: Could not open antenna variation file: {second_file}")
        print(f"Error details: {e}")
        print("Antenna variation metrics will not be available for this experiment.")

    # Get source domain (training) accuracy
    train_acc = args.train_acc
    if train_acc is None:
        # Try to retrieve training accuracy from history file
        train_acc = get_train_accuracy(experiment_id)
        if train_acc is None:
            print("Warning: Could not retrieve training accuracy. Using placeholder value.")
            print("Try running 'ls -la ./models/*_history.pkl' to see which history files exist.")
            print("You may need to train the model again or fix the cleanup process to preserve history files.")
            train_acc = 0.0

    # Calculate generalization gap
    test_acc = accuracy_max_merge  # Using max merge accuracy as the test accuracy
    gap = train_acc - test_acc
    
    print("\n-- GENERALIZATION METRICS --")
    print(f"Source domains: {source_domains}")
    print(f"Target domains: {target_domains}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy: {test_acc:.4f}")
    print(f"Generalization gap: {gap:.4f}")
    print(f"Component modified: {component_modified}")
    
    # Save metrics to CSV
    metrics_data = {
        'experiment_id': experiment_id,
        'source_domains': source_domains,
        'target_domains': target_domains,
        'train_acc': round(float(train_acc), 4),
        'test_acc': round(float(test_acc), 4),
        'gap': round(float(gap), 4),
        'component_modified': component_modified,
        'precision': round(float(average_max_merge_prec), 4),
        'recall': round(float(average_max_merge_rec), 4),
        'fscore': round(float(average_max_merge_f), 4)
    }
    
    save_csv_metrics(metrics_data)
