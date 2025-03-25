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
    
    # Extract model name (used as experiment_id)
    model_match = re.search(r'complete_different.*?_([a-zA-Z0-9_]+)_band', filename)
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
    
    # Extract target domain for leave-one-out experiments
    if experiment_id.startswith("no_"):
        domain_type = experiment_id.replace("no_", "")
        target_domains = domain_type
        
        # Set source domains based on what's left out
        all_domains = ["bedroom", "living", "kitchen", "lab", "office", "semi"]
        source_domains = ",".join([d for d in all_domains if d != domain_type])
    
    # Extract source domain count for scaling experiments
    elif experiment_id.startswith("source"):
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
    
    return experiment_id, source_domains, target_domains, component_modified


def get_train_accuracy(experiment_id):
    """
    Retrieve training accuracy from the model history file
    Returns the final training accuracy or None if not found
    """
    try:
        history_file = f"./models/{experiment_id}_history.pkl"
        if os.path.exists(history_file):
            with open(history_file, "rb") as fp:
                history = pickle.load(fp)
                if 'accuracy' in history:
                    return history['accuracy'][-1]  # Get final training accuracy
                elif 'acc' in history:
                    return history['acc'][-1]  # Some models use 'acc' instead
        
        # Fallback to models directory search
        history_files = glob.glob(f"./models/*{experiment_id}*_history.pkl")
        if history_files:
            with open(history_files[0], "rb") as fp:
                history = pickle.load(fp)
                if 'accuracy' in history:
                    return history['accuracy'][-1]
                elif 'acc' in history:
                    return history['acc'][-1]
        
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
    print('fscores - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (fscore[0], fscore[1], fscore[2], fscore[3], fscore[4]))
    print('average fscore %f' % (np.mean(fscore)))
    print('accuracies - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4]))

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
    print('fscores - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (fscore_max_merge[0], fscore_max_merge[1], fscore_max_merge[2], fscore_max_merge[3], fscore_max_merge[4]))
    print('accuracies - empty %f, sitting %f, walking %f, running %f, jumping %f'
          % (accuracies_max_merge[0], accuracies_max_merge[1], accuracies_max_merge[2], accuracies_max_merge[3],
             accuracies_max_merge[4]))

    # performance assessment by changing the number of monitor antennas
    # Fix the second file path handling as well
    second_file_base = 'change_number_antennas_' + os.path.basename(name_file.replace('.txt', ''))
    if second_file_base.startswith('./results/') and second_file_base.endswith('.txt'):
        second_file = normalize_path(second_file_base)
    elif second_file_base.startswith('./results/'):
        second_file = normalize_path(second_file_base + '.txt')
    elif second_file_base.endswith('.txt'):
        second_file = normalize_path(folder_name + second_file_base)
    else:
        second_file = normalize_path(folder_name + second_file_base + '.txt')
    
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
        print(f"Warning: Could not open antenna variation file: {e}")

    # Get source domain (training) accuracy
    train_acc = args.train_acc
    if train_acc is None:
        # Try to retrieve training accuracy from history file
        train_acc = get_train_accuracy(experiment_id)
        if train_acc is None:
            print("Warning: Could not retrieve training accuracy. Using placeholder value.")
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
