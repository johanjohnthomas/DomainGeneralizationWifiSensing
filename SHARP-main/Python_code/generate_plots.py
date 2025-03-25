#!/usr/bin/env python3
"""
Simple script to generate example visualizations from the test results.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from plots_utility import plt_confusion_matrix

def main():
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    print("Generating example plots from the test results...")
    
    # Find the latest metrics file
    latest_metrics_file = None
    folder_name = './outputs/'
    for file in os.listdir(folder_name):
        if file.startswith('complete_different_') and file.endswith('.txt'):
            if latest_metrics_file is None or os.path.getmtime(os.path.join(folder_name, file)) > os.path.getmtime(os.path.join(folder_name, latest_metrics_file)):
                latest_metrics_file = file
    
    if latest_metrics_file is None:
        print("No metrics file found in outputs directory")
        return
    
    metrics_file = os.path.join(folder_name, latest_metrics_file)
    print(f"Using metrics file: {metrics_file}")
    
    # Load the confusion matrices
    try:
        with open(metrics_file, "rb") as fp:
            conf_matrix_dict = pickle.load(fp)
        
        # Read activities from common_activities.txt instead of hardcoding
        try:
            with open("common_activities.txt", "r") as activity_file:
                activity_list = [line.strip() for line in activity_file if line.strip()]
                activities = np.array(activity_list)
                print(f"Using activities from common_activities.txt: {activities}")
        except Exception as e:
            print(f"Warning: Could not read common_activities.txt: {e}")
            print("Falling back to using activities from the confusion matrix dimensions")
            # Fallback: determine activities from the confusion matrix dimensions
            num_activities = conf_matrix_dict['conf_matrix'].shape[0]
            activities = np.array([chr(65 + i) for i in range(num_activities)])  # Use A, B, C, etc.
            print(f"Using fallback activities: {activities}")
        
        conf_matrix = conf_matrix_dict['conf_matrix']
        conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']
        
        # Plot confusion matrix
        plt_confusion_matrix(activities.shape[0], conf_matrix, activities=activities, name="example_confusion_matrix")
        print("Generated example confusion matrix plot")
        
        # Plot max merge confusion matrix
        plt_confusion_matrix(activities.shape[0], conf_matrix_max_merge, activities=activities, name="example_max_merge_confusion_matrix")
        print("Generated example max merge confusion matrix plot")
        
        # Create a simple bar chart of accuracies
        plt.figure(figsize=(10, 6))
        acc_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
        plt.bar(activities, acc_per_class)
        plt.xlabel('Activity')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('./plots/per_class_accuracy.png')
        plt.savefig('./plots/per_class_accuracy.pdf')
        plt.close()
        print("Generated per-class accuracy plot")
        
        # Create a simple bar chart of F1 scores if available
        if 'f1_scores' in conf_matrix_dict:
            plt.figure(figsize=(10, 6))
            plt.bar(activities, conf_matrix_dict['f1_scores'])
            plt.xlabel('Activity')
            plt.ylabel('F1 Score')
            plt.title('Per-Class F1 Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('./plots/per_class_f1.png')
            plt.savefig('./plots/per_class_f1.pdf')
            plt.close()
            print("Generated per-class F1 score plot")
        
        print(f"All plots have been saved to {os.path.abspath('./plots/')}")
        print("Available plots:")
        for plot_file in os.listdir('./plots/'):
            print(f"  - {plot_file}")
    
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == '__main__':
    main() 