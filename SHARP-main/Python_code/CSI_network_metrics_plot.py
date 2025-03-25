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
from plots_utility import plt_confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name_file', help='Name of the output image file (without extension)')
    args = parser.parse_args()

    name_file = args.name_file  # Name for the output image file
    
    # Read activities from common_activities.txt instead of hardcoding
    try:
        with open("common_activities.txt", "r") as activity_file:
            activity_list = [line.strip() for line in activity_file if line.strip()]
            activities = np.array(activity_list)
            print(f"Using activities from common_activities.txt: {activities}")
    except Exception as e:
        print(f"Warning: Could not read common_activities.txt: {e}")
        print("Falling back to determining activities from the confusion matrix dimensions")
        
        # We need to read the metrics file first to get the dimensions in the fallback case
        folder_name = './outputs/'

        # Find the latest metrics file
        latest_file = None
        for file in os.listdir(folder_name):
            if file.startswith('complete_different_') and file.endswith('.txt'):
                if latest_file is None or os.path.getmtime(os.path.join(folder_name, file)) > os.path.getmtime(os.path.join(folder_name, latest_file)):
                    latest_file = file
        
        if latest_file is None:
            print("No metrics file found in outputs directory")
            exit(1)
            
        metrics_file = os.path.join(folder_name, latest_file)
        with open(metrics_file, "rb") as fp:
            tmp_dict = pickle.load(fp)
        num_activities = tmp_dict['conf_matrix'].shape[0]
        activities = np.array([chr(65 + i) for i in range(num_activities)])  # Use A, B, C, etc.
        print(f"Using fallback activities: {activities}")

    folder_name = './outputs/'

    # Find the latest metrics file
    latest_file = None
    for file in os.listdir(folder_name):
        if file.startswith('complete_different_') and file.endswith('.txt'):
            if latest_file is None or os.path.getmtime(os.path.join(folder_name, file)) > os.path.getmtime(os.path.join(folder_name, latest_file)):
                latest_file = file
    
    if latest_file is None:
        print("No metrics file found in outputs directory")
        exit(1)
        
    metrics_file = os.path.join(folder_name, latest_file)
    print(f"Using metrics file: {metrics_file}")

    with open(metrics_file, "rb") as fp:  # Pickling
        conf_matrix_dict = pickle.load(fp)
    
    conf_matrix = conf_matrix_dict['conf_matrix']
    conf_matrix_max_merge = conf_matrix_dict['conf_matrix_max_merge']

    name_plot = name_file
    plt_confusion_matrix(activities.shape[0], conf_matrix, activities=activities, name=name_plot)
    print(f"Created confusion matrix plot: {name_plot}.png")

    name_plot = name_file + '_max_merge'
    plt_confusion_matrix(activities.shape[0], conf_matrix_max_merge, activities=activities, name=name_plot)
    print(f"Created max merge confusion matrix plot: {name_plot}.png")
