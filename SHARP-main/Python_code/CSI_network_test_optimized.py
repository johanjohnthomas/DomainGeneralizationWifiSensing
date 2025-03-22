#!/usr/bin/env python3
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
    
    OPTIMIZED VERSION WITH GPU PERFORMANCE IMPROVEMENTS
"""

import argparse
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
import glob
import time
import gc
from dataset_utility import create_dataset_single, expand_antennas
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tensorflow as tf

# Set environment variables for better GPU performance
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "32"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress most TensorFlow logs

# Clear any TensorFlow cache lockfiles
for lockfile in glob.glob("*_cache_*.lockfile"):
    try:
        os.remove(lockfile)
        print(f"Removed lockfile: {lockfile}")
    except:
        pass

# Configure mixed precision for faster processing
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    using_mixed_precision = True
    print("Using mixed precision (float16) for faster GPU processing")
except:
    using_mixed_precision = False
    print("Mixed precision not available")

def create_mirrored_strategy():
    """Create a MirroredStrategy for GPU utilization"""
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using strategy: {strategy.__class__.__name__}")
        return strategy
    except:
        print("Using default strategy")
        return None

if __name__ == '__main__':
    # Record start time to measure performance
    start_time = time.time()
    
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
    parser.add_argument('--prefetch_buffer', help='Number of batches to prefetch (default 4)', 
                       default=4, required=False, type=int)
    parser.add_argument('--no_cache', help='Disable dataset caching to avoid lockfile issues',
                       action='store_true')
    parser.add_argument('--max_memory', help='Maximum GPU memory to allocate in MB (default: 0=all available)',
                       default=0, required=False, type=int)
    args = parser.parse_args()

    # Configure GPU and check availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    
    if gpus:
        try:
            # Set memory growth to prevent OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optionally limit GPU memory if requested
            if args.max_memory > 0:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.max_memory)]
                )
                print(f"Limited GPU memory to {args.max_memory}MB")
            else:
                print("Using all available GPU memory with growth enabled")
                
            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"Using GPU: {gpu_details.get('device_name', 'Unknown')}")
            print(f"Compute capability: {gpu_details.get('compute_capability', 'Unknown')}")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found - running on CPU which will be much slower")

    # Create a strategy for model loading and prediction
    strategy = create_mirrored_strategy()
    
    # Parse activities list
    activities = np.array(args.activities.split(','))
    
    # Set suffix based on bandwidth
    bandwidth = args.bandwidth
    if bandwidth == 80:
        suffix = '_BW80.pkl'
    elif bandwidth == 40:
        suffix = '_BW40_' + str(args.sub_band) + '.pkl'
    elif bandwidth == 20:
        suffix = '_BW20_' + str(args.sub_band) + '.pkl'
    else:
        suffix = '_BW80.pkl'
        
    # Set model base name
    name_base = args.name_base
    subdirs_complete = args.subdirs
    csi_act = args.activities.replace(',', '_')
    
    # Generate activity mapping
    labels_map = {}
    for i, activity in enumerate(activities):
        labels_map[i] = activity
    
    # Print labels mapping
    print("\nActivity Mapping:")
    for idx, label in labels_map.items():
        print(f"  {idx}: {label}")
    
    # Process input shapes
    num_antennas = args.num_tot
    feature_length = args.feature_length
    sample_length = args.sample_length
    channels = args.channels
    batch_size = args.batch_size
    input_shape = (sample_length, feature_length, channels)
    
    print(f"\nInput Parameters:")
    print(f"  Feature length: {feature_length}")
    print(f"  Sample length: {sample_length}")
    print(f"  Channels: {channels}")
    print(f"  Number of antennas: {num_antennas}")
    print(f"  Batch size: {batch_size}")
    print(f"  Using dataset caching: {not args.no_cache}")
    print(f"  Prefetch buffer size: {args.prefetch_buffer}")
    
    # Initialize variables for data collection
    labels_complete = []
    all_files_complete = []
    all_streams_complete = []
    
    # Load test data from each subdirectory
    print("\nLoading test data from directories:")
    for sdir in subdirs_complete.split(','):
        if not sdir:  # Skip empty entries
            continue
            
        print(f"  Processing {sdir}...")
        dir_complete = args.dir + sdir + '/complete_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_complete_antennas_' + str(csi_act) + suffix
        name_f = args.dir + sdir + '/files_complete_antennas_' + str(csi_act) + suffix
        
        # Check if files exist
        if not os.path.exists(name_labels):
            print(f"Error: File not found - {name_labels}")
            continue
            
        if not os.path.exists(name_f):
            print(f"Error: File not found - {name_f}")
            continue
        
        # Load labels and filenames
        try:
            with open(name_labels, "rb") as fp:
                domain_labels = pickle.load(fp)
                
            with open(name_f, "rb") as fp:
                domain_files = pickle.load(fp)
                
            # Create antenna streams array (all ones for now)
            domain_streams = np.ones((len(domain_labels),), dtype=np.int32)
                
            # Extend our data arrays
            labels_complete.extend(domain_labels)
            all_files_complete.extend(domain_files)
            all_streams_complete.extend(domain_streams.tolist())
                
            # Print statistics for this domain
            unique_labels, counts = np.unique(domain_labels, return_counts=True)
            domain_stats = {activities[l]: c for l, c in zip(unique_labels, counts)}
            print(f"    Loaded {len(domain_labels)} samples with distribution: {domain_stats}")
                
        except Exception as e:
            print(f"Error loading data from {sdir}: {e}")
            continue
    
    # Convert lists to numpy arrays
    labels_complete = np.array(labels_complete)
    all_files_complete = np.array(all_files_complete)
    all_streams_complete = np.array(all_streams_complete)
    
    # Get total number of samples
    num_samples_complete = len(labels_complete)
    
    # Expand antennas if needed
    if num_antennas > 1:
        print("\nExpanding samples for multiple antennas...")
        labels_complete, file_complete_expanded, stream_ant_complete = expand_antennas(
            num_antennas, labels_complete, all_files_complete, all_streams_complete)
        print(f"Expanded to {len(labels_complete)} samples")
    else:
        file_complete_expanded = all_files_complete
        stream_ant_complete = all_streams_complete
        
    # Get sample count per class
    unique_labels, counts = np.unique(labels_complete, return_counts=True)
    print("\nSample Distribution After Expansion:")
    for l, c in zip(unique_labels, counts):
        if l < len(activities):
            print(f"  {activities[l]}: {c} samples")
            
    # Create optimized dataset
    print("\nCreating dataset pipeline...")
    name_cache_complete = None if args.no_cache else f"{name_base}_{csi_act}_cache_complete"
    
    # Create the dataset with optimized parameters
    dataset_csi_complete = create_dataset_single(
        file_complete_expanded, 
        labels_complete,
        stream_ant_complete, 
        input_shape, 
        batch_size,
        shuffle=False, 
        cache_file=name_cache_complete, 
        prefetch=args.prefetch_buffer
    )
    
    # Inspect dataset shapes
    for shapes in dataset_csi_complete.element_spec:
        if hasattr(shapes, 'shape'):
            print(f"[DEBUG] Raw CSI shape: {shapes.shape}")
    
    # Get a test batch to inspect shape
    for x, y in dataset_csi_complete.take(1):
        print(f"[DEBUG] Final dataset shape: ({x.shape}, {y.shape})")
        break
        
    # Define custom loss function for focal loss support
    def focal_loss(gamma=2., alpha=4.):
        def focal_loss_fixed(y_true, y_pred):
            # Focal loss implementation
            epsilon = 1e-9
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = tf.pow(1. - y_pred, gamma) * alpha
            loss = weight * cross_entropy
            return tf.reduce_sum(loss, axis=-1)
        return focal_loss_fixed
        
    # Define custom metrics wrapper
    def create_per_class_metrics(num_classes):
        # For custom metrics
        return lambda y_true, y_pred: tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(y_true, tf.int32), tf.argmax(y_pred, axis=1, output_type=tf.int32)), tf.float32)
        )
    
    # Define custom objects dictionary for model loading
    custom_objects = {
        'focal_loss': focal_loss,
        'create_per_class_metrics': create_per_class_metrics
    }
    
    # Attempt to load the model
    model_loaded = False
    model_path = f"{name_base}_{csi_act}_network.keras"
    
    print(f"\nLoading model...")
    print(f"Strategy 1: Attempting to load model from {model_path}")
    
    try:
        if strategy:
            with strategy.scope():
                csi_model = tf.keras.models.load_model(model_path)
        else:
            csi_model = tf.keras.models.load_model(model_path)
            
        # Print model summary
        print("Model loaded successfully with .keras format!")
        model_loaded = True
        
    except Exception as e:
        print(f"Could not load with standard format: {e}")
        
    # Try alternative loading methods if needed
    if not model_loaded:
        print("Strategy 2: Attempting to load with custom objects...")
        try:
            if strategy:
                with strategy.scope():
                    csi_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                csi_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                
            print("Model loaded successfully with custom objects!")
            model_loaded = True
        except Exception as e:
            print(f"Could not load with custom objects: {e}")
            
    # Exit if model loading failed
    if not model_loaded:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Predict with optimized batch processing
    print("\nRunning inference on test data...")
    complete_steps_per_epoch = int(np.ceil(num_samples_complete / batch_size))
    
    # Set a reasonable number of steps to avoid issue with cache lockfiles
    max_steps = min(complete_steps_per_epoch, 10000)
    
    try:
        # Use predict with optimizations
        complete_prediction_list = csi_model.predict(
            dataset_csi_complete,
            steps=complete_steps_per_epoch, 
            batch_size=batch_size,
            verbose=1
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        
        # Try again with smaller batches if there was an error
        print("Retrying with smaller batch size...")
        new_batch_size = max(1, batch_size // 2)
        
        try:
            # Recreate dataset with smaller batch size
            dataset_csi_complete = create_dataset_single(
                file_complete_expanded, 
                labels_complete,
                stream_ant_complete, 
                input_shape, 
                new_batch_size,
                shuffle=False, 
                cache_file=None,  # Disable caching for retry
                prefetch=args.prefetch_buffer
            )
            
            # Recalculate steps
            complete_steps_per_epoch = int(np.ceil(num_samples_complete / new_batch_size))
            
            # Try prediction again
            complete_prediction_list = csi_model.predict(
                dataset_csi_complete,
                steps=complete_steps_per_epoch,
                batch_size=new_batch_size,
                verbose=1
            )
        except Exception as e2:
            print(f"Final error during prediction: {e2}")
            exit(1)
    
    # Process predictions
    if len(complete_prediction_list) > len(labels_complete):
        # Trim excess predictions
        complete_prediction_list = complete_prediction_list[:len(labels_complete)]
    
    # Get predicted class indices
    complete_labels_pred = np.argmax(complete_prediction_list, axis=1)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels_complete, complete_labels_pred)
    
    # Calculate metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels_complete, complete_labels_pred, zero_division=0)
    accuracy = accuracy_score(labels_complete, complete_labels_pred)
    
    # Print results
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for i, act in enumerate(activities):
        if i in unique_labels:
            print(f"  {act}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={fscore[i]:.4f}")
    
    # Save results to CSV
    results = []
    for i in range(len(activities)):
        if i in unique_labels:
            results.append([activities[i], precision[i], recall[i], fscore[i]])
    
    # Write results to CSV file
    with open(f'test_results_{name_base}.csv', 'w') as f:
        f.write('Activity,Precision,Recall,F1-score\n')
        for act, prec, rec, f1 in results:
            f.write(f'{act},{prec},{rec},{f1}\n')
    
    # Write confusion matrix
    np.savetxt(f'confusion_matrix_{name_base}.txt', conf_matrix, fmt='%d', delimiter=',')
    
    # Print ending message
    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    print(f"Results saved to test_results_{name_base}.csv")
    print(f"Confusion matrix saved to confusion_matrix_{name_base}.txt") 