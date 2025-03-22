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
import keras.backend as K

# GPU memory and device placement fixes
import os
import tensorflow as tf

# Set environment variables to help with device placement
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # Disable XLA auto-clustering

# Configure GPU properly
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configure memory growth to avoid taking all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
        
        # Use only the first GPU if multiple are available
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        
        # Enable soft device placement to automatically place ops on the right device
        tf.config.set_soft_device_placement(True)
        print("GPU configuration complete")
    except Exception as e:
        print(f"GPU configuration error: {e}")

physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)

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
    # Convert commas to underscores for filename construction
    activity_str = csi_act.replace(',', '_')
    
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)

    name_base = args.name_base
    # Remove all cache files using glob pattern instead of hardcoded filenames
    for f in glob.glob(f"{name_base}_{activity_str}_cache_train*"):
        if os.path.exists(f):
            os.remove(f)
    for f in glob.glob(f"{name_base}_{activity_str}_cache_val*"):
        if os.path.exists(f):
            os.remove(f)
    for f in glob.glob(f"{name_base}_{activity_str}_cache_train_test*"):
        if os.path.exists(f):
            os.remove(f)
    for f in glob.glob(f"{name_base}_{activity_str}_cache_test*"):
        if os.path.exists(f):
            os.remove(f)
    # Also remove any lockfiles that might be present
    for f in glob.glob(f"{name_base}_{activity_str}_cache_*.lockfile"):
        if os.path.exists(f):
            os.remove(f)

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
        dir_train = args.dir + sdir + '/train_antennas_' + activity_str + '/'
        name_labels = args.dir + sdir + '/labels_train_antennas_' + activity_str + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_train.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_train_antennas_' + activity_str + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_train.extend(pickle.load(fp))

        dir_val = args.dir + sdir + '/val_antennas_' + activity_str + '/'
        name_labels = args.dir + sdir + '/labels_val_antennas_' + activity_str + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_val.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_val_antennas_' + activity_str + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_val.extend(pickle.load(fp))

        dir_test = args.dir + sdir + '/test_antennas_' + activity_str + '/'
        name_labels = args.dir + sdir + '/labels_test_antennas_' + activity_str + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_test.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_test_antennas_' + activity_str + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_test.extend(pickle.load(fp))

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                           labels_considered]
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                             labels_considered]

    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas)

    name_cache = name_base + '_' + activity_str + '_cache_train'
    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                              stream_ant_train, input_network, batch_size,
                                              shuffle=True, cache_file=name_cache)

    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)

    name_cache_val = name_base + '_' + activity_str + '_cache_val'
    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                            stream_ant_val, input_network, batch_size,
                                            shuffle=False, cache_file=name_cache_val)

    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                          labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)

    name_cache_test = name_base + '_' + activity_str + '_cache_test'
    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                             stream_ant_test, input_network, batch_size,
                                             shuffle=False, cache_file=name_cache_test)
                                             
    # Create model variables
    # This fixes the device placement issue by ensuring all variables are pinned to GPU
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    with mirrored_strategy.scope():
        # Create the model and compile it within the strategy scope
        # This ensures all variables are created on the same device
        csi_model = csi_network_inc_res(input_network, output_shape)
        
        # Configure optimizer and loss
        optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
        
        # Compile model
        csi_model.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        
    # Display model summary
    csi_model.summary()

    num_samples_train = len(file_train_selected_expanded)
    num_samples_val = len(file_val_selected_expanded)
    num_samples_test = len(file_test_selected_expanded)
    lab, count = np.unique(labels_train_selected_expanded, return_counts=True)
    lab_val, count_val = np.unique(labels_val_selected_expanded, return_counts=True)
    lab_test, count_test = np.unique(labels_test_selected_expanded, return_counts=True)
    train_steps_per_epoch = int(np.ceil(num_samples_train/batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val/batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test/batch_size))

    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    name_model_h5 = name_base + '_' + activity_str + '_network.keras'  # For HDF5
    name_model_tf = name_base + '_' + activity_str + '_network'     # For SavedModel (directory)
    callback_save = tf.keras.callbacks.ModelCheckpoint(
        name_model_h5, 
        save_freq='epoch', 
        save_best_only=True,
        monitor='val_sparse_categorical_accuracy'
    )

    # Verify input shape matches model's expected input shape
    sample_batch = next(iter(dataset_csi_train))
    print(f"Actual input shape: {sample_batch[0].shape}")  # Should show batch size and input dimensions
    print(f"Model expected shape: {csi_model.input_shape}")  # Should match except for batch dimension

    # Train model with GPU acceleration
    # Since we created it with MirroredStrategy, the variables are properly located on GPU
    results = csi_model.fit(dataset_csi_train, epochs=25, steps_per_epoch=train_steps_per_epoch,
                          validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                          callbacks=[callback_save, callback_stop])

    # Save model in Keras format (compatible with Keras 3)
    csi_model.save(name_model_h5)
    
    # Clear TensorFlow session to free memory
    K.clear_session()
    
    # Load the saved model with the same strategy to ensure consistent device placement
    with mirrored_strategy.scope():
        csi_model = tf.keras.models.load_model(name_model_h5)

    # Make predictions on training set
    train_labels_true = np.array(labels_train_selected_expanded)
    name_cache_train_test = name_base + '_' + activity_str + '_cache_train_test'
    dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                 stream_ant_train, input_network, batch_size,
                                                 shuffle=False, cache_file=name_cache_train_test, prefetch=False)
    
    train_prediction_list = csi_model.predict(dataset_csi_train_test, 
                                            steps=train_steps_per_epoch)[:train_labels_true.shape[0]]
    train_labels_pred = np.argmax(train_prediction_list, axis=1)
    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

    # Make predictions on validation set
    val_labels_true = np.array(labels_val_selected_expanded)
    val_prediction_list = csi_model.predict(dataset_csi_val, 
                                          steps=val_steps_per_epoch)[:val_labels_true.shape[0]]
    val_labels_pred = np.argmax(val_prediction_list, axis=1)
    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

    # Make predictions on test set
    test_labels_true = np.array(labels_test_selected_expanded)
    test_prediction_list = csi_model.predict(dataset_csi_test, 
                                          steps=test_steps_per_epoch)[:test_labels_true.shape[0]]

    test_labels_pred = np.argmax(test_prediction_list, axis=1)

    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                   test_labels_pred,
                                                                   labels=labels_considered, zero_division=0)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)

    # merge antennas test
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    for i_lab in range(len(labels_test_selected)):
        pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
        lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

        pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
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

    name_file = './outputs/test_' + activity_str + '_' + subdirs_training + '_band_' + str(bandwidth) + '_subband_' + \
                str(sub_band) + suffix
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

    name_file = './outputs/change_number_antennas_test_' + activity_str + '_' + subdirs_training + '_band_' + \
                str(bandwidth) + '_subband_' + str(sub_band) + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)