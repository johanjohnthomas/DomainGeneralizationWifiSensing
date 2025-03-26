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

import tensorflow as tf


def conv2d_bn(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x


def reduction_a_block_small(x_in, base_name):
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)

    x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')

    x3 = conv2d_bn(x_in, 3, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, 6, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, 9, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')

    x4 = tf.keras.layers.Concatenate()([x1, x2, x3])
    return x4


def csi_network_inc_res(input_sh, output_sh):
    x_input = tf.keras.Input(input_sh)

    x2 = reduction_a_block_small(x_input, base_name='1st')

    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')

    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x)
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
    return model


def csi_network_lstm_cnn(input_sh, output_sh):
    """
    LSTM-CNN hybrid architecture for CSI data processing.
    
    This model applies 2D CNN layers to extract spatial features,
    followed by LSTM layers to capture temporal dependencies.
    
    Args:
        input_sh: Input shape tuple (sample_length, feature_length, channels)
        output_sh: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    x_input = tf.keras.Input(input_sh)
    
    # CNN layers for spatial feature extraction
    x = conv2d_bn(x_input, 32, (3, 3), padding='same', activation='relu', bn=True, name='conv1')
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = conv2d_bn(x, 64, (3, 3), padding='same', activation='relu', bn=True, name='conv2')
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = conv2d_bn(x, 128, (3, 3), padding='same', activation='relu', bn=True, name='conv3')
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for LSTM
    # Calculate the new shape after pooling operations
    # After 3 pooling layers of (2,2), dimensions are reduced to 1/8
    # Assuming input_sh is (sample_length, feature_length, channels)
    new_height = input_sh[0] // 8
    new_width = input_sh[1] // 8
    
    # Reshape to (sequence_length, features) where each time step contains flattened features
    # Use the width as the sequence length and height*channels as features
    x = tf.keras.layers.Reshape((new_width, new_height * 128))(x)
    
    # LSTM layers for temporal dependencies
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    
    # Create model
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_lstm_cnn_model')
    
    return model


def csi_network_gru_cnn(input_sh, output_sh):
    """
    GRU-CNN hybrid architecture for CSI data processing based on the PyTorch model.
    
    This model applies 2D CNN layers to extract spatial features,
    followed by GRU layer to capture temporal dependencies.
    
    Args:
        input_sh: Input shape tuple (sample_length, feature_length, channels)
        output_sh: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Print input shape for debugging
    print(f"GRU-CNN model input shape: {input_sh}")
    
    # Input layer
    x_input = tf.keras.Input(input_sh)
    
    # First, we'll treat the sample_length as our time dimension
    # and process each time step through a series of convolutions
    
    # Determine kernel size based on the feature dimensions
    # For small feature dimensions, use a smaller kernel
    kernel_size = (3, 3) if input_sh[1] >= 5 else (1, 1)
    print(f"Using kernel size: {kernel_size}")
    
    # Apply Conv2D directly with the channels
    x = tf.keras.layers.Conv2D(16, kernel_size, padding='same', activation='relu')(x_input)
    print(f"After Conv2D shape: {x.shape}")
    
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    print(f"After MaxPooling2D shape: {x.shape}")
    
    # Flatten the spatial dimensions but keep the time dimension
    # Calculate new dimensions after pooling
    new_height = x.shape[1] if x.shape[1] is not None else input_sh[0] // 2
    new_width = x.shape[2] if x.shape[2] is not None else input_sh[1] // 2
    print(f"Calculated reshape dimensions: ({new_height}, {new_width * 16})")
    
    # Reshape to (batch_size, time_steps, features)
    x = tf.keras.layers.Reshape((new_height, new_width * 16))(x)
    print(f"After Reshape shape: {x.shape}")
    
    # Dense layers applied to each time step
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    print(f"After Dense(64) shape: {x.shape}")
    
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    print(f"After second Dense(64) shape: {x.shape}")
    
    # GRU layer for temporal processing
    x = tf.keras.layers.GRU(128, return_sequences=False)(x)
    print(f"After GRU shape: {x.shape}")
    
    # Final classification
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    print(f"Final output shape: {x.shape}")
    
    # Create model
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_gru_cnn_model')
    
    return model


def csi_network_pytorch_style(input_sh, output_sh):
    """
    PyTorch-style CNN-GRU model that closely follows the PyTorch implementation.
    
    This model first reshapes the input to add a channel dimension,
    processes the data through a CNN pathway, and then feeds it to a GRU.
    
    Args:
        input_sh: Input shape tuple (sample_length, feature_length, channels)
        output_sh: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    print(f"PyTorch-style model input shape: {input_sh}")
    
    # Input layer
    x_input = tf.keras.Input(input_sh)
    
    # Reshape to treat the last dimension as channels and add a new channel dimension
    # Equivalent to PyTorch input format (batch, time, height, width)
    x = tf.keras.layers.Reshape((input_sh[0], input_sh[1], 1))(x_input)
    print(f"After initial reshape: {x.shape}")
    
    # CNN Feature Extraction (equivalent to l_ex in PyTorch model)
    # Determine kernel size based on the feature dimensions
    kernel_size = (3, 3) if input_sh[1] >= 5 else (1, 1)
    print(f"Using kernel size: {kernel_size}")
    
    x = tf.keras.layers.Conv2D(16, kernel_size, padding='same')(x)
    print(f"After Conv2D shape: {x.shape}")
    
    # Use same padding in MaxPooling to ensure we don't lose too much spatial information
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    print(f"After MaxPooling2D shape: {x.shape}")
    
    x = tf.keras.layers.Flatten()(x)
    print(f"After Flatten shape: {x.shape}")
    
    x = tf.keras.layers.Dense(64)(x)
    print(f"After Dense(64) shape: {x.shape}")
    
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.ReLU()(x)
    print(f"After second Dense(64) + ReLU shape: {x.shape}")
    
    # Reshape for GRU - we need to add a time dimension
    # In PyTorch model, this is handled differently because PyTorch processes batches differently
    # Here we add a time dimension of 1 (single timestep)
    x = tf.keras.layers.Reshape((1, 64))(x)
    print(f"After reshape for GRU: {x.shape}")
    
    # GRU layer
    x = tf.keras.layers.GRU(128)(x)
    print(f"After GRU shape: {x.shape}")
    
    # Final classification
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None)(x)
    print(f"Final output shape: {x.shape}")
    
    # Create model
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_pytorch_style_model')
    
    return model
