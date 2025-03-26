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
