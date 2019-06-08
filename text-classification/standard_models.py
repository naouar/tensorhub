""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 01:37:20
@Last Modified Time: 2019-05-28 01:37:20
"""

# Load packages
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api import keras


class SimpleRNN(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, num_nodes=[512, 1024, 1024], activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=100, embedding_matrix=None):
        super(SimpleRNN, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.rnn_layer = keras.layers.SimpleRNN(units=num_nodes[0], activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=num_nodes[1], activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=num_nodes[2], activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleLSTM(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, num_nodes=[512, 1024, 1024], activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=100, embedding_matrix=None):
        super(SimpleLSTM, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.lstm_layer = keras.layers.LSTM(units=num_nodes[0], activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=num_nodes[1], activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=num_nodes[2], activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.lstm_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleGRU(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, num_nodes=[512, 1024, 1024], activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=100, embedding_matrix=None):
        super(SimpleGRU, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.gru_layer = keras.layers.GRU(units=num_nodes[0], activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=num_nodes[1], activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=num_nodes[2], activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.gru_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class TextCNN(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, filters=[128, 128], kernals=[5, 5], strides=[2, 2], drop_rate=0.4, activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=100, embedding_matrix=None):
        super(TextCNN, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.reshape_layer = keras.layers.Reshape((max_length, embed_dim, 1))
        self.conv_layer1 = keras.layers.Conv2D(filters=filters[0], kernel_size=(kernals[0]), activation=activation)
        self.pool_layer1 = keras.layers.MaxPool2D(pool_size=max_length-filters[0]+1, strides=(strides[0]))
        self.conv_layer2 = keras.layers.Conv2D(filters=filters[1], kernel_size=(kernals[1]), activation=activation)
        self.pool_layer2 = keras.layers.MaxPool2D(pool_size=max_length-filters[1]+1, strides=(strides[1]))
        self.flatten_layer = keras.layers.Flatten()
        self.dropout_layer = keras.layers.Dropout(rate=drop_rate)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.reshape_layer(x)
        x = self.conv_layer1(x)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.flatten_layer(x)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x