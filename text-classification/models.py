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
    def __init__(self, vocab_size, num_classes, max_length=512, activation="relu", output_activation="softmax", learn_embedding=True, embedding_matrix=None):
        super(SimpleRNN, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.rnn_layer = keras.layers.SimpleRNN(units=512, activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=1024, activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleLSTM(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, activation="relu", output_activation="softmax", learn_embedding=True, embedding_matrix=None):
        super(SimpleLSTM, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.lstm_layer = keras.layers.LSTM(units=512, activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=1024, activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.lstm_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleGRU(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, activation="relu", output_activation="softmax", learn_embedding=True, embedding_matrix=None):
        super(SimpleGRU, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.gru_layer = keras.layers.GRU(units=512, activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=1024, activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.gru_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class TextCNN(keras.Model):
    def __init__(self, vocab_size, num_classes, max_length=512, activation="relu", output_activation="softmax", learn_embedding=True, embedding_matrix=None):
        super(TextCNN, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], trainable=False, input_length=max_length)
        self.dropout_layer = keras.layers.Dropout(rate=0.2)
        self.reshape_layer = keras.layers.Reshape((100, max_length))
        self.conv_layer1 = keras.layers.Conv1D(filters=64, kernel_size=(5), activation=activation)
        self.pool_layer1 = keras.layers.GlobalMaxPool1D()
        self.conv_layer2 = keras.layers.Conv1D(filters=128, kernel_size=(5), activation=activation)
        self.pool_layer2 = keras.layers.GlobalMaxPool1D()
        self.conv_layer3 = keras.layers.Conv1D(filters=256, kernel_size=(5), activation=activation)
        self.pool_layer3 = keras.layers.GlobalMaxPool1D()
        self.flat_layer = keras.layers.Flatten()
        self.dense_layer = keras.layers.Dense(units=1024, activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        x = self.reshape_layer(x)
        x = self.conv_layer1(x)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.conv_layer3(x)
        x = self.pool_layer3(x)
        x = self.flat_layer(x)
        x = self.dense_layer(x)
        x = self.output_layer(x)
        return x