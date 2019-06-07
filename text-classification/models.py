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
        print("Initiated SimpleRNN Model.")
        super(SimpleRNN, self).__init__()
        if learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        self.rnn_layer = keras.layers.SimpleRNN(units=512, return_sequences=True, activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
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
        print("Initiated SimpleLSTM Model.")
        super(SimpleLSTM, self).__init__()
        if learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        self.lstm_layer = keras.layers.LSTM(units=512, return_sequences=True, activation=activation)
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
        print("Initiated SimpleGRU Model.")
        super(SimpleGRU, self).__init__()
        if learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_length, mask_zero=True)
        self.gru_layer = keras.layers.GRU(units=512, return_sequences=True, activation=activation)
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
    def __init__(self, vocab_size, num_classes, max_length, num_nodes, num_filter, kernal_size, stride, dropout_rate, activation, output_activation, learn_embedding, embedding_matrix):
        print("Initiated TextCNN Model.")
        super(TextCNN, self).__init__()
        if learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, input_length=max_length, mask_zero=True)
        self.spatial_dropout_layer = keras.layers.SpatialDropout1D(rate=0.2)
        self.cnn_layer = keras.layers.Conv1D(filters=num_filter, kernel_size=(kernal_size), padding="valid", activation=activation, strides=stride)
        self.pool_layer = tf.keras.layers.GlobalMaxPooling1D()
        self.dense_layer1 = keras.layers.Dense(units=1024, activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=1024, activation=activation)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.spatial_dropout_layer(x)
        x = self.cnn_layer(x)
        x = self.pool_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x
