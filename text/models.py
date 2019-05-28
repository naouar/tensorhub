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
    def __init__(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix):
        print("Initiated SimpleRNN Model.")
        super(SimpleRNN, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, input_length=max_length)
        self.layer1 = keras.layers.SimpleRNN(units=num_nodes, return_sequences=True)
        self.layer2 = keras.layers.SimpleRNN(units=num_nodes)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class SimpleLSTM(keras.Model):
    def __init__(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix):
        print("Initiated SimpleLSTM Model.")
        super(SimpleLSTM, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, input_length=max_length)
        self.lstm_layer1 = keras.layers.LSTM(units=num_nodes, return_sequences=True)
        self.lstm_layer2 = keras.layers.LSTM(units=num_nodes)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.lstm_layer1(x)
        x = self.lstm_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleGRU(keras.Model):
    def __init__(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix):
        print("Initiated SimpleGRU Model.")
        super(SimpleGRU, self).__init__()
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, weights=[embedding_matrix], trainable=False, input_length=max_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=num_nodes, input_length=max_length)
        self.layer1 = keras.layers.GRU(units=num_nodes, return_sequences=True)
        self.layer2 = keras.layers.GRU(units=num_nodes)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=activation)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x
