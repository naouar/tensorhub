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


class MLP(tf.keras.Model):
    def __init__(self, width, num_class, out_act):
        print("Initiated Multi-Layer Perceptron Model.")
        super(MLP, self).__init__()
        self.layer1 = tf.keras.layers.Dense(units=width, activation="relu")
        self.layer2 = tf.keras.layers.Dense(units=width, activation="relu")
        self.output_layer = tf.keras.layers.Dense(units=num_class, activation=out_act)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class SimpleRNN(tf.keras.Model):
    def __init__(self, max_features, width, num_class, out_act):
        print("Initiated SimpleRNN Model.")
        super(SimpleRNN, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(max_features, output_dim=2*width)
        self.layer1 = tf.keras.layers.SimpleRNN(units=width, return_sequences=True)
        self.layer2 = tf.keras.layers.SimpleRNN(units=width)
        self.output_layer = tf.keras.layers.Dense(units=num_class, activation=out_act)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class SimpleLSTM(tf.keras.Model):
    def __init__(self, max_features, width, num_class, out_act):
        print("Initiated SimpleLSTM Model.")
        super(SimpleLSTM, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(max_features, output_dim=2*width)
        self.lstm_layer1 = tf.keras.layers.LSTM(units=width, return_sequences=True)
        self.lstm_layer2 = tf.keras.layers.LSTM(units=width)
        self.output_layer = tf.keras.layers.Dense(units=num_class, activation=out_act)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.lstm_layer1(x)
        x = self.lstm_layer2(x)
        x = self.output_layer(x)
        return x


class SimpleGRU(tf.keras.Model):
    def __init__(self, max_features, width, num_class, out_act):
        print("Initiated SimpleGRU Model.")
        super(SimpleGRU, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(max_features, output_dim=2*width)
        self.layer1 = tf.keras.layers.GRU(units=width, return_sequences=True)
        self.layer2 = tf.keras.layers.GRU(units=width)
        self.output_layer = tf.keras.layers.Dense(units=num_class, activation=out_act)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x
