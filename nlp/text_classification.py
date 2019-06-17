""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 01:37:20
"""

# Load packages
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

"""Text Classification Model Implementations."""

class SimpleTextClassification(keras.Model):
    """Simple text classification model implementation using recurrent neural networks."""
    def __init__(self, model_name, vocab_size, num_classes, max_length=512, num_nodes=[512, 1024, 1024], activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=300, embedding_matrix=None):
        """Text classification using some basic standard recurrent neural network architectures.
        
        Arguments:
            model_name {str} -- Name of model to use. Choices = [(bi)'rnn', (bi)'lstm', (bi)'gru']
            vocab_size {int} -- Integer denoting number of top words to consider for the vocab.
            num_classes {int} -- Number of classes.
        
        Keyword Arguments:
            max_length {int} -- Maximum length of the input sequence length. (default: {512})
            num_nodes {list} -- Number of nodes in each layer respectively. (default: {[512, 1024, 1024]})
            activation {str} -- Activation to be used. (default: {"relu"})
            output_activation {str} -- Output layer activation to use. (default: {"softmax"})
            learn_embedding {bool} -- Flag denoting to learn embedding or use a pre-trained one. (default: {True})
            embed_dim {int} -- Embedding dimension. (default: {300})
            embedding_matrix {numpy} -- Pre-trained embedding loaded into a numpy matrix if 'learn_embedding' == False. (default: {None})
        """
        super(SimpleTextClassification, self).__init__()
        # Embedding layer
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        # Recurrent layer
        if model_name == "rnn":
            self.main_logic_layer = keras.layers.SimpleRNN(units=num_nodes[0], activation=activation)
        elif model_name == "lstm":
            self.main_logic_layer = keras.layers.LSTM(units=num_nodes[0], activation=activation)
        elif model_name == "gru":
            self.main_logic_layer = keras.layers.GRU(units=num_nodes[0], activation=activation)
        elif model_name == "birnn":
            self.main_logic_layer = keras.layers.Bidirectional(keras.layers.RNN(units=num_nodes[0], activation=activation))
        elif model_name == "bilstm":
            self.main_logic_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=num_nodes[0], activation=activation))
        elif model_name == "bigru":
            self.main_logic_layer = keras.layers.Bidirectional(keras.layers.GRU(units=num_nodes[0], activation=activation))
        # Dense layer
        self.dense_layer1 = keras.layers.Dense(units=num_nodes[1], activation=activation)
        self.dense_layer2 = keras.layers.Dense(units=num_nodes[2], activation=activation)
        # Final output layer with softmax/sigmoid
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    def call(self, x):
        """Forward pass to the network.
        
        Arguments:
            x {Tensor} -- Input tensor in the forward pass.
        
        Returns:
            numpy -- Output tensor.
        """
        x = self.embedding_layer(x)
        x = self.main_logic_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class TextCNN(keras.Model):
    """Text classification using 2D convolutional neural networks."""
    def __init__(self, vocab_size, num_classes, max_length=512, filters=[128, 128], kernals=[5, 5], strides=[2, 2], drop_rate=0.4, activation="relu", output_activation="softmax", learn_embedding=True, embed_dim=300, embedding_matrix=None):
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