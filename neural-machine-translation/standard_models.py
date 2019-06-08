""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-09 00:40:43
@Last Modified Time: 2019-06-09 00:40:43
"""

# Load packages
import os
import sys
import tensorflow as tf
from tensorflow.python.keras.api import keras


class SimpleEncoderDecoder(keras.Model):
    def __init__(self, src_vocab_size, tar_vocab_size, encoder="lstm", decoder="lstm", src_max_seq=512, tar_max_seq=512, num_nodes=[512, 512], embed_dim=300, learn_embedding=True, embedding_matrix=None):
        # Embedding layer
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(src_vocab_size, embed_dim, input_length=src_max_seq, mask_zero=True)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=src_vocab_size, output_dim=embed_dim, weights=[embedding_matrix], trainable=False, input_length=src_max_length)
        # Set encoder
        if encoder == "lstm":
            self.encoder = keras.layers.LSTM(num_nodes[0], recurrent_initializer="glorot_uniform", recurrent_activation="sigmoid")
        elif encoder == "gru":
            self.encoder = keras.layers.GRU(num_nodes[0], recurrent_initializer="glorot_uniform", recurrent_activation="sigmoid")
        else:
            raise ValueError("Wrong encoder selected -> {}".format(encoder))
        # Set decoder
        if decoder == "lstm":
            self.decoder = keras.layers.LSTM(num_nodes[0], recurrent_initializer="glorot_uniform", recurrent_activation="sigmoid", return_sequences=True)
        elif decoder == "gru":
            self.decoder = keras.layers.GRU(num_nodes[0], recurrent_initializer="glorot_uniform", recurrent_activation="sigmoid", return_sequences=True)
        else:
            raise ValueError("Wrong decoder selected -> {}".format(decoder))
        self.repeat_vector = keras.layers.RepeatVector(tar_max_seq)
        self.output_layer = keras.layers.TimeDistributed(keras.layers.Dense(tar_vocab_size, activation="softmax"))
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.encoder(x)
        x = self.repeat_vector(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x