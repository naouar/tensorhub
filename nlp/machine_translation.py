""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-09 00:40:43
"""

# Load packages
import os
import sys
import tensorflow as tf
from tensorflow import keras
from attention import BahdanauAttention

"""Neural Machine Translation Model Implementations."""

class Encoder(keras.Model):
    def __init__(self, src_vocab_size, name="gru", embedding_dim=300, enc_units=128):
        super(Encoder, self).__init__()
        self.embedding = keras.layers.Embedding(src_vocab_size, embedding_dim)
        if name == "lstm":
            self.encoder_layer = keras.layers.LSTM(enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.encoder_layer = keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(name))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.encoder_layer(x, initial_state=hidden)
        return output, state


class Decoder(keras.Model):
    def __init__(self, tar_vocab_size, name="gru", embedding_dim=300, dec_units=128):
        super(Decoder, self).__init__()
        self.embedding = keras.layers.Embedding(tar_vocab_size, embedding_dim)
        if name == "lstm":
            self.decoder_layer = keras.layers.LSTM(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.decoder_layer = keras.layers.GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(name))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.decoder_layer(x, initial_state=hidden)
        return output, state


class AttentionDecoder(keras.Model):
    def __init__(self, tar_vocab_size, name="gru", embedding_dim=300, dec_units=128, batch_sz=32):
        super(Decoder, self).__init__()
        # Embedding layer
        self.embedding = keras.layers.Embedding(tar_vocab_size, embedding_dim)
        # Encoder layer
        if name == "lstm":
            self.decoder_layer = keras.layers.LSTM(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.decoder_layer = keras.layers.GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(encoder))
        # Fully connect layer
        self.fc = keras.layers.Dense(tar_vocab_size)
        # Used for attention
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        # Attention on encoder output
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # Embedding layer
        x = self.embedding(x)

        # Residual with attention and target sequence
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passing the concatenated vector to the decoder
        output, state = self.decoder_layer(x)

        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through fully connected layer
        x = self.fc(output)
        return x, state, attention_weights
