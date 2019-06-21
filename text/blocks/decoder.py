"""
@Author: Kumar Nityan Suman
@Date: 2019-06-20 17:09:34
"""


# Load packages
import os
import sys
import tensorflow as tf
from tensorflow import keras

# Import TensorMine Lego Block
from attention import BahdanauAttention


class Decoder(keras.Model):
    """Standard Decoder Implementation."""

    def __init__(self, tar_vocab_size, batch_sz, name="gru", embedding_dim=300, learn_embedding=True, embedding_matrix=None, dec_units=128):
        """Initialize decoder architecture.
        
        Arguments:
            tar_vocab_size {int} -- Size of the target sequence vocabulary size.
        
        Keyword Arguments:
            name {str} -- Name of the recurrent layer. Choices : ['lstm', 'gru'] (default: {"gru"})
            embedding_dim {int} -- Size of the token embedding. (default: {300})
            learn_embedding {bool} -- Boolean flag as indicator for learning embedding or using a pre-trained done. (default: {True})
            embedding_matrix {numpy} -- If using pre-trained embedding. Load pre-trained embedding here. (default: {None})
            dec_units {int} -- Number of decoder nodes. (default: {128})
        
        Raises:
            ValueError: Raise error when wrong model name is passed.
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        if learn_embedding == True:
            # Learn embedding as a part of the network
            self.embedding = keras.layers.Embedding(input_dim=tar_vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True)
        else:
            # Use pre-trained embeddings like glove
            self.embedding = keras.layers.Embedding(input_dim=tar_vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        # Decoder
        if name == "rnn":
            self.decoder_layer = keras.layers.RNN(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "lstm":
            self.decoder_layer = keras.layers.LSTM(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.decoder_layer = keras.layers.GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(name))

    def call(self, x, hidden, enc_output):
        """Forward pass over the network.
        
        Arguments:
            x {tensor} -- Input sequence to the decoder.
            hidden {tensor} -- Hidden state from the encoder block.
            enc_output {tensor} -- Encoder block output over the source sequence.
        
        Returns:
            tensor, tensor -- Returns decoded output from the decoder and hidden state of the decoder.
        """
        x = self.embedding(x)
        output, state = self.decoder_layer(x, initial_state=hidden)
        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))
        # Pass through fully connected layer
        x = self.fc(output)
        return x, state
    
    def initialize_hidden_state(self):
        """Initialize hidden state for the decoder.
        
        Returns:
            tensor -- Returns initial hidden state for the decoder.
        """
        return tf.zeros((self.batch_sz, self.dec_units))


class AttentionDecoder(keras.Model):
    """Standard Attention based Decoder Implementation."""

    def __init__(self, tar_vocab_size, batch_sz, name="gru", embedding_dim=300, learn_embedding=True, embedding_matrix=None, dec_units=128):
        """Initialize attention based decoder architecture.
        
        Arguments:
            tar_vocab_size {int} -- Size of the target sequence vocabulary size.
        
        Keyword Arguments:
            name {str} -- Name of the recurrent layer. Choices : ['lstm', 'gru'] (default: {"gru"})
            embedding_dim {int} -- Size of the token embedding. (default: {300})
            learn_embedding {bool} -- Boolean flag as indicator for learning embedding or using a pre-trained done. (default: {True})
            embedding_matrix {numpy} -- If using pre-trained embedding. Load pre-trained embedding here. (default: {None})
            dec_units {int} -- Number of decoder nodes. (default: {128})
        
        Raises:
            ValueError: Raise error when wrong model name is passed.
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        if learn_embedding == True:
            # Learn embedding as a part of the network
            self.embedding = keras.layers.Embedding(input_dim=tar_vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True)
        else:
            # Use pre-trained embeddings like glove
            self.embedding = keras.layers.Embedding(input_dim=tar_vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        # Decoder
        if name == "lstm":
            self.decoder_layer = keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.decoder_layer = keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(encoder))
        # Dense layer
        self.fc = keras.layers.Dense(tar_vocab_size)
        # Attention layer
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        """Forward pass over the network.
        
        Arguments:
            x {tensor} -- Input sequence to the decoder.
            hidden {tensor} -- Hidden state from the encoder block.
            enc_output {tensor} -- Encoder block output over the source sequence.
        
        Returns:
            tensor, tensor, tensor -- Returns decoded output from the decoder, hidden state of the decoder, attention weights.
        """
        # Attention on encoder output
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # Get embeddings
        x = self.embedding(x)
        # Residual connection between attention and target sequence
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # Passing the concatenated vector to the decoder
        output, state = self.decoder_layer(x)
        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))
        # Pass through fully connected layer
        x = self.fc(output)
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        """Initialize hidden state for the decoder.
        
        Returns:
            tensor -- Returns initial hidden state for the decoder.
        """
        return tf.zeros((self.batch_sz, self.dec_units))