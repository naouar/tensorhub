"""
@Author: Kumar Nityan Suman
@Date: 2019-06-20 17:11:03
"""


# Load packages
import tensorflow as tf
from tensorflow import keras


class Encoder(keras.Model):
    """Basic Encoder Implementation."""

    def __init__(self, src_vocab_size, batch_sz, name="gru", embedding_dim=300, learn_embedding=True, embedding_matrix=None, enc_units=128):
        """Initialize encoder architecture.
        
        Arguments:
            tar_vocab_size {int} -- Size of the target sequence vocabulary size.
            batch_sz {int} -- Batch size.
        
        Keyword Arguments:
            name {str} -- Name of the recurrent layer. Choices : ['lstm', 'gru'] (default: {"gru"})
            embedding_dim {int} -- Size of the token embedding. (default: {300})
            learn_embedding {bool} -- Boolean flag as indicator for learning embedding or using a pre-trained done. (default: {True})
            embedding_matrix {numpy} -- If using pre-trained embedding. Load pre-trained embedding here. (default: {None})
            enc_units {int} -- Number of decoder nodes. (default: {128})
        
        Raises:
            ValueError: Raise error when wrong model name is passed.
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if learn_embedding == True:
            # Learn embedding as a part of the network
            self.embedding = keras.layers.Embedding(input_dim=src_vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True)
        else:
            # Use pre-trained embeddings like glove
            self.embedding = keras.layers.Embedding(input_dim=src_vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False, input_length=max_length)
        # Encoder
        if name == "rnn":
            self.encoder_layer = keras.layers.RNN(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "lstm":
            self.encoder_layer = keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        elif name == "gru":
            self.encoder_layer = keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Wrong encoder type passed! {}".format(name))

    def call(self, x, hidden):
        """Forward pass over network.
        
        Arguments:
            x {tensor} -- Input sequence tensor.
            hidden {tensor} -- Hidden state of the encoder.
        
        Returns:
            tensor, tensor -- Returns output and hidden state tensor from the encoder.
        """
        x = self.embedding(x)
        output, hidden_state = self.encoder_layer(x, initial_state=hidden)
        return output, hidden_state

    def initialize_hidden_state(self):
        """Initialize encoder hidden state.
        
        Returns:
            tensor -- Returns initial hidden state of the encoder.
        """
        return tf.zeros((self.batch_sz, self.enc_units))