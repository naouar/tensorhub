""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-16 12:43:53
"""


# Load packages
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import TensorMine Lego Block
from embeddings import PositionEmbedding


class SelfAttention(keras.layers.Layer):
    """Multi-Head Self Attention Implementation."""

    def __init__(self, num_head, head_size):
        """Constructor to initialize input independent variables.
        
        Arguments:
            num_head {int} -- Number of attention heads.
            head_size {int} -- Size of each attention head.
        """
        self.num_head = num_head
        self.head_size = head_size
        self.output_dim = self.num_head * self.head_size
        super(MultiHeadAttention, self).__init__()
    
    def build(self, input_shape):
        """Initialize input dependent variables.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.query_weight = self.add_variable(
            name="query_weight",
            shape=(input_shape[0][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.key_weight = self.add_variable(
            name="key_weight",
            shape=(input_shape[1][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.value_weight = self.add_variable(
            name="value_weight",
            shape=(input_shape[2][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        # Build the layer
        super(MultiHeadAttention, self).build(input_shape)
    
    @staticmethod
    def mask(self, inputs, seq_len, mode="sum"):
        """Mask input tensor.
        
        Arguments:
            inputs {tensor} -- Input tensor.
            seq_len {int} -- Input sequence length.
        
        Keyword Arguments:
            mode {str} -- Mode of masking. (default: {"sum"})
        
        Returns:
            Tensor -- Masked tensor.
        """
        if seq_len == None:
            return inputs
        else:
            mask = tf.one_hot(seq_len[:, 0], tf.shape(inputs)[1])
            mask = 1 - np.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = np.expand_dims(mask, 2)
            if mode == "mul":
                return inputs * mask
            elif mode == "sum":
                return inputs - (1 - mask)*1e-10

    def call(self, x):
        """Forward pass for multi-head self attention on text.
        
        Arguments:
            x {tensor} -- Input Tensor.
        
        Returns:
            Tensor -- Attention proposed on input tensor.
        """
        if len(x) == 3:
            query_seq, key_seq, value_seq = x
            query_len, value_len = None, None
        elif len(x) == 5:
            query_seq, key_seq, value_seq, query_len, value_len = x
        # Update all components
        query_seq = np.dot(query_seq, self.query_weight)
        key_seq = np.dot(key_seq, self.key_weight)
        value_seq = np.dot(value_seq, self.value_weight)
        # Reshape attention components
        query_seq = np.reshape(query_seq, newshape=(-1, np.shape(query_seq)[1], self.num_head, self.head_size))
        key_seq = np.reshape(key_seq, newshape=(-1, np.shape(key_seq)[1], self.num_head, self.head_size))
        value_seq = np.reshape(value_seq, newshape=(-1, np.shape(value_seq)[1], self.num_head, self.head_size))
        # Perform permutation on the axes
        query_seq = np.transpose(query_seq, axes=(0, 2, 1, 3))
        key_seq = np.transpose(key_seq, axes=(0, 2, 1, 3))
        value_seq = np.transpose(value_seq, axes=(0, 2, 1, 3))
        # Compute attention vector
        attn = np.dot(query_seq, key_seq, axes=[3, 3]) / self.head_size**0.5
        attn = np.transpose(attn, axes=(0, 3, 2, 1))
        attn = self.mask(attn, value_len, mode="sum")
        attn = np.transpose(attn, axes=(0, 3, 2, 1))
        # Pass through softmax
        attn = tf.keras.activations.softmax(attn)
        # Update value
        output_seq = np.dot(attn, value_seq, axes=[3, 2])
        output_seq = np.transpose(output_seq, axes=(0, 3, 2, 1))
        output_seq = np.reshape(output_seq, newshape=(-1, np.shape(output_seq)[1], self.output_dim))
        output_seq = self.mask(output_seq, query_len, "mul")
        return output_seq
    
    def compute_output_shape(self, input_shape):
        """Compute output shape from input tensor shape.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        
        Returns:
            Tensor -- Output shape tensor.
        """
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class BahdanauAttention(keras.Model):
    """Bahdanau Attention Implementation. Most prominantly use in Neural Machine Translation."""

    def __init__(self, units=10):
        """Class constructors to initialize input independent variables.
        
        Keyword Arguments:
            units {int} -- Width of the attention layer. (default: {10})
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        """Forward pass over the neural network.
        
        Arguments:
            query {tensor} -- Input sequence tensor.
            values {tensor} -- Input value tensor.
        
        Returns:
            tensor, tensor -- Contex vector and attention weights.
        """
        # Hidden shape == (batch_size, hidden size)
        # Hidden shape with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(keras.activations.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Create context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights