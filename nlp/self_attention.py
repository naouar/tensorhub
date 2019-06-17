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
from utils import PositionEmbedding

"""Multi-Head Self-Attention Implementation as a Layer."""

class PositionEmbedding(keras.layers.Layer):
    """Compute position embedding for attention layer.
    
    Returns:
        NumpyArray -- Position embdding.
    """
    def __init__(self, mode="sum", size=None):
        """Class constructor.
        
        Keyword Arguments:
            mode {str} -- Type of position embedding. (default: {"sum"})
            size {[type]} -- Size of the position embedding. (default: {None})
        """
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__()
        
    def call(self, x):
        """Forward pass for positional embeddings.
        
        Arguments:
            x {Tensor} -- Input tensor to create position embedding.
        
        Returns:
            Tensor -- Created position embedding for the inputt tensor.
        """
        # Update position embedding size
        if self.size == None or self.mode == "sum":
            self.size = int(x.shape[-1])
        batch_size, seq_len = np.shape(x)[0], np.shape(x)[1]
        # Compute position j
        position_j = 1. / np.pow(10000., 2 * np.arange(self.size / 2, dtype="float32") / self.size)
        position_j = np.expand_dims(position_j, 0)
        # Compute position i
        position_i = np.cumsum(np.ones_like(x[:,:,0]), 1) -1
        position_i = np.expand_dims(position_i, 2)
        # Compute relative position
        position_ij = np.dot(position_i, position_j)
        position_ij = np.concatenate([np.cos(position_ij), np.sin(position_ij)], 2)
        # Update embedding based on modes
        if self.mode == "sum":
            return position_ij + x
        elif self.mode == "concat":
            return np.concatenate([position_ij, x], 2)
    
    def compute_output_shape(self, input_shape):
        """COmpute output shape of the tensor using the input shape.
        
        Arguments:
            input_shape {Tensor} -- Shape tensor.
        
        Returns:
            Tensor -- Output shape tensor.
        """
        # Compute output shape
        if self.mode == "sum":
            return input_shape
        elif self.mode == "concat":
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(keras.layers):
    def __init__(self, num_head, head_size):
        """Constructor to initialize input independent variables.
        
        Arguments:
            num_head {Int} -- Number of attention heads.
            head_size {[type]} -- Size of each attention head.
        """
        self.num_head = num_head
        self.head_size = head_size
        self.output_dim = self.num_head * self.head_size
        super(Attention, self).__init__()
    
    def build(self, input_shape):
        """Initialize input dependent variables.
        
        Arguments:
            input_shape {Tensor} -- Input shape tensor.
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
        super(Attention, self).build(input_shape)
    
    def mask(self, inputs, seq_len, mode="sum"):
        """Mask input tensor.
        
        Arguments:
            inputs {Tensor} -- Input tensor.
            seq_len {Int} -- Input sequence length.
        
        Keyword Arguments:
            mode {String} -- Mode of masking. (default: {"sum"})
        
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
            x {Tensor} -- Input Tensor.
        
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
            input_shape {Tensor} -- Input shape tensor.
        
        Returns:
            Tensor -- Output shape tensor.
        """
        return (input_shape[0][0], input_shape[0][1], self.output_dim)