"""
@Author: Kumar Nityan Suman
@Date: 2019-06-16 12:43:53
"""


# Load packages
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
        super(SelfAttention, self).__init__()
    
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
        attn = mask(inputs=attn, seq_len=value_len, mode="sum")
        attn = np.transpose(attn, axes=(0, 3, 2, 1))
        # Pass through softmax
        attn = tf.keras.activations.softmax(attn)
        # Update value
        output_seq = np.dot(attn, value_seq, axes=[3, 2])
        output_seq = np.transpose(output_seq, axes=(0, 3, 2, 1))
        output_seq = np.reshape(output_seq, newshape=(-1, np.shape(output_seq)[1], self.output_dim))
        output_seq = mask(inputs=output_seq, seq_len=query_len, mode="mul")
        return output_seq
    
    def compute_output_shape(self, input_shape):
        """Compute output shape from input tensor shape.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        
        Returns:
            Tensor -- Output shape tensor.
        """
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class BahdanauAttention(keras.layers.Layer):
    """Bahdanau Attention Implementation. Most prominantly use in Neural Machine Translation."""

    def __init__(self, num_output):
        """Class constructors to initialize input independent variables."""
        super(BahdanauAttention, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        """Initialize input dependent variables.
        
        Arguments:
            input_shape {tensor} -- Input tensor shape.
        """
        self.W1 = self.add_variable("weight1", shape=(int(input_shape[-1]), self.num_outputs)))
        self.W2 = self.add_variable("weight1", shape=(int(input_shape[-1], self.num_outputs))
        self.V = self.add_variable("weight1", shape=(1, int(input_shape[-1])))

    def call(self, query, value):
        """Forward pass over the neural network.
        
        Arguments:
            query {tensor} -- Encoder hidden state output.
            value {tensor} -- Decoder input sequence.
        
        Returns:
            tensor -- Contex vector.
        """
        hidden_state = tf.expand_dims(query, 1)

        score = self.V(keras.activations.tanh(tf.matmul(self.W1, value) + tf.matmul(self.W2, hidden_state)))

        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Create context vector
        context_vector = tf.reduce_sum(attention_weights * value, axis=1)
        return context_vector