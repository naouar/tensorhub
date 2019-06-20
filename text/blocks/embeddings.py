""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-20 14:22:05
"""


# Load packages
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Embeddings:
    """Handle Embeddings."""
    
    def __init__(self):
        pass

    @staticmethod
    def create_vocabulary(corpus, type_embedding="word", num_words=10000):
        """Create a sequence tokenizer and generate vocabulary. Supports both 'word' and 'char' sequences.
        
        Arguments:
            corpus {list} -- A list of strings from which word-index or char-index mapping is created.
            num_words {int} -- Maximum number of words to keep, based on word frequency. \
                Only the most common (num_words-1) tokens will be kept. Not necessary when doing character embedding.
        
        Returns:
            TokenizerObject -- Tokenizer object to fit sequences.
            Dict -- Vocabulary dictionary.
        """
        # Custom tokenizer
        if type_embedding.lower() == "word":
            # Word embeddings
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<UNK>")
        else:
            # Character embeddings
            tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, oov_token="<UNK>")
        # Fit tokenizer on the corpus
        tokenizer.fit_on_texts(corpus)
        # Generate vocabulary
        vocab = tokenizer.word_index
        return tokenizer, vocab

    @staticmethod
    def load_embedding(filepath, token_index_mapping, embedding_dim=300):
        """Create an embedding matrix from the given pre-trained vector.
        
        Arguments:
            filepath {str} -- Path to load pre-trained embeddings (ex: glove).
            embedding_dim {int} -- Dimension of the pre-trained embedding.
            token_index_mapping {dict} -- A dictionary containing token-index mapping from the whole corpus.
        
        Returns:
            Matrix -- A numpy matrix containing embeddings for each token in the token-index mapping.
        """
        # Placeholder for embedding
        embedding_index = dict()
        # Access file to load pre-trained embedding
        with open(filepath, mode="r") as fp:
            for line in fp:
                values = line.split()
                token = values[0:-dim]
                coefs = values[-dim:]
                embedding_index[token[0]] = coefs
        # Create a weight matrix for token in training docs
        embedding_matrix = np.zeros((len(token_index_mapping), embedding_dim))
        # Create token-index mapping
        for token, i in word_index.items():
            embedding_vector = embeddings_index.get(token)
            # Update embedding
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    @staticmethod
    def EmbeddingLayer(vocab_size, embedding_dim, max_seq_length, learn_embedding, embedding_matrix):
        """Create an embedding layer for training or to load pre-trained embeddings.
        
        Arguments:
            vocab_size {int} -- [description]
            embedding_dim {int} -- [description]
            max_seq_length {int} -- [description]
            learn_embedding {bool} -- [description]
            embedding_matrix {numpyArray} -- [description]
        
        Returns:
            layer -- Keras embedding layer to be used with othr model.
        """
        if learn_embedding == True:
            embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, mask_zero=True)
        else:
            embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False, input_length=max_seq_length)
        return embedding_layer


class PositionEmbedding(keras.layers.Layer):
    """Compute positional embedding."""
    
    def __init__(self, mode="sum", size=None):
        """Class constructor.
        
        Keyword Arguments:
            mode {str} -- Type of position embedding. (default: {"sum"})
            size {int} -- Size of the position embedding. (default: {None})
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
        """Compute output shape of the tensor using the input shape.
        
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