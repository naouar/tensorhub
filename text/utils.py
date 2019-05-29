""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 22:18:30
@Last Modified Time: 2019-05-28 22:18:30
"""

# Load packages
import os
import sys
import re
import numpy as np


def load_embedding(filepath, dim, word_index):
    """Create an embedding matrix from the given pre-trained vector.
    
    Arguments:
        filepath {Strign} -- Location to the pre-trained embedding.
        dim {Int} -- An integer denoting the embedding dimension of the pre-trained embedding.
        word_index {Dict} -- A dictionary containing word-index mapping from the whole corpus of train and test.
    
    Returns:
        NumpyArray -- A numpy matrix containing embeddings for words in the word-index mapping.
    """
    embedding_index = dict()
    with open(filepath, mode="r") as fp:
        for line in fp:
            values = line.split()
            word = values[0:-dim]
            coefs = values[-dim:]
            embedding_index[word[0]] = coefs
    # Create a weight matrix for words in training docs
    embedding_matrix = np.zeros((len(word_index), dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
