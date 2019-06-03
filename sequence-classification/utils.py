""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 22:18:30
@Last Modified Time: 2019-05-28 22:18:30
"""

# Load packages
import os
import re
import sys
import numpy as np
import pandas as pd
from tensorflow.python.keras.api import keras


class data_loader:
    """Template class for data loaders.
    
    Returns:
        DataFrame -- Returns a df object.
    """
    def __init__(self):
        print("Initiated Data Loader!")

    @staticmethod
    def load_json(filepath):
        """Static method to load JSON data into DF objct.
        
        Arguments:
            filepath {String} -- JSON file path.
        
        Returns:
            DataFrame -- Returns a DF object.
        """
        df = pd.read_json(filepath, orient="records", encoding="utf-8", lines=True)
        return df
    
    @staticmethod
    def load_csv(filepath):
        """Static method to load CSV data into DF objct.
        
        Arguments:
            filepath {String} -- CSV file path.
        
        Returns:
            DataFrame -- Returns a DF object.
        """
        df = pd.read_csv(filepath, encoding="utf-8")
        return df
    
    @staticmethod
    def load_tsv(filepath):
        """Static method to load TSV data into DF objct.
        
        Arguments:
            filepath {String} -- TSV file path.
        
        Returns:
            DataFrame -- Returns a DF object.
        """
        df = pd.read_csv(filepath, encoding="utf-8", sep="\t")
        return df


def create_embeddings(text_data, type_embedding):
    """Method to create a tokenizer and generate word vocabulary.
    
    Arguments:
        text_data {List} -- A list of strings from which word-index mapping is created. Generally it is the entire corpus of text available.
    """
    # Create a custom tokenizer
    if type_embedding.lower() == "word":
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=False, oov_token="<UNK>")
    else:
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, oov_token="<UNK>")
    # Fit tokenizer on the corpus
    tokenizer.fit_on_texts(text_data)

    word_index = tokenizer.word_index # Get vocabulary
    print("Number of unique tokens in vocabulary are:", len(word_index) + 1) # 116618

    # Restructure the word_index and save other members
    # Shift everything by 3
    # Original mapping starts from 1
    return tokenizer, word_index


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
