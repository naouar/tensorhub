""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 22:18:30
"""

# Load packages
import os
import sys
import numpy as np
import pandas as pd
from tensorflow.python.keras.api import keras


"""Contains Utilities Methods for Model Training and Validation."""


class DataLoader:
    """Template class for loading data from multiple sources into a dataframe.
    
    Returns:
        DataFrame -- Returns a dataframe object.
    """
    def __init__(self):
        """Class constructor.
        """
        pass

    @staticmethod
    def load_json(filepath):
        """Load .json file into a dataframe object.
        
        Arguments:
            filepath {String} -- Path to source file.
        
        Returns:
            DataFrame -- Returns a dataframe object.
        """
        return pd.read_json(filepath, orient="records", encoding="utf-8", lines=True)
        
    @staticmethod
    def load_csv(filepath):
        """Load .csv file into a dataframe object.
        
        Arguments:
            filepath {String} -- Path to source file.
        
        Returns:
            DataFrame -- Returns a dataframe object.
        """
        return pd.read_csv(filepath, encoding="utf-8")
        
    @staticmethod
    def load_tsv(filepath):
        """Load .tsv file into a dataframe object.
        
        Arguments:
            filepath {String} -- Path to source file.
        
        Returns:
            DataFrame -- Returns a dataframe object.
        """
        return pd.read_csv(filepath, encoding="utf-8", sep="\t")
        

class Embeddings:
    """Class implementing plaground for embeddings.
    
    Returns:
        Matrix -- Embedding matrix.
    """
    def __init__(self):
        """Class Constructor.
        """
        pass

    @staticmethod
    def create_vocabulary(list_text, type_embedding="word", num_words=10000):
        """Create a sequence tokenizer and generate vocabulary.
        
        Arguments:
            list_text {List} -- A list of strings from which word-index or char-index mapping is created.
            num_words {Int} -- Maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
        
        Returns:
            TokenizerObject -- Tokenizer object to fit sequences.
            Dict -- Vocabulary dictionary.
        """
        # Create a custom tokenizer
        if type_embedding.lower() == "word":
            # Word embeddings
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<UNK>")
        else:
            # Character embeddings
            tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, oov_token="<UNK>")
        # Fit tokenizer on the corpus
        tokenizer.fit_on_texts(list_text)
        # Access vocabulary
        vocab = tokenizer.word_index
        print("Number of unique tokens in vocabulary are:", len(vocab) + 1)
        return tokenizer, vocab

    @staticmethod
    def load_embedding(word_index, filepath, dim=300):
        """Create an embedding matrix from the given pre-trained vector.
        
        Arguments:
            filepath {String} -- Path to load pre-trained embeddings (ex: glove).
            dim {Int} -- Dimension of the embedding.
            word_index {Dict} -- A dictionary containing word-index mapping from the whole corpus.
        
        Returns:
            Matrix -- A matrix containing embeddings for words in the word-index mapping.
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
        print("Number of unique tokens in vocabulary are:", len(embedding_matrix) + 1)
        return embedding_matrix