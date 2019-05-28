""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 00:46:39
@Last Modified Time: 2019-05-28 00:46:39
"""

# Load packages
import os
import sys

from models import *


class SequenceClassification():
    def __init__(self):
        pass

    def get_simple_rnn(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding=False, embedding_matrix=None):
        return SimpleRNN(max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix)

    def get_simple_lstm(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding=False, embedding_matrix=None):
        return SimpleLSTM(max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix)
        
    def get_simple_gru(self, max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding=False, embedding_matrix=None):
        return SimpleGRU(max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix)


class NamedEntityRecognition():
    def __init__(self):
        pass
