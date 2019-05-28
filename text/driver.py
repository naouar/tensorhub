""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 00:46:39
@Last Modified Time: 2019-05-28 00:46:39
"""

# Load packages
import os
import sys
import numpy as np
import tensorflow as tf

from models import *


class SequenceClassification():
    def __init__(self):
        pass

    def get_simple_rnn(self, max_features, width, num_class, out_act):
        return SimpleRNN(max_features, width, num_class, out_act)

    def get_simple_lstm(self, max_features, width, num_class, out_act):
        return SimpleLSTM(max_features, width, num_class, out_act)
        
    def get_simple_gru(self, max_features, width, num_class, out_act):
        return SimpleGRU(max_features, width, num_clas, out_act)
    
    def get_mlp(self, width, num_class, out_act):
        return MLP(width, num_class, out_act)


class NamedEntityRecognition():
    def __init__(self):
        pass
