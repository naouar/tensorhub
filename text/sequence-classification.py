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

    def get_simple_rnn(self, width=128, num_class=10, out_act="softmax"):
        return SimpleRNN(width, num_class, out_act)

    def get_simple_lstm(self, width=128, num_class=10, out_act="softmax"):
        return SimpleLSTM(width=128, num_class=10, out_act="softmax")
        
    def get_simple_gru(self, width=128, num_class=10, out_act="softmax"):
        return SimpleGRU(width, num_class, out_act)


# Driver
if __name__ == "__main__":
    my_model = SequenceClassification().get_simple_lstm()
    print(my_model)

    my_model = SequenceClassification().get_simple_rnn()
    print(my_model)

    my_model = SequenceClassification().get_simple_gru()
    print(my_model)