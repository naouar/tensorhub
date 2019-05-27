""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 02:56:59
@Last Modified Time: 2019-05-28 02:56:59
"""

# Load packages
import os
import sys
import tensorflow as tf

from driver import SequenceClassification


if __name__ == "__main__":
    obj = SequenceClassification().get_mlp()