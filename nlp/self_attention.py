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

"""Multi-head Self-Attention implementation as a layer."""


class Attention(keras.layers):
    pass