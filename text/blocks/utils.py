""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 22:18:30
"""


# Load packages
import os
import sys
import pandas as pd


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
