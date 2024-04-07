# importing the required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import re
import string
import tensorflow_datasets as tfds
import time


""" ************************************
    This file contains Utility functions
    
    ************************************ """

# defining a function that cleans the text
def clean_text(s):
  # changing the short form words like didn't, wouldn't
  cleaned = re.sub(r"n\'t", " not", s)
  cleaned = re.sub(r"\'re", " are", cleaned)
  cleaned = re.sub(r"\'s", " is", cleaned)
  cleaned = re.sub(r"\'d", " would", cleaned)
  cleaned = re.sub(r"\'ll", " will", cleaned)
  cleaned = re.sub(r"\'t", " not", cleaned)
  cleaned = re.sub(r"\'ve", " have", cleaned)
  cleaned = re.sub(r"\'m", " am", cleaned)
  cleaned = re.sub("(\s+)", " ", cleaned)
  out = cleaned.translate(str.maketrans('', '', string.punctuation))
  return out



# defining a function to add special tokens
def add_special_tokens(s):
  return '<SOS>'+' '+s+' '+'<EOS>'



# defining a function which preprocess the data
def preprocess(df):
  # remove the NaNs
  df=df.dropna()
  # removing the duplicates
  df=df.drop_duplicates()
  # applying the clean_text function to remove punctuations from hindi
  df['hinglish']=df['hinglish'].apply(lambda x: clean_text(x))
  # making all hinglish words into lower case
  df['hinglish']=df['hinglish'].apply(lambda x: x.lower())
  # applying the clean_text function to english
  df['english']=df['english'].apply(lambda x: clean_text(x))
  # making all english words into lower case
  df['english']=df['english'].apply(lambda x: x.lower())
  return df


# wrapping our encode function using tf_encode function
def tf_encode(hin,en):
  hin_result,en_result=tf.py_function(encode,[hin,en],[tf.int64,tf.int64])
  hin_result.set_shape([None])
  en_result.set_shape([None])
  return hin_result,en_result


# writing a function to filter our data ie remove datapoints (hin,en) where if any one (pt or en) has more than 40 tokens
def max_len_filter(hin,en,max_len=40):
  # returning a mask (1 if both pt and en has less than 40 tokens each else 0)
  return tf.logical_and(tf.size(hin)<=max_len,
                        tf.size(en)<=max_len)