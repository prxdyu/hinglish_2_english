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

from utils import *
from model import *


# importing train data
train_data=pd.read_csv("data/TRAIN.csv")
# importing the validation data
val_data=pd.read_csv("data/VALIDATION.csv")


# preprocessing train_data
train_df = preprocess(train_data)
# preprocessing the validation data
val_df = preprocess(val_data)



# Define file paths for saved tokenizers
hindi_tokenizer_path = "./hindi_tokenizer.subwords"
en_tokenizer_path = "./en_tokenizer.subwords"



# Check if the saved tokenizers exist
if os.path.exists(hindi_tokenizer_path+".subwords") and os.path.exists(en_tokenizer_path+".subwords"):
    # Load the Hindi tokenizer
    hindi_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(hindi_tokenizer_path)

    # Load the English tokenizer
    en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(en_tokenizer_path)
    print("Tokenizers loaded successfully.")
else:
    print("Tokenizers not found. Building from scratch")
    SOS_TOKEN='<SOS>'
    EOS_TOKEN='<EOS>'

    # building the tokenizer for hinglish
    hindi_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator=(row[1][1] for row in train_df.iterrows()),
        target_vocab_size=2**13,
        reserved_tokens=[SOS_TOKEN,EOS_TOKEN])

    # building the tokenizer for english
    en_tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=(row[1][0] for row in train_df.iterrows()),
                target_vocab_size=2**13,
                reserved_tokens=[SOS_TOKEN,EOS_TOKEN])
    
def encode(lang1,lang2):
  """
  lang1: hindi
  lang2: english
  encodes the tokens to indices
  """
  lang1= hindi_tokenizer.encode(lang1.numpy())
  lang2= en_tokenizer.encode(lang2.numpy())
  return lang1,lang2



# converting pandas dataframe to tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['hinglish'].values, train_df['english'].values))
val_dataset = tf.data.Dataset.from_tensor_slices((val_df['hinglish'].values, val_df['english'].values))

# applying the tf_encode to our train dataset
train_data=train_dataset.map(tf_encode)
# filtering the train data
train_data=train_data.filter(max_len_filter)

# applying the tf_encode to our validation dataset
val_data=val_dataset.map(tf_encode)
# filtering the validation data
val_data=val_data.filter(max_len_filter)


# defining the global variables
BATCH_SIZE=64
BUFFER_SIZE=2000

# caching the train data
train_data=train_data.cache()
#shuffling and padding the train data
train_data=train_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
# prefetching batches in training data
train_data=train_data.prefetch(tf.data.experimental.AUTOTUNE)

# caching the val data
val_data=val_data.cache()
# shuffling and padding the val data
val_data=val_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
# prefetching batches in val data
val_data=val_data.prefetch(tf.data.experimental.AUTOTUNE)


# defining the hyper parameters
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1


# defining a transformer model
transformer= Transformer(num_layers=NUM_LAYERS,
                         d_model=D_MODEL,
                         num_heads=NUM_HEADS,
                         dff=DFF,
                         input_vocab_size=hindi_tokenizer.vocab_size,
                         target_vocab_size=en_tokenizer.vocab_size,
                         pe_input=1000,
                         pe_target=1000,
                         rate=DROPOUT_RATE)

# creating learning rate custom scheduler
learning_rate= CustomScheduler(D_MODEL)
# creating the optimizer
optimizer= tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


# TRAINING
num_epochs=40

for epoch in range(num_epochs):
  # noting the time of the start
  start=time.time()
  # resetting the train loss and train accuracy to calculate fresh for the each epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  # resetting the validation loss and val accuracy to calculate fresh for each epoch
  val_loss.reset_states()
  val_accuracy.reset_states()

  # TRAINING
  for (batch,(inp,tar)) in enumerate(train_data):
    # training the batch using train_step function
    train_step(inp,tar,transformer,optimizer)
    if batch % 250 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  # VALIDATION
  for (batch,(inp, tar)) in enumerate(val_data):
    # validating the batch using val_step fuction
    val_step(inp, tar, transformer)

  # Print training and validation metrics
  print(f'\n Epoch {epoch + 1} \n Training Loss: {train_loss.result():.4f} Training Accuracy: {train_accuracy.result():.4f} \n Validation Loss: {val_loss.result():.4f} Validation Accuracy: {val_accuracy.result():.4f}')
  print(f'Time taken for epoch {epoch + 1}: {time.time() - start:.2f} secs\n')

  # Save checkpoint
  ckpt_save_path = ckpt_manager.save()
  print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')


# Inference
sample_sentence=input("Enter a Hinglish sentence: ")
transated_sentence=translate(sample_sentence)

print(sample_sentence)
print(transated_sentence)


