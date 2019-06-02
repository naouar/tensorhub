""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 02:56:59
@Last Modified Time: 2019-05-28 02:56:59
"""

# Load packages
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.api import keras

from driver import SequenceClassification
from utils import data_loader, load_embedding, create_voabulary



"""Multiclass Text classification on 'News Healines' dataset."""

# News categorization dataset
filepath = "~/__data__/news-category.json"

# Load data using an appropriate data loader
df = data_loader().load_json(filepath)

print("Data Shape:", df.shape) # Data Shape: (200853, 6)
print("Columns:", df.columns) # Columns: Index(['authors', 'category', 'date', 'headline', 'link', 'short_description'], dtype='object')

# Select appropriate columns and create a numpy array with proper shape
x_train = np.asarray(df.headline).reshape((df.shape[0], 1))
y_train = np.asarray(df.category).reshape((df.shape[0], 1))

# Split data into train and test
test_ratio = 0.3
test_indexes = np.random.randint(low=0, high=df.shape[0], size=int(df.shape[0]*test_ratio))
train_indexes = np.random.randint(low=0, high=df.shape[0], size=int(df.shape[0]*(1-test_ratio)))
x_train, x_test = x_train[train_indexes,: ], x_train[test_indexes,: ]
y_train, y_test = y_train[train_indexes,: ], y_train[test_indexes,: ]

# Convert numpy array into list
x_train = x_train.tolist()
y_train = y_train.tolist()
x_test = x_test.tolist()
y_test = y_test.tolist()

print("Train Shape:", len(x_train)) # Train Shape: (140597, 1)
print("Test Shape:", len(x_test)) # Test Shape: (60255, 1)

# A dictionary mapping words to an integer index
# Collect all textual data: Merge data from headlines and short description
text_data = list()
text_data.extend(list(df.headline))
text_data.extend(list(df.short_description))

# Create a custom tokenizer
tokenizer = keras.preprocessing.text.Tokenizer()

# Fit tokenizer on the corpus
tokenizer.fit_on_texts(text_data)

word_index = tokenizer.word_index # Get vocabulary

print("Number of unique tokens in vocabulary are:", len(word_index) + 1) # 116618

# Restructure the word_index and save other members
# Shift everything by 3
# Original mapping starts from 1
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<END>"] = 3

print("Number of unique tokens in vocabulary are:", len(word_index) + 1) # 116618 + 4

# Tokenize data
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad or truncate sequences to make fixed length input
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train,
    value=word_index["<PAD>"],
    padding="post",
    truncating="post",
    maxlen=128
)

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test,
    value=word_index["<PAD>"],
    padding="post",
    truncating="post",
    maxlen=128
)

# Convert data type to float from int
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Get all class labels
classes = np.unique(y_train)
# Generate a unique index for each class
index = range(len(classes))
# Create class to idnex mapping
class_index = dict(zip(classes, index))

def numerify(labels):
    numerified_labels = list()
    for label in labels:
        label = label[0] # All labels are store in a list
        numerified_labels.append(class_index[label])
    return numerified_labels

# Integer encode labels
y_train = numerify(y_train)
y_test = numerify(y_test)

# Convert integer encoded labels into categorical
y_train = keras.utils.to_categorical(y_train, num_classes=len(classes))
y_test = keras.utils.to_categorical(y_test, num_classes=len(classes))

# Create batch datasets: batches of 32 for train and 64 for test
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Load model architecture and set your configuration
sequence_classifier = SequenceClassification()
my_model = sequence_classifier.get_simple_lstm(
    vocab_size=len(word_index) + 1,
    max_length=128, # Default: 512
    num_nodes=256, # Default: 512
    num_classes=len(classes),
    learn_embedding=True, # Default
    embedding_matrix=None, # Default
    activation=None, # Default
    output_activation="softmax" # Default
)

# Define model configuration
loss_function = keras.losses.CategoricalCrossentropy()

optimizer = keras.optimizers.RMSprop()

# Accumulate performance metrics while training
train_loss = keras.metrics.Mean(name="train_loss")
train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_loss = keras.metrics.Mean(name="test_loss")
test_accuracy = keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Train model using a tensor function
@tf.function()
def train_step(text, labels):
    # Use gradient tape for training the model
    with tf.GradientTape() as tape:
        # Get predictions
        predictions = my_model(text)
        # Compute instantaneous loss
        loss = loss_function(labels, predictions)
    # Update gradients
    gradients = tape.gradient(loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    # Store
    train_loss(loss)
    train_accuracy(labels, predictions)

# Test model using another tensor function
@tf.function()
def test_step(text, labels):
    # Get predictions
    predictions = my_model(text)
    # Compute instantaneous loss
    loss = loss_function(labels, predictions)
    # Store
    test_loss(loss)
    test_accuracy(labels, predictions)

# Set run configuration
epochs = 2
template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"

# Run
for epoch in range(epochs):
    print("Training:", end=" ")
    # Run model on batches
    for text, label in train_ds:
        print(".", end="")
        train_step(text, label)
    print()
    print("Tetsing:", end=" ")
    # Test model on batches    
    for t_text, t_label in test_ds:
        print(".", end="")
        test_step(t_text, t_label)
    print()
    # Prompt user using defined template
    print(template.format(epoch+1, \
        train_loss.result(), train_accuracy.result()*100, \
        test_loss.result(), test_accuracy.result()*100))


if __name__ == "__main__":
    text_classifier()