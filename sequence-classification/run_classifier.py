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
from sklearn.model_selection import train_test_split

from driver import SequenceClassification
from utils import data_loader, load_embedding, create_tokenizer



"""Multiclass Text classification on 'News Healines' dataset."""

# News categorization dataset
filepath = "~/__data__/news-category.json"

# Load data using an appropriate data loader
df = data_loader().load_json(filepath)

print("Data Shape:", df.shape) # Data Shape: (200853, 6)
print("Columns:", df.columns) # Columns: Index(['authors', 'category', 'date', 'headline', 'link', 'short_description'], dtype='object')

# Select appropriate columns and create a numpy array with proper shape
x = list(df.headline)
y = list(df.category)

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y) # Default test size of 25%

print("Train samples:", len(x_train)) # Train size: 150639
print("Test samples:", len(x_test)) # Test size: 50214

# A dictionary mapping words to an integer index
# Collect all textual data: Merge data from headlines and short description
corpus = list()
corpus.extend(x_train)
corpus.extend(x_test)

# Compute the maximum number of words in a sequence
max_num_words = len(max(corpus).split())

# Train a custom tokenier on the corpus and generate tokenizer instance and word vocabulary
tokenizer, word_index = create_tokenizer(corpus)

# Tokenize data using the created tokenizer
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad or truncate sequences to make fixed length input
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train,
    value=word_index["<PAD>"],
    padding="post",
    truncating="post",
    maxlen=max_num_words
)

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test,
    value=word_index["<PAD>"],
    padding="post",
    truncating="post",
    maxlen=max_num_words
)

# Convert data type to float from int
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Get all class labels
classes = np.unique(y_train)
# Generate a unique index for each class
index = range(len(classes))
# Create class to index mapping
class_index = dict(zip(classes, index))

def numerify(labels):
    numerified_labels = list()
    for label in labels:
        numerified_labels.append(class_index[label])
    return numerified_labels

# Now encode labels using the created mapping
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
    max_length=max_num_words, # Default: 512
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
print("{:#^50s}".format("Train and Validate"))
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch, epochs))
    # Run model on batches
    for text, label in train_ds:
        train_step(text, label)
    # Test model on batches    
    for t_text, t_label in test_ds:
        test_step(t_text, t_label)
    # Prompt user using defined template
    print(template.format(epoch+1, \
        train_loss.result(), train_accuracy.result()*100, \
        test_loss.result(), test_accuracy.result()*100))
