""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 02:56:59
"""

# Load packages
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.api import keras
from sklearn.model_selection import train_test_split

from utils import DataLoader, Embeddings
from nlp.text_classification import SimpleRNN, SimpleLSTM, SimpleGRU, TextCNN


"""Multiclass Text Classification on 'News Healines' Dataset."""

filepath = "~/__data__/news-category.json" # News dataset

# Load data using an appropriate data loader
df = DataLoader().load_json(filepath)

print("Data Shape:", df.shape)
print("Columns:", df.columns)

# Select feature and target column
x = list(df.headline)
y = list(df.category)

# Split data into train and test
# Train -> 75%
# Test -> 25%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Number of train samples:", len(x_train)) # List
print("Number of test samples:", len(x_test))

# Compute the maximum number of words and characters in a sequence from the whole corpus
max_num_words = len(max(x).split())
max_num_chars = len(max(x))

print("Maximum number of word in a sequence are", max_num_words)
print("Maximum number of chars in a sequence are", max_num_chars)

# Train a custom tokenier on the corpus and generate tokenizer instance with vocabulary
# type_embedding: 'word' & 'char'
tokenizer, word_index = Embeddings.create_vocabulary(x, type_embedding="word")

# Load pre-trained embedding matrix if learn_embedding is False. See model configuration.
# embedding_matrix = Embeddings.load_embedding(word_index, filepath="‎⁨~/__embeddings__⁩/glove.840B.300d.txt", dim=300)

# Tokenize data using the created tokenizer instance
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad or truncate sequences to make fixed length input
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train,
    value=0, # Pad with 0
    padding="post",
    truncating="post",
    maxlen=max_num_words # When using 'char' embedding use max_num_chars
)

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test,
    value=0, # Pad with 0
    padding="post",
    truncating="post",
    maxlen=max_num_words # When using 'char' embedding use max_num_chars
)

# Convert data type to float from int
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Get all class labels
classes = np.unique(y_train)

# Generate a unique index for each class from 0 to len(classes) - 1
index = range(len(classes))

# Create class to index mapping
class_index = dict(zip(classes, index))
reverse_class_index = dict(zip(index, classes))

# Now encode labels using the created mapping
y_train = [class_index[label] for label in y_train]
y_test = [class_index[label] for label in y_test]

# Convert to categorical 
y_train = keras.utils.to_categorical(y_train, num_classes=len(classes))
y_test = keras.utils.to_categorical(y_test, num_classes=len(classes))

# Load model architecture with its default settings
# # RNN model
# rnn_model = SimpleRNN(
#     vocab_size=len(word_index) + 1,
#     num_classes=len(classes)
# )

# LSTM model
lstm_model = SimpleLSTM(
    vocab_size=len(word_index) + 1,
    num_classes=len(classes)
)

# OR
# Call a model with custom configuration
# GRU model
# gru_model = SimpleGRU(
#     vocab_size=len(word_index) + 1,
#     max_length=max_num_words,
#     num_classes=len(classes),
#     num_nodes=[512, 1024, 1024],
#     activation="relu",
#     output_activation="softmax",
#     learn_embedding=False,
#     embed_dim=300,
#     embedding_matrix=embedding_matrix
# )

# Text-CNN model
# text_cnn_model = TextCNN(
#     vocab_size=len(word_index)+1,
#     num_classes=len(classes),
#     filters=[64, 64],
#     kernals=[3, 3],
#     strides=[1, 1],
#     max_length=max_num_words,
#     drop_rate=0.4,
#     activation="relu",
#     output_activation="softmax",
#     learn_embedding=True,
#     embed_dim=100,
#     embedding_matrix=None
# )

# Create batch datasets: batches of 32 for train and 64 for test
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# Define model configuration
loss_function = keras.losses.CategoricalCrossentropy()

# Define optimizer
optimizer = keras.optimizers.RMSprop()

# Accumulate performance metrics while training
train_loss = keras.metrics.Mean(name="train_loss")
train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_loss = keras.metrics.Mean(name="test_loss")
test_accuracy = keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Train model using a tensor function
def train_validate_model(model, train_ds, test_ds, epochs=3):
    template = "Epoch {}, Loss: {}, Accuracy: {}%, Test Loss: {}, Test Accuracy: {}%"
    @tf.function()
    def train_step(text, labels):
        # Use gradient tape for training the model
        with tf.GradientTape() as tape:
            # Get predictions
            predictions = model(text)
            # Compute instantaneous loss
            loss = loss_function(labels, predictions)
        # Update gradients
        gradients = tape.gradient(loss, lstm_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, lstm_model.trainable_variables))
        # Store
        train_loss(loss)
        train_accuracy(labels, predictions)

    # Test model using another tensor function
    @tf.function()
    def test_step(text, labels):
        # Get predictions
        predictions = model(text)
        # Compute instantaneous loss
        loss = loss_function(labels, predictions)
        # Store
        test_loss(loss)
        test_accuracy(labels, predictions)

    print("{:#^50s}".format("Train and Validate"))
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        # Train model on batches
        for text, labels in train_ds:
            train_step(text, labels)
        # Test model on batches    
        for t_text, t_labels in test_ds:
            test_step(t_text, t_labels)
        # Prompt user
        print(template.format(epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    print("Training Completed!")

# Run
train_validate_model(lstm_model,  train_ds, test_ds)