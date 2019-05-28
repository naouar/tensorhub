""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 02:56:59
@Last Modified Time: 2019-05-28 02:56:59
"""

# Load packages
import os
import sys
import numpy as np
import tensorflow as tf

from driver import SequenceClassification


def text_classifier_on_imdb():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

    # A dictionary mapping words to an integer index
    word_index = tf.keras.datasets.imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    # Pad sequence for fixed length input
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train,
        value=word_index["<PAD>"],
        padding="post",
        maxlen=256
    )

    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test,
        value=word_index["<PAD>"],
        padding="post",
        maxlen=256
    )

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Create batch datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # Load model architecture
    my_model = SequenceClassification().get_simple_lstm(max_features=len(word_index), width=512, num_class=1, out_act="sigmoid")

    # Model configuration
    loss_function = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Accumulate the values over epochs and then print the overall result
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

    # Train model
    def train_step(text, labels):
        # Use gradient tape for training the model
        with tf.GradientTape() as tape:
            # Get predictions
            predictions = my_model(text)
            # Compute instance loss
            loss = loss_function(labels, predictions)
        # Update gradients
        gradients = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
        # Store
        train_loss(loss)
        train_accuracy(labels, predictions)
    
    # Test model
    def test_step(text, labels):
        # Get predictions
        predictions = my_model(text)
        # Compute instance loss
        loss = loss_function(labels, predictions)
        # Store
        test_loss(loss)
        test_accuracy(labels, predictions)

    # Train
    epochs = 1
    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"

    for epoch in range(epochs):
        # Run model on batches
        for text, label in train_ds:
            train_step(text, label)
        
        for t_text, t_label in test_ds:
            test_step(t_text, t_label)
        
        print(template.format(epoch+1, \
            train_loss.result(), train_accuracy.result()*100, \
                test_loss.result(), test_accuracy.result()*100))


if __name__ == "__main__":
    text_classifier_on_imdb()