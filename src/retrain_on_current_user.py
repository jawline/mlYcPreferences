import time
import os
import shutil
import re
from urllib.parse import urlparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from official.nlp import optimization  # to create AdamW optmizer
from model import build_bert_model, build_classifier_model, prepare_input
import matplotlib.pyplot as plt


def load_data():
    try:
        df = pd.read_csv("user.csv")
    except Exception as e:
        print(f"{e} - creating fresh")
        df = pd.DataFrame({})
    return df

# Returns a triple of (bert_inputs, other_inputs, expected_outputs)
# The bert inputs should first be sent to the BERT classifier to get the BERT encoding
# The bert encodings should be combined with the other inputs to get the input to the classification model
# The model should then be trained with the expected outputs given in train_outputs


def make_training(data_frame):
    train_bert_titles = []
    train_bert_fqdn = []
    train_other_inputs = []
    train_outputs = []

    for row in data_frame.iterrows():
        prepared = prepare_input(row[1])

        if prepared == None:
            continue

        train_bert_titles.append(prepared[0])
        train_bert_fqdn.append(prepared[1])
        train_other_inputs.append(prepared[2:-1])
        train_outputs.append(prepared[-1])

    return (np.array(train_bert_titles), np.array(train_bert_fqdn), np.array(train_other_inputs), np.array(train_outputs))


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    df = load_data()
    training_data = make_training(df)

    # Preprocess the inputs with bert and resave our training data and predictions
    bert_model = build_bert_model()
    bert_outputs_titles = bert_model.predict(training_data[0])
    bert_outputs_fqdn = bert_model.predict(training_data[1])

    # Combine the BERT output with the other features (the scores, comments, age, etc)
    training_data_x = np.array([np.concatenate((bert_outputs_titles[i], bert_outputs_fqdn[i],
                               training_data[2][i])) for i in range(0, len(bert_outputs_titles))])
    training_data_y = training_data[-1]

    print(f"Training data: {len(training_data[0])} split 70/30%")
    classifier_model = build_classifier_model(512 + 512 + 3)

    print("Prepared classifier layer")

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    print("Selected loss and metrics")

    epochs = 400
    batch_size = 4
    steps_per_epoch = len(training_data[0]) / batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-2
    optimizer = optimization.create_optimizer(
        init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')

    print("Selected optimizer")

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Setting up checkpoints")
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        '.checkpoint.hdf5', save_freq='epoch', save_best_only=True, monitor='val_loss', mode='min')

    print("Preparing to train")
    history = classifier_model.fit(x=training_data_x, y=training_data_y, batch_size=batch_size,
                                   validation_split=0.3, callbacks=[mcp_save], epochs=epochs)

    print("Loading best saved checkpoint")
    classifier_model.load_weights(".checkpoint.hdf5")

    print("Printing results")
    loss, accuracy = classifier_model.evaluate(
        x=training_data_x, y=training_data_y)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    print("Saving Model")

    # Save the entire model as a SavedModel.
    try:
        os.makedirs("saved_model")
    except:
        pass

    # Save the model as the current timestamp
    classifier_model.save(f"saved_model/{int(time.time())}")

    # Delete the current model if it exists
    try:
        shutil.rmtree("saved_model/current/")
    except:
        pass

    # Save the model as current
    classifier_model.save("saved_model/current")
