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

def make_training(data_frame):
  train_inputs = []
  train_outputs = []

  for url in data_frame:
    prepared = prepare_input(url, data_frame[url])

    if prepared == None:
      continue

    train_inputs.append(prepared[0])
    train_outputs.append(prepared[1])

  return (np.array(train_inputs), np.array(train_outputs))

if __name__ == "__main__":
  tf.get_logger().setLevel('DEBUG')
  df = load_data()
  training_data = make_training(df)

  # Preprocess the inputs with bert and resave our training data and predictions
  bert_model = build_bert_model()
  bert_outputs = bert_model.predict(training_data[0])
  training_data = (bert_outputs, training_data[1])

  print(f"Training data: {len(training_data[0])} split 70/30%")

  classifier_model = build_classifier_model(512)

  print("Prepared classifier layer")

  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()

  print("Selected loss and metrics")

  epochs = 10
  batch_size = 32
  steps_per_epoch = len(training_data[0]) / batch_size
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  init_lr = 3e-2
  optimizer = optimization.create_optimizer(init_lr=init_lr,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps, optimizer_type='adamw')

  print("Selected optimizer")

  classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  print("Preparing to train")
  history = classifier_model.fit(x=training_data[0], y=training_data[1], batch_size=batch_size, validation_split=0.3 , epochs=epochs)

  print("Printing results")
  loss, accuracy = classifier_model.evaluate(x=training_data[0], y=training_data[1])

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
