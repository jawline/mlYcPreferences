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
from model import build_classifier_model, prepare_input
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

def emit_training(training_data):
  root_train_dir = "./tmp_train/"

  try:
    shutil.rmtree(root_train_dir)
    os.makedirs(root_train_dir)
  except:
    pass

  train_inputs = training_data[0]
  train_outputs = training_data[1]

  for i in range(0, len(train_inputs)):

    if int(train_outputs[i]) < 2:
      train_outputs[i] = "bad"
    else:
      train_outputs[i] = "good"

    try:
      os.makedirs(f"{root_train_dir}/{train_outputs[i]}/")
    except:
      pass

    with open(f"{root_train_dir}/{train_outputs[i]}/{i}.txt", "w") as f:
      f.write(train_inputs[i])

if __name__ == "__main__":
  tf.get_logger().setLevel('DEBUG')
  df = load_data()
  training_data = make_training(df)
  emit_training(training_data)

  for i in range(0, len(training_data[0])):
    print(training_data[0][i], training_data[1][i])


  # Now we have dumped it all to that directory so we can load is using the keras directory method
  AUTOTUNE = tf.data.AUTOTUNE
  batch_size = 32
  seed = 42

  raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
      './tmp_train/',
      batch_size=batch_size,
      validation_split=0.3,
      subset='training',
      seed=seed)
  class_names = raw_train_ds.class_names
  train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = tf.keras.preprocessing.text_dataset_from_directory(
      './tmp_train/',
      batch_size=batch_size,
      validation_split=0.3,
      subset='validation',
      seed=seed)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  classifier_model = build_classifier_model()

  print("Prepared classifier layer")

  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()

  print("Selected loss and metrics")

  epochs = 50
  steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  init_lr = 3e-2
  optimizer = optimization.create_optimizer(init_lr=init_lr,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps, optimizer_type='adamw')

  print("Selected optimizer")

  classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  print("Preparing to train")
  history = classifier_model.fit(x=train_ds, validation_data=val_ds, epochs=epochs)

  print("Printing results")
  loss, accuracy = classifier_model.evaluate(val_ds)

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
