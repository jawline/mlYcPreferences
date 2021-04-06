import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer

import matplotlib.pyplot as plt

if __name__ == "__main__":
  tf.get_logger().setLevel('DEBUG')
  print("Entered")
