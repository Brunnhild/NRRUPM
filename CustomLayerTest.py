from tensorflow import keras
import tensorflow as tf
import numpy as np

a = tf.constant([
    [1, 2, 3],
    [2, 3, 4]
])

print(tf.reduce_sum(a, axis=0))
