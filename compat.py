import tensorflow as tf
import numpy as np

a = tf.constant([
    [1, 2, 3],
    [1, 2, 3]
])

b = tf.constant([
    [1],
    [2]
])

print(b * a)