from tensorflow import keras
import tensorflow as tf
import numpy as np

print(tf.argmax(tf.constant([1, 2, 1])) + 1)
dd = tf.constant([1, 1, 1]).numpy()
f = open('333', 'w')
f.write(str(dd))
f.close()