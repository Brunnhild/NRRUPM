from tensorflow import keras
import tensorflow as tf
import numpy as np

cc = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])
print(cc)
cc = tf.expand_dims(cc, 1)
print(cc.shape)

dd = np.tile(cc, tf.constant((1, 2, 1)))
print(dd)
