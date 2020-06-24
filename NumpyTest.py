import numpy as np
from collections import Counter
import tensorflow as tf

a = tf.Variable([1, 2, 3])
b = tf.Variable([2, 3, 4])
c = tf.concat([a, b], axis=-1)
print(c)
