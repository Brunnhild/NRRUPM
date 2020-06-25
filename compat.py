import tensorflow as tf
import numpy as np

a = np.reshape(range(8), (2, 2, 2))

b = np.reshape([1, 2, 3, 4], (2, 2))

print(a)
print(b)
print(a + b)
