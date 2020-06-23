from tensorflow import keras
import tensorflow as tf
import numpy as np


class NewLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NewLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)

    def call(self, inputs):
        print(inputs)
        x1, x2 = inputs
        return x1[0] + x2[0]


class MergeLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MergeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs.shape)
        res = 0
        a = inputs[0]
        b = inputs[1]
        c = inputs[2]
        return a + b + c


X1 = np.reshape(np.array([1, 2, 3]), (-1, 1))
X2 = np.reshape(np.array([2, 3, 4]), (-1, 1))
# y = NewLayer()([X1, X2])
X = np.array([1, 2, 3])
y = MergeLayer()(X)


X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])
y = np.array([0, 0, 0])
model = keras.Sequential()
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10)

print(y)
