from tensorflow import keras
import tensorflow as tf
import numpy as np


class CustomModel(keras.Model):

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        layer = self.get_layer('sp')
        layer.

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


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
        self.x = tf.Variable([1., 1., 1.])

    def call(self, inputs):
        # print(inputs)
        return self.wrap(inputs, self.x)

    def wrap(self, i, j):
        
        return sum([i[idx] * j[idx] for idx in range(3)])



X1 = np.reshape(np.array([1, 2, 3]), (-1, 1))
X2 = np.reshape(np.array([2, 3, 4]), (-1, 1))
# y = NewLayer()([X1, X2])
# X = np.array([1, 2, 3])
# y = MergeLayer()(X)


X = np.array([
    [1.5, 2., 3.],
    [2., 3., 4.],
    [3., 4., 5.]
], dtype=float)
y = np.array([0, 0, 0])
inputs = keras.Input(shape=(3))
output = MergeLayer(name='sp')(inputs)
model = CustomModel(inputs, output)
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10)

print(y)
