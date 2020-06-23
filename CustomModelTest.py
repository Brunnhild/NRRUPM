import numpy as np
from tensorflow import keras
import tensorflow as tf


mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
loss_tracker = keras.metrics.Mean(name="loss")


class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        tf.print(x)
        layer = self.get_layer('sp')
        tf.print(layer.weights)
        # w, b = layer.weights
        # w.assign_add(w)
        # b.assign_add(tf.constant([1.]))
        # tf.print(layer.weights)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.losses.mean_squared_error(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}


X = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5]
])

Y = np.array([1, 1, 1, 1])

inputs = keras.Input(shape=(4,))
a = keras.layers.Reshape((-1, 2))(inputs)
outputs = keras.layers.Dense(1, name='sp')(a)
model = CustomModel(inputs, outputs)

model.compile(optimizer="adam")

model.fit(X, Y, epochs=3)