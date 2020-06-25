from data import get_train_input, get_vocabulary
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from collections import Counter
import numpy as np
import os
import pickle
import tensorflow as tf


WORD_DIM = 200
MIDDLE_OUTPUT = 50
DOCUMENT_ENCODER_OUTPUT = 25
USER_VEC_DIM = 50
PROD_VEC_DIM = 50
SENTENCE_MEM_DIM = 50
DOCUMENT_MEM_DIM = 50
REPRESENTATIVES = 2
EPOCHS = 2
BATCH_SIZE = 2
MEMORY_UPDATE_CORE_UNITS = 50


def dot_product(a, b):
    return tf.reduce_sum(tf.multiply(a, b))

'''
Input: The user vector or product vector
Output: The concatenated column vector of dimension (2 * USER_VEC_DIM, 1)
'''
class ItemVectorTransform(Layer):

    def __init__(self, user_memory_vec, units=MEMORY_UPDATE_CORE_UNITS):
        super(ItemVectorTransform, self).__init__()
        self.user_memory_vec = tf.Variable(
            initial_value=user_memory_vec,
            trainable=False
        )
        self.units = units

    def update_memory(self, d_batch, u_batch):
        for i in range(d_batch.shape[0]):
            d = d_batch[i]
            u = u_batch[i]
            g = []
            weighted_u = []
            for i in self.user_memory_vec.shape[0]:
                m = self.user_memory_vec[i]
                wu = tf.matmul(self.wu, u)
                wm = tf.matmul(self.wm, m)
                wd = tf.matmul(self.wd, d)
                s = tf.sigmoid(wu + wm + wd + self.bg)
                g.append(s)
                weighted_u.append((1 - s) * u)
            g = tf.reshape(g, (-1, 1))
            self.user_memory_vec = g * self.user_memory_vec + weighted_u
    
    def build(self, input_shape):
        self.wu = self.add_weight(
            shape=(self.units, USER_VEC_DIM),
            initializer="random_normal",
            trainable=True
        )
        self.wm = self.add_weight(
            shape=(self.units, USER_VEC_DIM),
            initializer="random_normal",
            trainable=True
        )
        self.wd = self.add_weight(
            shape=(self.units, 2 * DOCUMENT_ENCODER_OUTPUT),
            initializer="random_normal",
            trainable=True
        )

        self.bg = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        res = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for u in inputs:
            mu = tf.matmul(self.user_memory_vec, tf.reshape(u, (-1, 1))) # shape: (rep, 1)
            gammar = tf.exp(mu) / tf.reduce_sum(tf.exp(mu)) # shape: (rep, 1)
            gammar = tf.tile(gammar, (1, u.shape[0])) # shape: (rep, USER_VEC_DIM)
            gm = gammar * self.user_memory_vec
            u_ = tf.reduce_sum(gm, axis=0)
            res = res.write(res.size(), tf.concat([u, u_], axis=0))
            # sum_r = sum([tf.exp(dot_product(self.user_memory_vec[i], u)) for i in range(self.user_memory_vec.shape[0])])
            # r = [tf.exp(dot_product(self.user_memory_vec[i], u)) for i in range(self.user_memory_vec.shape[0])]
            # u_ = sum([r[i] * self.user_memory_vec[i] for i in range(self.user_memory_vec.shape[0])])
            # res.write(res.size(), tf.concat([u, tf.transpose(u_)], axis=-1))
        return res.stack()

    def compute_output_shape(self, input_shape):
        return (None, 2 * input_shape[-1])


'''
Input: 1) Output of sentence encoder (-1, max_word * max_sentence, 2 * MIDDLE_OUTPUT)
       2) The concatenated user vector width dimension (2 * USER_VEC_DIM, 1)
Output: Dimension (sentence_number, 2 * MIDDLE_OUTPUT)
'''
class SentenceToDocument(Layer):

    def __init__(self, sentence_number, units=SENTENCE_MEM_DIM, **kwargs):
        super(SentenceToDocument, self).__init__(**kwargs)
        self.sentence_number = sentence_number
        self.units = units

    def build(self, input_shape):
        self.vw = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )
        self.wh = self.add_weight(
            shape=(self.units, 2 * MIDDLE_OUTPUT),
            initializer="random_normal",
            trainable=True
        )
        self.wu = self.add_weight(
            shape=(self.units, 2 * USER_VEC_DIM),
            initializer="random_normal",
            trainable=True
        )
        self.bw = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        res = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for con_input in inputs:
            sentence_output = con_input[:-1] 
            section_size = sentence_output.shape[0] // self.sentence_number
            user_vec = con_input[-1] # shape: (USER_VEC_DIM,)
            wpu = tf.matmul(self.wu, tf.reshape(user_vec, (-1, 1)))
            wub = tf.transpose(wpu + self.bw) # shape: (1, self.units)
            wub = tf.tile(wub, (section_size, 1)) # shape: (section_size, units)
            section_s = tf.TensorArray(tf.float32, size=self.sentence_number, dynamic_size=True)
            for i in range(self.sentence_number):
                h = sentence_output[i*section_size:(i+1)*section_size] # shape: (section_size, 2 * MIDDLE_OUTPUT)
                wph = tf.matmul(h, tf.transpose(self.wh)) # shape: (section_size, units)
                tanh = tf.tanh(wph + wub)
                e = tf.matmul(tanh, self.vw) # shape: (section_size, 1)
                alpha = tf.exp(e) / tf.reduce_sum(tf.exp(e)) # shape: (section_size, 1)
                s = tf.tile(alpha, (1, h.shape[1])) * h # shape: (section_size, 2 * MIDDLE_OUTPUT)
                s = tf.reduce_sum(s, axis=0) #shape: (2 * MIDDLE_OUTPUT,)
                section_s = section_s.write(i, s)
            section_s = section_s.stack() # shape: (self.sentence_number, 2 * MIDDLE_OUTPUT)
            res = res.write(res.size(), section_s)
        return res.stack()

    def compute_output_shape(self, input_shape):
        return (None, self.sentence_number, 2 * MIDDLE_OUTPUT)

'''
Input: 1) Output of document encoder (-1, max_sentence, 2 * MIDDLE_OUTPUT)
       2) The concatenated user vector width dimension (2 * USER_VEC_DIM, 1)
Output: Dimension (sentence_number, 2 * MIDDLE_OUTPUT)
'''
class DocumentToOutput(Layer):

    def __init__(self, sentence_number, units=DOCUMENT_MEM_DIM):
        super(DocumentToOutput, self).__init__()
        self.sentence_number = sentence_number
        self.units = units

    def build(self, input_shape):
        self.vs = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )
        self.wh = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.wu = self.add_weight(
            shape=(self.units, 2 * USER_VEC_DIM),
            initializer="random_normal",
            trainable=True
        )
        self.bs = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        res = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for con_input in inputs:
            document_output = con_input[:-1] # shape: (self.sentence_number, 2 * MIDDLE_OUTPUT)
            user_vec = con_input[-1]
            wph = tf.matmul(document_output, tf.transpose(self.wh)) # shape: (self.sentence_number, self.units)
            wpu = tf.matmul(self.wu, tf.reshape(user_vec, (-1, 1)))
            wub = tf.transpose(wpu + self.bs) # shape: (1, self.units)
            wub = tf.tile(wub, (self.sentence_number, 1)) # shape: (self.sentence_number, units)
            tanh = tf.tanh(wph + wub)
            e = tf.matmul(tanh, self.vs) # shape: (self.sentence_number, 1)
            beta = tf.exp(e) / tf.reduce_sum(tf.exp(e)) # shape: (self.sentence_number, 1)
            d = tf.tile(beta, (1, document_output.shape[1])) * document_output # shape: (self.sentence_number, 2 * MIDDLE_OUTPUT)
            d = tf.reduce_sum(d, axis=0)
            res = res.write(res.size(), d)
        return res.stack()

def transform_id(item_vec, item_id):
    item_map = dict()
    item_vector_input, idx = [], 0
    for i in item_id:
        if item_map.__contains__(i):
            item_vector_input.append(item_vec[item_map[i]])
        else:
            item_map[i] = idx
            item_vector_input.append(item_vec[idx])
            idx += 1
    return np.reshape(item_vector_input, (-1, USER_VEC_DIM))


if __name__ == '__main__':
    # train_input = get_train_input(get_vocabulary())[:1000]
    # print(get_train_input(get_vocabulary()))
    if os.path.exists('input'):
        training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence = pickle.load(open('input', 'rb'))
    else:
        training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence = get_train_input(get_vocabulary())
        pickle.dump((training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence), open('input', 'wb'))
    print('Start processing')
    print('The max document length is %d' % (max_sentence))
    print('The max sentence length is %d' % (max_word))
    user_cnt = len(Counter(training_user_id))
    product_cnt = len(Counter(training_product_id))
    user_vector = np.array(np.random.rand(user_cnt, USER_VEC_DIM), dtype='float32')
    product_vector = np.array(np.random.rand(product_cnt, USER_VEC_DIM), dtype='float32')

    UserVectorTransformer = ItemVectorTransform(user_vector[:REPRESENTATIVES])
    ProVectorTransformer = ItemVectorTransform(product_vector[:REPRESENTATIVES])

    # Prepare the input data
    user_vector_input = transform_id(user_vector, training_user_id)
    pro_vector_input = transform_id(product_vector, training_product_id)

    doc_cnt = 10
    X = training_X[:doc_cnt * max_sentence]
    y = training_Y[:doc_cnt]
    y = [i - 1 for i in y]
    user_vector_input = user_vector_input[:doc_cnt]
    pro_vector_input = pro_vector_input[:doc_cnt]

    X = np.array(X)
    X = np.reshape(X, (-1, max_word * max_sentence, WORD_DIM, 1))
    n_cat = len(Counter(y))
    y = np.array(y)

    # y = to_categorical(y)

    # The sentence encoder
    text_inputs = Input(shape=(max_word * max_sentence, WORD_DIM))
    mask_input = Masking(0)(text_inputs)
    lstm_1 = Bidirectional(LSTM(MIDDLE_OUTPUT // 2, return_sequences=True))(mask_input)

    cnn_input = Reshape((max_word * max_sentence, WORD_DIM, 1))(mask_input)
    cnn_1 = Conv2D(1, (5, 3), padding='same', activation='relu', strides=1)(cnn_input)
    cnn_2 = Dropout(0.5)(cnn_1)
    cnn_3 = MaxPooling2D(pool_size=(1, 4))(cnn_2)
    cnn_4 = Reshape((max_word * max_sentence, WORD_DIM // 4))(cnn_3)

    c1 = concatenate([lstm_1, cnn_4], axis=2)

    # The input and transformation of the user vectors and the product vectors
    user_vec_input = Input(shape=(USER_VEC_DIM))
    pro_vec_input = Input(shape=(USER_VEC_DIM))
    con_user_vec = UserVectorTransformer(user_vec_input)
    con_pro_vec = ProVectorTransformer(pro_vec_input)

    # Further concatenate user and product vector with sentence output
    fur_user_vec = tf.expand_dims(con_user_vec, 1)
    user_sentence_input = tf.concat([c1, fur_user_vec], axis=1)
    fur_pro_vec = tf.expand_dims(con_pro_vec, 1)
    pro_sentence_input = tf.concat([c1, fur_pro_vec], axis=1)
    # The output of the whole first layer of both sides
    # The dimension now is (None, sentence_number, 2 * MIDDLE_OUTPUT)
    user_sentence_output = SentenceToDocument(max_sentence)(fur_user_vec)
    pro_sentence_output = SentenceToDocument(max_sentence)(fur_pro_vec)

    # The document encoder
    lstm_user_input = Reshape((max_sentence, 2 * MIDDLE_OUTPUT))(user_sentence_output)
    lstm_pro_input = Reshape((max_sentence, 2 * MIDDLE_OUTPUT))(pro_sentence_output)
    lstm_user_1 = Bidirectional(LSTM(DOCUMENT_ENCODER_OUTPUT, return_sequences=True))(lstm_user_input)
    lstm_pro_1 = Bidirectional(LSTM(DOCUMENT_ENCODER_OUTPUT, return_sequences=True))(lstm_pro_input)

    cnn_user_input = Reshape((max_sentence, 2 * MIDDLE_OUTPUT, 1))(user_sentence_output)
    cnn_pro_input = Reshape((max_sentence, 2 * MIDDLE_OUTPUT, 1))(pro_sentence_output)
    cnn_user_1 = Conv2D(1, (5, 3), padding='same', activation='relu', strides=1)(cnn_user_input)
    cnn_pro_1 = Conv2D(1, (5, 3), padding='same', activation='relu', strides=1)(cnn_pro_input)
    cnn_user_2 = Dropout(0.5)(cnn_user_1)
    cnn_pro_2 = Dropout(0.5)(cnn_pro_1)
    cnn_user_3 = MaxPooling2D(pool_size=(1, 2))(cnn_user_2)
    cnn_pro_3 = MaxPooling2D(pool_size=(1, 2))(cnn_pro_2)
    cnn_user_output = Reshape((max_sentence, MIDDLE_OUTPUT))(cnn_user_3)
    cnn_pro_output = Reshape((max_sentence, MIDDLE_OUTPUT))(cnn_pro_3)

    user_con = concatenate([lstm_user_1, cnn_user_output])
    pro_con = concatenate([lstm_pro_1, cnn_pro_output])

    user_final_input = tf.concat([user_con, fur_user_vec], axis=1)
    pro_final_input = tf.concat([pro_con, fur_pro_vec], axis=1)

    # Output of each document with dimension (-1, 2 * DOCUMENT_ENCODER_OUTPUT)
    user_output = DocumentToOutput(max_sentence)(user_final_input)
    pro_output = DocumentToOutput(max_sentence)(pro_final_input)

    final_con = concatenate([user_output, pro_output])
    final_output = Dense(n_cat, activation='softmax')(final_con)

    model = Model([text_inputs, user_vec_input, pro_vec_input], final_output)
    get_document_output = Model([text_inputs, user_vec_input, pro_vec_input], [user_output, pro_output])

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch))

        for step in range(X.shape[0] // BATCH_SIZE):

            x_batch = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            user_vec_batch = user_vector_input[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            pro_vec_batch = pro_vec_input[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            y_batch = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

            # batch_input = [x_batch, user_vec_batch, pro_vec_batch]
            # batch_input_images = tf.convert_to_tensor(batch_input, tf.float32)

            x_batch = tf.convert_to_tensor(x_batch, tf.float32)
            user_vec_batch = tf.convert_to_tensor(user_vec_batch, tf.float32)
            pro_vec_batch = tf.convert_to_tensor(pro_vec_batch, tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x_batch)
                tape.watch(user_vec_batch)
                tape.watch(pro_vec_batch)
                # tape.watch(batch_input_images)
                logits = model([x_batch, user_vec_batch, pro_vec_batch], training=True)
                loss_value = loss_fn(y_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            internal_user_output, internal_pro_output = get_document_output([x_batch, user_vec_batch, pro_vec_batch], training=False)
            UserVectorTransformer.update_memory(internal_user_output, user_vec_batch)
            ProVectorTransformer.update_memory(internal_pro_output, pro_vec_batch)

            # Log every 200 batches.
            if step % 1 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
