from data import get_train_input, get_vocabulary, get_test_input
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
REPRESENTATIVES = 10
EPOCHS = 5
BATCH_SIZE = 30
MEMORY_UPDATE_CORE_UNITS = 1


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
        res = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in range(self.user_memory_vec.shape[0]):
            m = tf.reshape(self.user_memory_vec[i], (-1, 1)) # shape: (USER_VEC_DIM, 1)
            wuu = tf.matmul(u_batch, tf.transpose(self.wu)) # shape: (batch, units)
            wmm = tf.matmul(self.wm, m) # shape: (units, 1)
            wdd = tf.matmul(d_batch, tf.transpose(self.wd)) # shape: (batch, units)
            wmm = tf.tile(tf.transpose(wmm), (u_batch.shape[0], 1)) # shape: (batch, units)
            bg = tf.tile(tf.transpose(self.bg), (u_batch.shape[0], 1)) #shape: (batch, units)
            g_batch = tf.math.sigmoid(wuu + wmm + wdd + bg) # shape: (batch, units)
            g_batch = tf.reshape(g_batch, (-1,))
            m = tf.reshape(m, (-1,))
            for i in range(g_batch.shape[0]):
                g = g_batch[i]
                m = g * m + (1 - g) * u_batch[i]
            res = res.write(res.size(), m)
        self.user_memory_vec = res.stack()
    
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
            shape=(self.units, 4 * DOCUMENT_ENCODER_OUTPUT),
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
            gammar = tf.nn.softmax(mu, axis=0) # shape: (rep, 1)
            gammar = tf.tile(gammar, (1, u.shape[0])) # shape: (rep, USER_VEC_DIM)
            gm = gammar * self.user_memory_vec
            u_ = tf.reduce_sum(gm, axis=0)
            res = res.write(res.size(), tf.concat([u, u_], axis=0))
        return res.stack()

    def compute_output_shape(self, input_shape):
        return (None, 2 * input_shape[-1])


'''
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
                alpha = tf.nn.softmax(e, axis=0) # shape: (section_size, 1)
                s = tf.tile(alpha, (1, h.shape[1])) * h # shape: (section_size, 2 * MIDDLE_OUTPUT)
                s = tf.reduce_sum(s, axis=0) #shape: (2 * MIDDLE_OUTPUT,)
                section_s = section_s.write(i, s)
            section_s = section_s.stack() # shape: (self.sentence_number, 2 * MIDDLE_OUTPUT)
            res = res.write(res.size(), section_s)
        return res.stack()

    def compute_output_shape(self, input_shape):
        return (None, self.sentence_number, 2 * MIDDLE_OUTPUT)

'''
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
            beta = tf.nn.softmax(e, axis=0) # shape: (self.sentence_number, 1)
            d = tf.tile(beta, (1, document_output.shape[1])) * document_output # shape: (self.sentence_number, 2 * MIDDLE_OUTPUT)
            d = tf.reduce_sum(d, axis=0)
            res = res.write(res.size(), d)
        return res.stack()


def transform_id(item_vec, item_id, item_map):
    item_vector_input, idx = [], 0
    for i in item_id:
        if item_map.__contains__(i):
            item_vector_input.append(item_vec[item_map[i]])
        else:
            item_map[i] = idx
            item_vector_input.append(item_vec[idx])
            idx += 1
    return np.reshape(item_vector_input, (-1, USER_VEC_DIM))


def get_test_id(item_vec, item_id, item_map):
    item_vector_input = []
    for i in item_id:
        item_vector_input.append(item_vec[item_map[i]])
    return np.reshape(item_vector_input, (-1, USER_VEC_DIM))


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)
    max_sentence = 40
    max_word = 50
    if os.path.exists('input'):
        training_X, training_Y, training_user_id, training_product_id = pickle.load(open('input', 'rb'))
    else:
        training_X, training_Y, training_user_id, training_product_id = get_train_input(get_vocabulary(), max_word, max_sentence)
        pickle.dump((training_X, training_Y, training_user_id, training_product_id), open('input', 'wb'))
    if os.path.exists('test_input'):
        test_X, test_Y, test_user_id, test_product_id = pickle.load(open('test_input', 'rb'))
    else:
        test_X, test_Y, test_user_id, test_product_id = get_test_input(get_vocabulary(), max_word, max_sentence)
        pickle.dump((test_X, test_Y, test_user_id, test_product_id), open('test_input', 'wb'))
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
    user_id_mapper = dict()
    pro_id_mapper = dict()
    user_vector_input = transform_id(user_vector, training_user_id, user_id_mapper)
    pro_vector_input = transform_id(product_vector, training_product_id, pro_id_mapper)
    test_user_id = get_test_id(user_vector, test_user_id, user_id_mapper)
    test_product_id = get_test_id(product_vector, test_product_id, pro_id_mapper)

    doc_cnt = len(training_X)
    X = training_X[:doc_cnt]
    y = training_Y[:doc_cnt]
    y = [i - 1 for i in y]
    user_vector_input = user_vector_input[:doc_cnt]
    pro_vector_input = pro_vector_input[:doc_cnt]

    # X = np.array(X)
    # X = np.reshape(X, (-1, max_word * max_sentence, WORD_DIM, 1))
    n_cat = len(Counter(y))
    y = np.array(y)

    '''
    The size of the input is 
    X: (None, max_word, max_sentence, WORD_DIM)
    y: (None,) of type int
    user_vector_input: (None, USER_VEC_DIM)
    pro_vector_input: (None, USER_VEC_DIM)
    '''

    # y = to_categorical(y)

    # The sentence encoder
    text_inputs = Input(shape=(max_sentence, max_word, WORD_DIM))
    lstm_input = Reshape((max_word * max_sentence, WORD_DIM))(text_inputs)
    mask_input = Masking(0)(lstm_input)
    lstm_1 = Bidirectional(LSTM(MIDDLE_OUTPUT // 2, return_sequences=True))(mask_input)

    cnn_input = Reshape((max_word * max_sentence, WORD_DIM, 1))(text_inputs)
    cnn_1 = Conv2D(1, (5, 3), padding='same', activation='relu', strides=1)(cnn_input)
    cnn_2 = Dropout(0.5)(cnn_1)
    cnn_3 = MaxPooling2D(pool_size=(1, 4))(cnn_2)
    cnn_4 = Reshape((max_word * max_sentence, WORD_DIM // 4))(cnn_3)

    c1 = concatenate([lstm_1, cnn_4], axis=2)

    # The input and transformation of the user vectors and the product vectors
    user_vec_input = Input(shape=(USER_VEC_DIM,))
    pro_vec_input = Input(shape=(USER_VEC_DIM,))
    con_user_vec = UserVectorTransformer(user_vec_input)
    con_pro_vec = ProVectorTransformer(pro_vec_input)

    # Further concatenate user and product vector with sentence output
    fur_user_vec = tf.expand_dims(con_user_vec, 1)
    user_sentence_input = tf.concat([c1, fur_user_vec], axis=1)
    fur_pro_vec = tf.expand_dims(con_pro_vec, 1)
    pro_sentence_input = tf.concat([c1, fur_pro_vec], axis=1)
    # The output of the whole first layer of both sides
    # The dimension now is (None, sentence_number, 2 * MIDDLE_OUTPUT)
    user_sentence_output = SentenceToDocument(max_sentence)(user_sentence_input)
    pro_sentence_output = SentenceToDocument(max_sentence)(pro_sentence_input)

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

    # examine_model = Model([text_inputs, user_vec_input, pro_vec_input], cnn_user_output)
    model = Model([text_inputs, user_vec_input, pro_vec_input], final_output)
    get_document_output = Model([text_inputs, user_vec_input, pro_vec_input], [user_output, pro_output])

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    me = keras.metrics.Accuracy()

    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch))

        for step in range(len(X) // BATCH_SIZE):

            x_batch = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            user_vec_batch = user_vector_input[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            pro_vec_batch = pro_vector_input[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            y_batch = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

            x_batch = tf.convert_to_tensor(x_batch, tf.float32)
            user_vec_batch = tf.convert_to_tensor(user_vec_batch, tf.float32)
            pro_vec_batch = tf.convert_to_tensor(pro_vec_batch, tf.float32)

            if step % 20 == 22:
                test_cat = tf.constant([], dtype=tf.int64)
                num_of_bat = len(test_X) // BATCH_SIZE + 1
                last_batch = len(test_X) - (len(test_X) // BATCH_SIZE) * BATCH_SIZE
                if last_batch == 0:
                    num_of_bat -= 1
                for i in range(num_of_bat):
                    bs = BATCH_SIZE
                    if i == num_of_bat - 1 and last_batch != 0:
                        bs = last_batch
                    ll = model([tf.reshape(test_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], (bs, max_sentence, max_word, WORD_DIM)), 
                                tf.reshape(test_user_id[i*BATCH_SIZE:(i+1)*BATCH_SIZE], (bs, -1)), 
                                tf.reshape(test_product_id[i*BATCH_SIZE:(i+1)*BATCH_SIZE], (bs, -1))], training=False)
                    test_cat = tf.concat([test_cat, tf.argmax(ll, axis=1) + 1], axis=0)
                me.reset_states()
                _ = me.update_state(test_Y, test_cat)
                test_acc = me.result().numpy()
                print('The test acc: %.2f' % (test_acc))
                f = open('epoch-%d-step-%d-acc-%.2f' % (epoch, step, test_acc), 'w')
                f.write(str(test_cat.numpy()))
                f.close()

            with tf.GradientTape() as tape:
                tape.watch(x_batch)
                tape.watch(user_vec_batch)
                tape.watch(pro_vec_batch)
                # exa = examine_model([x_batch, user_vec_batch, pro_vec_batch], training=False)
                internal_user_output, internal_pro_output = get_document_output([x_batch, user_vec_batch, pro_vec_batch], training=False)
                UserVectorTransformer.update_memory(internal_user_output, user_vec_batch)
                ProVectorTransformer.update_memory(internal_pro_output, pro_vec_batch)

                logits = model([x_batch, user_vec_batch, pro_vec_batch], training=True)
                me.reset_states()
                _ = me.update_state(y_batch, tf.argmax(logits, axis=1))
                acc = me.result().numpy()
                loss_value = loss_fn(y_batch, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)

            # test_cat = []
            # for i in range(len(test_X)):
            #     ll = model([test_X[i], test_user_id[i], test_product_id[i]], training=True)
            #     tf.print(ll)
            #     test_cat.append(tf.argmax(ll))
            # me.reset_states()
            # test_acc = me.update_state(test_Y, test_cat)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 1 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f and the accuracy is: %.2f"
                    % (step, float(loss_value), acc)
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
