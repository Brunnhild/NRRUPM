from data import get_train_input, get_vocabulary
from tf.keras.models import Model
from tf.keras.layers import *
from tf.keras.utils import np_utils
from collections import Counter
import numpy as np
import os
import pickle
import tensorflow as tf

WORD_DIM = 200
MIDDLE_OUTPUT = 50
DOCUMENT_ENCODER_OUTPUT = 25
USER_VEC_DIM = 100
PROD_VEC_DIM = 100
SENTENCE_MEM_DIM = 50
DOCUMENT_MEM_DIM = 50
REPRESENTATIVES = 10


'''
Input: The user vector or product vector
Output: The concatenated column vector of dimension (2 * USER_VEC_DIM, 1)
'''
class ItemVectorTransform(Layer, user_memory_vec):

    def __init__(self):
        super(ItemVectorTransform, self).__init__()
        self.user_memory_vec = tf.Variable(
            initial_value=user_memory_vec,
            trainable=False
        )

    def call(self, inputs):
        u = tf.reshape(inputs, (-1, 1))
        sum_r = sum([tf.exp(tf.matmul(i, u)) for i in self.user_memory_vec])
        r = [tf.exp(tf.matmul(i, u)) for i in self.user_memory_vec]
        u_ = sum([r[idx] * v for idx, v in self.user_memory_vec])
        return tf.stack([u, tf.transpose(u_)])


'''
Input: 1) Output of sentence encoder (-1, max_word * max_sentence, 2 * MIDDLE_OUTPUT)
       2) The concatenated user vector width dimension (2 * USER_VEC_DIM, 1)
Output: Dimension (sentence_number, 2 * MIDDLE_OUTPUT)
'''
class SentenceToDocument(Layer):

    def __init__(self, sentence_number, units=SENTENCE_MEM_DIM):
        super(SentenceToDocument, self).__init__()
        self.sentence_number = sentence_number
        self.units = units

    def build(self, input_shape):
        self.vw = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True
        )
        self.wh = self.add_weight(
            shape=(self.units, input_shape[0][-1]),
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
        sentence_output, user_vec = inputs
        sentences = []
        for i in range(self.sentence_number):
            sentences.append(sentences[i:i+inputs.shape[0]/self.sentence_number])
        sum_e, e = 0, [self.get_e(i, user_vec) for i in sentences]
        sum_e += sum([tf.exp(i) for i in e])
        alpha = [tf.exp(i) / sum_e for i in e]
        s = [v * alpha[idx] for idx, v in sentences]

        return s

    def get_e(self, h, u):
        h = tf.reshape(h, (-1, 1))
        wh = tf.matmul(self.wh, h)
        wu = tf.matmul(self.wu, u)
        tan = tf.tanh(wh + wu + self.bw)
        return tf.matmul(tf.transpose(self.vw), tan)


class DocumentToOutput(Layer):

    def __init__(self, units=DOCUMENT_MEM_DIM):
        super(DocumentToOutput).__init__()
        self.units = units

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass


if __name__ == '__main__':
    # train_input = get_train_input(get_vocabulary())[:1000]
    # print(get_train_input(get_vocabulary()))
    if os.path.exists('input'):
        training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence = pickle.load(open('input', 'rb'))
    else:
        training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence = get_train_input(get_vocabulary())
        pickle.dump((training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence), open('input', 'wb'))
    print('Start processing')
    user_cnt = Counter(training_user_id)
    product_cnt = Counter(training_product_id)
    user_vector = np.random.rand(user_cnt, USER_VEC_DIM)
    product_vector = np.random.rand(product_cnt, USER_VEC_DIM)

    UserVectorTransformer = ItemVectorTransform(user_vector[:REPRESENTATIVES])
    ProVectorTransformer = ItemVectorTransform(product_vector[:REPRESENTATIVES])

    # Prepare the input data
    doc_cnt = 10
    X = training_X[:doc_cnt * max_sentence]
    y = training_Y[:doc_cnt]
    y = [i - 1 for i in y]

    X = np.array(X)
    X = np.reshape(X, (-1, max_word * max_sentence, WORD_DIM))
    n_cat = len(Counter(y))
    y = np.array(y)

    y = np_utils.to_categorical(y)

    # The sentence encoder
    text_inputs = Input(shape=(max_word * max_sentence, WORD_DIM))
    mask_input = Masking(0)(text_inputs)
    lstm_1 = Bidirectional(LSTM(MIDDLE_OUTPUT, return_sequences=True))(mask_input)

    cnn_1 = Conv1D(5, 3, padding='same', activation='relu', strides=1)(mask_input)
    cnn_2 = Dropout(0.5)(cnn_1)
    cnn_3 = MaxPooling2D((1, 4))(cnn_2)
    
    c1 = concatenate([lstm_1, cnn_3], axis=2)

    # The input and transformation of the user vectors and the product vectors
    user_vec_input = Input(shape=(USER_VEC_DIM))
    pro_vec_input = Input(shape=(USER_VEC_DIM))
    con_user_vec = UserVectorTransformer(user_vec_input)
    con_pro_vec = ProVectorTransformer(pro_vec_input)

    # The output of the whole first layer of both sides
    # The dimension now is (-1, sentence_number, 2 * MIDDLE_OUTPUT)
    user_sentence_output = SentenceToDocument()([c1, con_user_vec])
    pro_sentence_output = SentenceToDocument()([c1, con_pro_vec])

    # The document encoder
    lstm_user_1 = Bidirectional(LSTM(DOCUMENT_ENCODER_OUTPUT, return_sequences=True))(user_sentence_output)
    lstm_pro_1 = Bidirectional(LSTM(DOCUMENT_ENCODER_OUTPUT, return_sequences=True))(pro_sentence_output)

    cnn_user_1 = Conv1D(5, 3, padding='same', activation='relu', strides=1)(user_sentence_output)
    cnn_pro_1 = Conv1D(5, 3, padding='same', activation='relu', strides=1)(pro_sentence_output)
    cnn_user_2 = Dropout(0.5)(cnn_user_1)
    cnn_pro_2 = Dropout(0.5)(cnn_pro_1)
    cnn_user_3 = Maxpooling2D((1, 4))(cnn_user_2)
    cnn_pro_3 = Maxpooling2D((1, 4))(cnn_pro_2)

    user_con = concatenate([lstm_user_1, cnn_user_3])
    pro_con = concatenate([lstm_pro_1, cnn_pro_3])

    # Output of each document with dimension (-1, 2 * DOCUMENT_ENCODER_OUTPUT)
    user_output = DocumentToOutput()([user_con, con_user_vec])
    pro_output = DocumentToOutput()([pro_con, con_pro_vec])

    final_con = concatenate([user_output, pro_output])
    final_output = Dense(n_cat, activition='softmax')(final_con)

    model = Model([text_inputs, user_vec_input, pro_vec_input], final_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit([X, user_vector, product_vector], y, epochs=500, batch_size=30, verbose=2)
