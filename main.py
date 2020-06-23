from data import get_train_input, get_vocabulary
from keras.models import Model
from keras.layers import *
from keras.utils import np_utils
from collections import Counter
import numpy as np
import os
import pickle
import tensorflow as tf

WORD_DIM = 200
MIDDLE_OUTPUT = 50
USER_VEC_DIM = 100
PROD_VEC_DIM = 100
SENTENCE_MEM_DIM = 50
REPRESENTATIVES = 10


'''
Input: The user vector
Output: The concatenated vector
'''
class ItemVectorTransform(Layer, user_memory_vec):

    def __init__(self):
        super(ItemVectorTransform, self).__init__()
        self.user_memory_vec = tf.Variable(
            initial_value=user_memory_vec,
        )

    def call(self, inputs):
        pass


'''
Input: 1) Output of sentence encoder (-1, max_word * max_sentence, 2 * MIDDLE_OUTPUT)
       2) The user vector ()
'''
class SentenceToDocument(Layer):

    def __init__(self, units=SENTENCE_MEM_DIM):
        super(SentenceToDocument, self).__init__()
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
            shape=(self.units, USER_VEC_DIM),
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
        
        return tf.matmul(inputs, self.w) + self.b


class DocumentMemory(Layer):

    def __init__(self, units=SENTENCE_MEM_DIM):
        super(SentenceMemory, self).__init__()
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
            shape=(self.units, USER_VEC_DIM),
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
        
        return tf.matmul(inputs, self.w) + self.b


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

    doc_cnt = 10
    X = training_X[:doc_cnt * max_sentence]
    y = training_Y[:doc_cnt]
    y = [i - 1 for i in y]

    X = np.array(X)
    X = np.reshape(X, (-1, max_word * max_sentence, WORD_DIM))
    n_cat = len(Counter(y))
    y = np.array(y)

    y = np_utils.to_categorical(y)

    text_inputs = Input(shape=(max_word * max_sentence, WORD_DIM))
    mask_input = Masking(0)(text_inputs)
    lstm_1 = Bidirectional(LSTM(MIDDLE_OUTPUT, return_sequences=True))(mask_input)

    cnn_1 = Conv1D(5, 3, padding='same', activation='relu', strides=1)(mask_input)
    cnn_2 = Dropout(0.5)(cnn_1)
    cnn_3 = MaxPooling2D((1, 4))(cnn_2)
    
    c1 = concatenate([lstm_1, cnn_3], axis=2)

    user_vec_input = Input(shape=(USER_VEC_DIM))
    con_user_vec = UserMemory()(user_vec_input)

    # The dimension now is (-1, sentence_number, 2 * MIDDLE_OUTPUT)
    sentence_output = SentenceToDocument()([c1, con_user_vec])

    lstm_2 = Bidirectional(LSTM(MIDDLE_OUTPUT, return_sequences=True))(sentence_output)

    cnn_4 = Conv1D(5, 3, padding='same', activation='relu', strides=1)(sentence_output)
    cnn_5 = Dropout(0.5)(cnn_4)
    cnn_6 = MaxPooling2D((1, 2))(cnn_5)




    output = Dense(n_cat, activation='softmax')(combine_1)

    model = Model(text_input, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X, y, epochs=500, batch_size=30, verbose=2)
