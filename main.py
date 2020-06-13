from data import get_train_input, get_vocabulary
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, Dropout, GlobalMaxPool1D, Input, concatenate
from keras.utils import np_utils
import numpy as np


if __name__ == '__main__':

    train_input = get_train_input(get_vocabulary())[:1000]
    # print(get_train_input(get_vocabulary()))
    X = []
    y = []
    cnt = 0
    for (user_id, product_id, score, comment) in train_input:
        if len(comment) != 1034:
            cnt += 1
            continue
        X.append(comment)
        y.append(score)
    print(cnt)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], len(X[0]), 200))
    y = np.array(y)
    y = np_utils.to_categorical(y)
    # model = Sequential()
    # model.add(Bidirectional(LSTM(32, input_shape=(1034, 200))))
    # model.add(Dense(6, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X, y, epochs=500, batch_size=30, verbose=2)

    text_input = Input(shape=(1034, 200))

    lstm_1 = LSTM(32)(text_input)
    lstm_2 = Dense(6)(lstm_1)

    cnn_1 = Conv1D(5, 3, padding='valid', activation='relu', strides=1)(text_input)
    cnn_2 = Dropout(0.5)(cnn_1)
    cnn_3 = GlobalMaxPool1D()(cnn_2)
    cnn_4 = Dense(100, activation='relu')(cnn_3)
    cnn_5 = Dense(20, activation='relu')(cnn_4)
    cnn_6 = Dropout(0.5)(cnn_5)
    cnn_7 = Dense(6)(cnn_6)

    combine = concatenate([lstm_2, cnn_7], axis=-1)
    output = Dense(6, activation='softmax')(combine)

    model = Model(text_input, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=500, batch_size=30, verbose=2)
