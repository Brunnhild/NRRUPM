from data import get_train_input, get_vocabulary
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, Dropout, GlobalMaxPool1D
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

    model = Sequential()
    model.add(Conv1D(5, 3, padding='valid', activation='relu', strides=1))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPool1D())
    model.add(Dense(100, activation='relu'))

    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=500, batch_size=30, verbose=2)
