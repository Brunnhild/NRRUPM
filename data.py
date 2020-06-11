import numpy as np


def get_vocabulary():
    vocabulary = dict()
    with open('./yelp-2013/yelp-2013-embedding-200d.txt', encoding='UTF-8') as f:
        for (idx, line) in enumerate(f):
            if idx == 0:
                continue
            fi = line.find(' ')
            vector = []
            for n in line[fi + 1:].split(' '):
                try:
                    vector.append(float(n))
                except Exception:
                    pass
            assert len(vector) == 200
            if line[:fi] == '</s>':
                vocabulary['<sssss>'] = vector
            else:
                vocabulary[line[:fi]] = vector
    return vocabulary


def get_train_input(vocabulary):
    train_input = []
    with open('yelp-2013/yelp-2013-seg-20-20.train.ss', encoding='UTF-8') as f:
        for (idx, line) in enumerate(f):
            user_id, product_id, score, comment_str = line.split('\t\t')
            score = int(score)
            comment = []
            comment_str = comment_str.split(' ')
            for i in comment_str:
                try:
                    comment.append(vocabulary[i])
                except Exception:
                    pass
            train_input.append((user_id, product_id, score, comment))
    return train_input