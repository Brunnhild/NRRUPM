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

#
def get_train_input(vocabulary):
    max_len = 0
    with open('yelp-2013/yelp-2013-seg-20-20.train.ss', encoding='UTF-8') as f:
        # 第一次遍历先得到最长句子长度
        for (idx, line) in enumerate(f):
            user_id, product_id, score, comment_str = line.split('\t\t')
            comment_str = comment_str.split(' ')
            len_count = 0
            for i in comment_str:
                if i == '<sssss>':
                    continue
                filter_list = '.\n0123456789!,.'
                flag = False
                for j in filter_list:
                    if j in i:
                        flag = True
                if flag:
                    continue
                len_count += 1
                
            if len_count > max_len:
                max_len = len_count
        #print(max_len)  
            
    with open('yelp-2013/yelp-2013-seg-20-20.train.ss', encoding='UTF-8') as f:
        train_input = []
        to_fill = []
        for i in range(200):
            to_fill.append(float(0))
        # output = open("word_matrix.txt", "w")
        for (idx, line) in enumerate(f):
            user_id, product_id, score, comment_str = line.split('\t\t')
            
            score = int(score)
            comment = []
            comment_str = comment_str.split(' ')
            for i in comment_str:
                if i == '<sssss>':
                    continue
                if vocabulary.__contains__(i):
                    comment.append(vocabulary[i])
            for i in range(max_len - len(comment)):
                comment.append(to_fill)
            train_input.append((user_id, product_id, score, comment))
    return train_input
