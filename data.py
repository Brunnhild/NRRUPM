import numpy as np

wordVec_dim = 200

def filtering(word, stop_words):
    filter_list = '.\n0123456789!,.' #去掉特殊符号
    flag = False
    for j in filter_list:
        if j in word:
            flag = True
    if flag:
        return True
                
    if word in stop_words:
        print(word)
        return True

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

def get_sentence(stop_words, vocabulary):
    file_path = "yelp-2013/yelp-2013-seg-20-20.train.ss"
    file = open(file_path, encoding = "utf-8")

    #得到最长的句子
    max_len = 0
    lengest_str = ""
    for i, line in enumerate(file):
        user_id, product_id, score, comment_str = line.split('\t\t')
        comment_sentences = comment_str.split('<sssss>')
        for sentence in comment_sentences:
            #print(sentence)
            temp_len = 0
            sentence_words = sentence.split(' ')
            for word in sentence_words:
                if filtering(word, stop_words):
                    continue
                temp_len += 1
            if temp_len > max_len:
                max_len = temp_len
                lengest_str = sentence
    print(max_len)

    file = open(file_path, encoding = "utf-8")
    # 将每句话都构造成词向量矩阵
    train_input = []
    to_fill = []
    for i in range(wordVec_dim):
        to_fill.append(float(0))
            
    for i, line in enumerate(file):
        user_id, product_id, score, comment_str = line.split('\t\t')
        comment_sentences = comment_str.split('<sssss>')
        for sentence in comment_sentences:
            sentence_matrix = []
            sentence_words = sentence.split(' ')
            for word in sentence_words:
                if filtering(word, stop_words):
                    continue
                if vocabulary.__contains__(word):
                    sentence_matrix.append(vocabulary[word])
            for i in range(max_len - len(sentence_matrix)):
                sentence_matrix.append(to_fill)
            train_input.append((user_id, product_id, score, sentence_matrix))
    return train_input
            
#
def get_train_input(vocabulary): #
    #获取停用词
    stop_path = "./stop_words.txt"
    stop_file = open(stop_path, encoding = "utf-8")
    stop_words = []
    for i, word in enumerate(stop_file):
        stop_words.append(word)
    train_input = get_sentence(stop_words, vocabulary)
    '''    
    max_len = 0    
    with open('yelp-2013/yelp-2013-seg-20-20.train.ss', encoding='UTF-8') as f:
        # 第一次遍历先得到最长句子长度
        for (idx, line) in enumerate(f):
            user_id, product_id, score, comment_str = line.split('\t\t')
            comment_str = comment_str.split(' ')
            len_count = 0
            for i in comment_str:
                if i == '<sssss>': #去掉句子结尾符
                    continue
                filter_list = '.\n0123456789!,.' #去掉特殊符号
                flag = False
                for j in filter_list:
                    if j in i:
                        flag = True
                if flag:
                    continue
                
                if i in stop_words:
                    print(i)
                    continue
                
                len_count += 1
                
            if len_count > max_len:
                max_len = len_count
        print(max_len)  
            
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
    '''
    return train_input
    
