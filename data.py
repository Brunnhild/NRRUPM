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
    max_word = 0
    max_sentence = 0
    lengest_str = ""
    for i, line in enumerate(file):
        user_id, product_id, score, comment_str = line.split('\t\t')
        comment_sentences = comment_str.split('<sssss>')
        max_sentence = max(max_sentence, len(comment_sentences))
        for sentence in comment_sentences:
            #print(sentence)
            sentence = filter(lambda e: not filtering(e, stop_words), sentence.split(' '))
            max_word = max(max_word, len(list(sentence)))
    print('The longest sentence is %d' % (max_word))
    print('The longest document is %d' % (max_sentence))

    # 将每句话都构造成词向量矩阵
    training_X = []
    training_Y = []
    training_user_id = []
    training_product_id = []
    filling_word = [float(0) for i in range(wordVec_dim)]
    filling_sentence = [filling_word for i in range(max_word)]

    file = open(file_path, encoding = "utf-8")

    for i, line in enumerate(file):
        user_id, product_id, score, comment_str = line.split('\t\t')
        training_user_id.append(user_id)
        training_product_id.append(product_id)
        training_Y.append(int(score))
        comment_sentences = comment_str.split('<sssss>')
        document = []
        for sentence in comment_sentences:
            sentence_matrix = []
            sentence_words = filter(lambda e: not filtering(e, stop_words), sentence.split(' '))
            for word in list(sentence_words):
                if vocabulary.__contains__(word):
                    sentence_matrix.append(vocabulary[word])
            for i in range(max_word - len(sentence_matrix)):
                sentence_matrix.append(filling_word)
            document.append(sentence_matrix)
        for i in range(max_sentence - len(comment_sentences)):
            document.append(filling_sentence)
        training_X.append(document)
    return training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence


def do_get_test_input(stop_words, vocabulary, max_word, max_sentence):
    file_path = "yelp-2013/yelp-2013-seg-20-20.test.ss"
    file = open(file_path, encoding = "utf-8")

    # 将每句话都构造成词向量矩阵
    training_X = []
    training_Y = []
    training_user_id = []
    training_product_id = []
    filling_word = [float(0) for i in range(wordVec_dim)]
    filling_sentence = [filling_word for i in range(max_word)]

    file = open(file_path, encoding = "utf-8")

    for i, line in enumerate(file):
        user_id, product_id, score, comment_str = line.split('\t\t')
        training_user_id.append(user_id)
        training_product_id.append(product_id)
        training_Y.append(int(score))
        comment_sentences = comment_str.split('<sssss>')
        document = []
        for sentence in comment_sentences:
            sentence_matrix = []
            sentence_words = filter(lambda e: not filtering(e, stop_words), sentence.split(' '))
            for word in list(sentence_words):
                if vocabulary.__contains__(word):
                    sentence_matrix.append(vocabulary[word])
            for i in range(max_word - len(sentence_matrix)):
                sentence_matrix.append(filling_word)
            document.append(sentence_matrix)
        for i in range(max_sentence - len(comment_sentences)):
            document.append(filling_sentence)
        training_X.append(document)
    return training_X, training_Y, training_user_id, training_product_id


def get_train_input(vocabulary):
    #获取停用词
    stop_path = "./stop_words.txt"
    stop_file = open(stop_path, encoding = "utf-8")
    stop_words = [e for e in enumerate(stop_file)]
    training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence = get_sentence(stop_words, vocabulary)
    return training_X, training_Y, training_user_id, training_product_id, max_word, max_sentence


def get_test_input(vocabulary, max_word, max_sentence):
    #获取停用词
    stop_path = "./stop_words.txt"
    stop_file = open(stop_path, encoding = "utf-8")
    stop_words = [e for e in enumerate(stop_file)]
    return do_get_test_input(stop_words, vocabulary, max_word, max_sentence)