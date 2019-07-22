# -*- coding: utf-8 -*-


testFile = 'textCNN_chinese\\model_save\\test.txt'
trainFile = 'textCNN_chinese\\model_save\\train.txt'
wordList = 'textCNN_chinese\\model_save\\wordList.txt'

def word_to_idx(path,dic):
    with open(path, 'r', encoding = 'utf_8') as f:
        data = f.readlines()
        for line in data:
            line = line.split('\t')[0].split(" ")
            for word in line:
                if word not in dic:
                    dic[word] = len(dic)

def get_worddict():
    word2ind = {}
    train_idx = word_to_idx(trainFile, word2ind)
    test_idx = word_to_idx(testFile, word2ind)

    ind2word = {word2ind[w]:w for w in word2ind}
    
    return word2ind, ind2word

