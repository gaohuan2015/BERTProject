# -*- coding: utf-8 -*-
import jieba

testFile = 'textCNN_chinese_enhance\model_save\\test4.txt'
trainFile = 'textCNN_chinese_enhance\model_save\\train4.txt'
wordList = 'textCNN_chinese_enhance\model_save\\wordList4.txt'

def word_to_idx(path,dic):
    with open(path, 'r', encoding = 'utf_8') as f:
        data = f.readlines()
        for line in data:
            line = line.split('\t')[0].split(" ")
            for word in line:
                for i in word:
                    if i not in dic:
                        dic[i] = len(dic)

def get_worddict():
    word2ind = {}
    train_idx = word_to_idx(trainFile, word2ind)
    test_idx = word_to_idx(testFile, word2ind)

    ind2word = {word2ind[w]:w for w in word2ind}
    
    return word2ind, ind2word

