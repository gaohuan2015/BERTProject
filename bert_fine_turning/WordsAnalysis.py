# -*- coding: utf-8 -*-
import csv


def get_words(filepath):
    words = []
    sentences = []
    f = csv.reader(open(filepath, 'r', encoding='utf-8'), delimiter='\t')
    for line in f:
        sentences.append(line[1])
        for w in line[1].split(' '):
            words.append(w)
    return set(words)


trainpath = 'F:\git_repo\BERTProject\data\enhance_train data\enhance_cut word_train3.csv'
testpath = 'F:\git_repo\BERTProject\data\cross validation\cut word_cross_validation_test3.csv'
trainwords = get_words(trainpath)
testwords = get_words(testpath)
oov = 0
f = open('./result.txt', 'w', encoding='utf_8')
for w in testwords:
    if w not in trainwords:
        print(w)
        oov += 1
        f.write(w + '\n')
f.close()
print(oov)
print(100.0 * oov / len(testwords))
print(len(testwords))
print(len(set(trainwords)))