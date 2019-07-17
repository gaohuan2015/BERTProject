#-*- coding: utf_8 -*-

import csv
import random

trainFile = 'integration_model\generation\\test.txt'
wordLabelFile = 'integration_model\generation\wordLabel.txt'
trainDataVecFile = 'integration_model\generation\\testdata_vec.txt'
maxLen = 20

labelFile = 'integration_model\generation\label.txt'
def read_labelFile(file):
    label_w2n = {}
    label_n2w = {}
    data = open(file, 'r', encoding='UTF-8').read().split('\n')
    data = list(filter(None, data))
    for line in data:
        line = line.split('\t')
        name_w = line[0]
        name_n = line[1]
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w

def get_worddict(file):
    word2ind = {}
    ind2word = {}
    data = open(file,'r',encoding='UTF-8').read().split('\n')
    data = list(filter(None, data))
    for line in data:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])
    
    ind2word = {word2ind[w]:w for w in word2ind}

    return word2ind, ind2word

def dic():
    word2ind, ind2word = get_worddict(wordLabelFile)

    traindataTxt = open(trainDataVecFile, 'w',encoding='UTF-8')
    datas = open(trainFile, 'r', encoding='UTF-8').read().split('\n')
    datas = list(filter(None, datas))
    for line in datas:
        line = line.split('\t')
        context = line[0]
        cla_ind = list(line[1])
        for w in context.split(' '):
            cla_ind.append(word2ind[w])
        length = len(cla_ind)

        if length > maxLen + 1:
            cla_ind = cla_ind[0:21]

        if length < maxLen + 1:
            cla_ind.extend([0] * (maxLen - length + 1))

        for n in cla_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')

def main():
    dic()

if __name__ == "__main__":
    main()
