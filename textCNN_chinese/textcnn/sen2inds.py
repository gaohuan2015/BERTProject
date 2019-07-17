#-*- coding: utf_8 -*-

import json
import jieba
import random



validFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\test.txt'
stopwordFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\stopword.txt'
wordLabelFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\wordLabel.txt'
validDataVecFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\testdata_vec.txt'
maxLen = 20

labelFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\label.txt'
def read_labelFile(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    data = list(filter(None, data))
    label_w2n = {}
    label_n2w = {}
    for line in data:
        line = line.split('\t')
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    ind2word = {}
    for line in datas:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])
    
    ind2word = {word2ind[w]:w for w in word2ind}

    return word2ind, ind2word


def dic():
    label_n2w, label_w2n = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)

    traindataTxt = open(validDataVecFile, 'w', encoding='utf_8')
    stoplist = read_stopword(stopwordFile)
    datas = open(validFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    # random.shuffle(datas)
    for line in datas:
        line = line.split('\t')
        title = line[0]
        # cla = int(line[1])
        cla_ind = int(line[1])

        # title_seg = jieba.cut(title, cut_all=False)
        title_ind = [cla_ind]
        for w in title:
            if w in stoplist:
                continue
            title_ind.append(word2ind[w])
        length = len(title_ind)
        if length > maxLen + 1:
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    dic()


if __name__ == "__main__":
    main()


