#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xlrd
import re
import copy
import json
from random import shuffle
label2idx = {}
sentences_catagory = {}
data_set = {}
reg = "['*' , '\t''']"


def readExcel(path, sheetname):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name(sheetname)
    rows = sheet.nrows
    for i in range(1, rows):
        line = sheet.row_values(i)
        # sentence = line[0]
        problem = line[0]
        sentence = re.sub(reg, '', problem)
        label = line[1]
        if label not in label2idx:
            label2idx[label] = len(label2idx)
            l_sentences = []
            l_sentences.append(sentence)
            sentences_catagory[label] = l_sentences
        else:
            l_sentences = sentences_catagory[label]
            l_sentences.append(sentence)
            sentences_catagory[label] = l_sentences


def buildDataSet(mode):
    id = 0
    f = open('./label2id.json', 'r', encoding='utf-8')
    label2idx1 = json.load(f)
    f.close()
    print(label2idx1)
    for label in sentences_catagory:
        snetences = sentences_catagory[label]
        segement_sentences = []
        for sentence in snetences:
            guid = str(id)
            s = ''
            if mode == 'char':
                for w in sentence:
                    s = s + w + ' '
                s = s.strip()
            else:
                s = sentence
            if label not in label2idx1:
                label2idx1[label] = len(label2idx1)
            l2id = label2idx1[label]
            t = tuple((guid, s, l2id))
            segement_sentences.append(t)
            id = id+1
        data_set[label] = segement_sentences
    print(len(label2idx1))


def write_dataset_to_file(path):
    test = []
    for label in data_set:
        segement_sentences = data_set[label]
        data_copy = copy.deepcopy(segement_sentences)
        data_size = len(segement_sentences)
        for didx in range(data_size):
            test.append(data_copy[didx])
    # shuffle(train)
    test_path = path + 'train_all' + '.txt'
    with open(test_path, 'w') as f:
        for d in test:
            id, sentence, label2id = d
            f.write('train'+str(id)+'\t'+str(sentence) +
                    '\t'+str(label2id)+'\n')


if __name__ == "__main__":
    readExcel('/home/llv19/PycharmProjects/duanxx/BertProjectDataset/3/大样本数据集.xlsx', 'Sheet1')
    buildDataSet('other')
    write_dataset_to_file('./data/big_sample/')
