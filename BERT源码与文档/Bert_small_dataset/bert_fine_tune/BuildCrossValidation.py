# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to generate the cross validation dataset
# --------------------------------****************************-----------------------------------------------------

import xlrd
import uuid
import re
import copy
import numpy as np
import json
from random import shuffle
label2idx = {}
sentences_catagory = {}
data_set = {}
label2idx1 = {}
workbook = xlrd.open_workbook('BertProjectDataset/2.xlsx')  # da shujuji bioage lujing
sheet = workbook.sheet_by_name('Sheet1')
rows = sheet.nrows
for i in range(0, rows):
    line = sheet.row_values(i)
    sentence = line[0]
    label = line[1]
    if label not in label2idx1:
        label2idx1[label] = len(label2idx1)
print(label2idx1)
f = open('label2id_large.json', 'w', encoding='utf-8')
json.dump(label2idx1,fp=f,ensure_ascii=False,)

reg = "['*' , '\t''']"


# read the excel file and return a dictionary that keys are the label id values are the sentence belong to the specific
# key
def readExcel(path, sheetname):
    """
    :param path: the file input path
    :param sheetname: the sheetname of the excel
    :return: None
    """
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


# bulid the total dataset
def buildDataSet(mode):
    """
    :param mode:the patamerter can select in "char" or "other"
    :return:None
    """
    id = 0
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


# bulid the cross validation dataset according to the rate of 0.2 so that write the dataset to the file
def write_dataset_to_file(path, nfold):
    """
    :param path: path of the folder wrote in
    :param nfold: the k of the k-folder
    :return: None
    """
    for i in range(nfold):
        train = []
        test = []
        for label in data_set:
            segement_sentences = data_set[label]
            data_copy = copy.deepcopy(segement_sentences)
            data_size = len(segement_sentences)
            print(data_size)
            number = int(data_size * 0.2)
            random_sampling = np.random.randint(0, data_size, number)
            for didx in range(data_size):
                if didx in random_sampling:
                    test.append(data_copy[didx])
                else:
                    train.append(data_copy[didx])
        # shuffle(train)
        train_path = path+'_train44'+str(1)+'.csv'
        test_path = path+'_test44'+str(1)+'.csv'
        with open(train_path, 'w') as f:
            for d in range(len(train)):
                id, sentence, label2id = train[d]
                f.write('train'+str(id)+'\t'+str(sentence) +
                        '\t'+str(label2id)+'\n')
        with open(test_path, 'w') as f:
            for d in test:
                id, sentence, label2id = d
                f.write('train'+str(id)+'\t'+str(sentence) +
                        '\t'+str(label2id)+'\n')


if __name__ == "__main__":
    readExcel('BertProjectDataset/2.xlsx', 'Sheet1')
    buildDataSet('other')
    # write_dataset_to_file('/home/llv19/PycharmProjects/duanxx/', 1)
