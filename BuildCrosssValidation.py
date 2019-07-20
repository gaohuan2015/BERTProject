import xlrd
import uuid
import copy
import numpy as np
label2idx = {}
sentences_catagory = {}
data_set = {}


def readExcel(path, sheetname):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name(sheetname)
    rows = sheet.nrows
    for i in range(0, rows):
        line = sheet.row_values(i)
        sentence = line[0]
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
            l2id = label2idx[label]
            t = tuple((guid, s, l2id))
            segement_sentences.append(t)
            id = id+1
        data_set[label] = segement_sentences


def write_dataset_to_file(path, nfold):
    for i in range(nfold):
        train = []
        test = []
        for label in data_set:
            segement_sentences = data_set[label]
            data_copy = copy.deepcopy(segement_sentences)
            data_size = len(segement_sentences)
            number = int(data_size/nfold)
            for didx in range(data_size):
                if didx >= i*number and didx <= (i+1)*number-1:
                    test.append(data_copy[didx])
                else:
                    train.append(data_copy[didx])
        train_path = path+'_train'+str(i)+'.csv'
        test_path = path+'_test'+str(i)+'.csv'
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
    readExcel('BERTProject/data/Chinese/小样本训练语料.xlsx', 'Sheet1')
    buildDataSet('other')
    write_dataset_to_file('cross_validation', 5)
