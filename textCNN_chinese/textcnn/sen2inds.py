#-*- coding: utf_8 -*-


import random
import get_wordlists


testFile = 'textCNN_chinese\model_save\\test.txt'
trainFile = 'textCNN_chinese\model_save\\train.txt'
testDataVecFile = 'textCNN_chinese\model_save\\testdata_vec.txt'
trainDataVecFile = 'textCNN_chinese\model_save\\traindata_vec.txt'

def read_labelFile(file):
    label_w2n = {}
    label_n2w = {}
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    data = list(filter(None, data))
    for line in data:
        line = line.split('\t')
        name_w = line[0]
        name_n = line[1]
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w

def dic(Original_Path ,path):
    word2ind, ind2word = get_wordlists.get_worddict()
    with open(path, 'w', encoding='utf_8') as p:
        with open(Original_Path, 'r', encoding='utf_8') as f:
            datas = f.readlines()
            datas = list(filter(None, datas))
            for line in datas:
                line = line.split('\t')
                context = line[0]
                label = int(line[1])
                sen2id = [label]
                for word in context:
                    if word.strip() == '':
                        continue
                    else:
                        sen2id.append(word2ind[word])

                if len(sen2id) < 91:
                    sen2id.extend([0] * (90 - len(sen2id) + 1))
                else:
                    sen2id = sen2id[:91]

                for n in sen2id:
                    p.write(str(n) + ',')
                p.write('\n')

dic(trainFile,trainDataVecFile)
dic(testFile,testDataVecFile)