# -*- coding: utf-8 -*-
import csv
import os, sys
from tqdm import tqdm

trainFile = 'integration_model\generation\train.txt'
testFile = 'integration_model\generation\\test.txt'
wordLabelFile = 'integration_model\generation\wordLabel.txt'
resultFile = 'integration_model\generation\\trainTest.txt'
lengthFile = 'integration_model\generation\length.txt'

file_1 = open(trainFile,'r',encoding='UTF-8')
file_2 = open(testFile,'r',encoding='UTF-8')

list1 = []
for line in file_1.readlines():
    ss = line.strip()
    list1.append(ss)
file_1.close()

list2 = []
for line in file_2.readlines():
    ss = line.strip()
    list2.append(ss)
file_2.close()

file_new = open(resultFile,'w',encoding='UTF-8')
for i in range(len(list1)):
    sline = list1[i]
    file_new.write(sline + '\n')
for i in range(len(list2)):
    sline = list2[i]
    file_new.write(sline + '\n')
file_new.close()


def main():
    worddict = {}
    datas = open(resultFile,'r',encoding='UTF-8').read().split('\n')
    datas = list(filter(None,datas))
    data_num = len(datas)
    for line in datas:
        txt = line.split('\t')[0]
        txt = txt.split()
        for w in txt:  
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1

    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='UTF-8')
    ind = 0
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)


if __name__ == "__main__":
    main()