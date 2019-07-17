# -*- coding: utf-8 -*-
'''
将训练数据使用jieba分词工具进行分词。并且剔除stopList中的词。
得到词表：
        词表的每一行的内容为：词 词的序号 词的频次
'''


import json
import jieba
from tqdm import tqdm

validFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\test.txt'
trainFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\train.txt'
resultFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\trainTest.txt'
stopwordFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\stopword.txt'
wordLabelFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\wordLabel.txt'
lengthFile = 'D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\length.txt'

file_1 = open(trainFile,'r',encoding='UTF-8')
file_2 = open(validFile,'r',encoding='UTF-8')

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

def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def main():
    worddict = {}
    stoplist = read_stopword(stopwordFile)
    datas = open(resultFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    data_num = len(datas)
    len_dic = {}
    for line in datas:
        title = line.split('\t')[0]
        # title_seg = jieba.cut(title, cut_all=False)
        length = 0
        for w in title:
            if w in stoplist:
                continue
            length += 1
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1
        if length in len_dic:
            len_dic[length] += 1
        else:
            len_dic[length] = 1

    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='utf_8')
    ind = 0
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

    for k, v in len_dic.items():
        len_dic[k] = round(v * 1.0 / data_num, 3)
    len_list = sorted(len_dic.items(), key=lambda item:item[0], reverse=True)
    f = open(lengthFile, 'w')
    for t in len_list:
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)

if __name__ == "__main__":
    main()