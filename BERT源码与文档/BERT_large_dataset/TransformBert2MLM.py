# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to generate the mask language model bert fine tune
# --------------------------------****************************-----------------------------------------------------
import codecs
import csv


def aaa(input_file):
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        li_exam = []
        csv_reader = csv.reader(f, delimiter='\n')
        for line in csv_reader:
            ind2 = line[0].rindex('\t')
            sentence = line[0][:ind2]
            if list(sentence) == []:
                sentence = ''
            lin = sentence
            li_exam.append(lin)
    if 'train' in input_file:
        filename = '.DataSet/MLM_Train.csv'
    else:
        filename = '.DataSet/MLM_Test.csv'
    with open(filename, 'w', encoding='utf-8') as f:
        for i in li_exam:
            f.write(i + '\n')
    return ''

input_file_bert = 'DataSet/train0628.csv'
bert = aaa(input_file_bert)
input_file_bert = 'DataSet/test0628.csv'
bert = aaa(input_file_bert)





