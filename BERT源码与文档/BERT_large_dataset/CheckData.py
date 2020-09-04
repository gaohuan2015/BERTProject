# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is the code to check whether the test data items in train data
# --------------------------------****************************-----------------------------------------------------
import xlrd
import csv
import json
import codecs


def data_check_excel():
    train_tuple_dict = {}
    with codecs.open('DataSet/train0628.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\n')
        for line in csv_reader:
            ind2 = line[0].rindex('\t')
            sentence = line[0][:ind2]
            lab = line[0][ind2 + 1:]
            if lab == 'label':
                continue
            if list(sentence) == []:
                sentence = ''
            train_tuple_dict[sentence] = lab
    traincomp_tuple_dict = {}
    with open('DataSet/label2id_large.json', 'r', encoding='utf-8') as f:
        label2idx1 = json.load(f)
        with open('traincomp.csv', 'w', encoding='utf-8') as f:
            f.write('sentence' + '\t' + 'label' + '\n')
            workbook = xlrd.open_workbook('traincomp.xlsx')
            sheet = workbook.sheet_by_name('Sheet1')
            rows = sheet.nrows
            for i in range(1, rows):
                line = sheet.row_values(i)
                sentsence = line[0]
                for i in range(len(sentsence)):
                    if sentsence[i] == '\t':
                        sentsence = sentsence[:i] + ' ' + sentsence[i+1:]
                label = line[1]
                labelid = label2idx1[label]
                traincomp_tuple_dict[sentsence] = labelid
                f.write(sentsence + '\t' + str(labelid) + '\n')
    j = 0
    for i in train_tuple_dict.keys():
        if i not in traincomp_tuple_dict.keys():
            print(i, train_tuple_dict[i])
            print('the inconsistent data totally:', j)
            j += 1


def data_check_csv():
    train_tuple_dict = {}
    with codecs.open('DataSet/train0628.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\n')
        for line in csv_reader:
            ind2 = line[0].rindex('\t')
            sentence = line[0][:ind2]
            lab = line[0][ind2 + 1:]
            if lab == 'label':
                continue
            if list(sentence) == []:
                sentence = ''
            train_tuple_dict[sentence] = lab
    traincomp_tuple_dict = {}
    with codecs.open('traincomp.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\n')
            for line in csv_reader:
                ind2 = line[0].rindex('\t')
                sentence = line[0][:ind2]
                lab = line[0][ind2 + 1:]
                if lab == 'label':
                    continue
                if list(sentence) == []:
                    sentence = ''
                traincomp_tuple_dict[sentence] = lab
    j = 0
    for i in train_tuple_dict.keys():
        if i not in traincomp_tuple_dict.keys():
            print(i, train_tuple_dict[i])
            print('the inconsistent data totally:', j)
            j += 1

if __name__ == "__main__":
    # data_check_excel()
    data_check_csv()