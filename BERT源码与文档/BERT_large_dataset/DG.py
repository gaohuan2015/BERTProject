# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to generate the test dataset from excel
# --------------------------------****************************-----------------------------------------------------
import xlrd

import json


def data_generate():
    with open('DataSet/label2id_large.json', 'r', encoding='utf-8') as f:
        label2idx1 = json.load(f)
        with open('test.csv', 'w', encoding='utf-8') as f:
            f.write('sentence' + '\t' + 'label' + '\n')
            workbook = xlrd.open_workbook('test.xlsx')
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
                f.write(sentsence + '\t' + str(labelid) + '\n')
    return ""


if __name__ == "__main__":
    data_generate()