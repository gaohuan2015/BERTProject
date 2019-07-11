# -*- coding: UTF-8 -*-
import csv
import xlrd
import numpy as np

ExcelFile=xlrd.open_workbook(r'D:\PycharmProjects\deal date\小样本训练语料.xlsx')
sheet=ExcelFile.sheet_by_index(0)
problem=sheet.col_values(0)
label=sheet.col_values(1)
if len(problem) == len(label):
    ran_size = int(len(problem) * 0.2)
    test_ind_list =  np.random.randint(low=0, high=len(problem),size=ran_size)
train_ind_list = []
for i in range(len(problem)):
    if i not in test_ind_list:
        train_ind_list.append(i)
f = open("./test.csv", "w", encoding='utf-8')
csvwriter= csv.writer(f, delimiter='\t')
seg_id = 0
for i in test_ind_list:
    segment_id = 'te-' + str(seg_id)
    csvwriter.writerow([segment_id, str(problem[i]), str(label[i])])
    seg_id += 1
f.close()
f = open("./train.csv", "w", encoding='utf-8')
csvwriter = csv.writer(f, delimiter='\t')
seg_id = 0
for i in train_ind_list:
    segment_id = 'tr-' + str(seg_id)
    csvwriter.writerow([segment_id, str(problem[i]), str(label[i])])
    seg_id += 1
f.close()