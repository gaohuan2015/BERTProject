# -*- coding: UTF-8 -*-
import csv
import re
import torch
import xlrd
import numpy as np
SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
excelFile = xlrd.open_workbook(r'F:\py programmes\programmes\bert_fine_tune\chinese_data\small_samp.xlsx')
sheet = excelFile.sheet_by_index(0)
problem_tmp = sheet.col_values(0)
problem = []
for pro in problem_tmp:
    tmp = str(pro)
    pro = re.sub(reg, '', tmp)
    problem.append(pro)
label = sheet.col_values(1)
label_dict = {}
i = 0
for label_ in label:
    if label_ not in label_dict.keys():
        label_dict[label_] = i
        i += 1

if len(problem) == len(label):
    ran_size = int(len(problem) * 0.2)
    test_ind_list = np.random.randint(low=0, high=len(problem), size=ran_size)
train_ind_list = []
for i in range(len(problem)):
    if i not in test_ind_list:
        train_ind_list.append(i)

f = open("./label2id.tsv", "w", encoding='utf-8')
csvwriter = csv.writer(f, delimiter='\t')
for name, id in label_dict.items():
    csvwriter.writerow([name, str(id)])
f.close()

f = open("./test.tsv", "w", encoding='utf-8')
csvwriter = csv.writer(f, delimiter='\t')
seg_id = 0
for i in test_ind_list:
    segment_id = 'te-' + str(seg_id)
    csvwriter.writerow([segment_id, str(problem[i]), str(label_dict[label[i]])])
    seg_id += 1
f.close()

f = open("./train.tsv", "w", encoding='utf-8')
csvwriter = csv.writer(f, delimiter='\t')
seg_id = 0

for i in train_ind_list:
    segment_id = 'tr-' + str(seg_id)
    pro = str(problem[i])
    pro = re.sub(reg, '', pro)
    csvwriter.writerow([segment_id, str(problem[i]), str(label_dict[label[i]])])
    seg_id += 1
f.close()
