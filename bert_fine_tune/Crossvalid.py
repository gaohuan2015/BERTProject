import os
import re
import csv
import xlrd
import torch
import numpy as np
# ########################### K折交叉验证 ##################################
# 定义K，即交叉验证的折数
k = 5
# 读取数据
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
excelFile = xlrd.open_workbook(r'\chinese_data\small_samp.xlsx')
sheet = excelFile.sheet_by_index(0)
problem_tmp = sheet.col_values(0)
problem = []
for pro in problem_tmp:
    pro = re.sub(reg, '', str(pro))
    problem.append(pro)
label = sheet.col_values(1)
label_dict = {}
i = 0
for label_ in label:
    if label_ not in label_dict.keys():
        label_dict[label_] = i
        i += 1
if len(problem) == len(label):
    # 将数据分成k份，即确定步长step
    dataid = np.arange(len(problem))
    np.random.shuffle(dataid)
    step = int(len(problem) / k)
    for zhe in range(k):
        test_id_list = dataid[zhe * step:(zhe + 1) * step]
        if not os.path.exists("Crossval/train_NO.{}/".format(zhe)):
            os.makedirs("Crossval/train_NO.{}/".format(zhe))
        f = open("Crossval/train_NO.{}/test.tsv".format(zhe), "w", encoding='utf-8')
        csvwriter = csv.writer(f, delimiter='\t')
        seg_id = 0
        for i in test_id_list:
            segment_id = 'te-' + str(seg_id)
            csvwriter.writerow([segment_id, str(problem[i]), str(label_dict[label[i]])])
            seg_id += 1
        f.close()
        train_id_list = []
        for i in range(len(problem)):
            if i not in test_id_list:
                train_id_list.append(i)
        f = open("Crossval/train_NO.{}/train.tsv".format(zhe), "w", encoding='utf-8')
        csvwriter = csv.writer(f, delimiter='\t')
        seg_id = 0
        for i in train_id_list:
            segment_id = 'tr-' + str(seg_id)
            csvwriter.writerow([segment_id, str(problem[i]), str(label_dict[label[i]])])
            seg_id += 1
        f.close()
        f = open("Crossval/train_NO.{}/label2id.tsv".format(zhe), "w", encoding='utf-8')
        csvwriter = csv.writer(f, delimiter='\t')
        for name, id in label_dict.items():
            csvwriter.writerow([name, str(id)])
        f.close()
