# -*- coding: UTF-8 -*-
import csv
import re
import xlrd
import numpy as np
import pandas as pd



file = r'chin_ dataset.csv'
new_file = r'segment_chin_dataset_{}.csv'
data = pd.read_csv(file, header=None, sep='\t')
indexs = np.arange(len(data))
np.random.shuffle(indexs)
indexs.resize((5, 132))  # 660 = 5 *132

train_ind_list = []
test_ind_list = []
data_index=[]
for i, index in enumerate(indexs):
    with open(new_file.format(i), 'w', encoding='utf-8'):
        sub_data = data.iloc()[index]
        sub_data.to_csv(new_file.format(i), index=False, sep='\t', header=None)
        for i in new_file.format(i)[-5]:
            data_index.append(int(i))
            train_ind_list.append(pd.read_csv(new_file.format(i)))


for i, index in enumerate(indexs):
    if len(test_ind_list)!=0.:
        test_ind_list.clear()
        train_ind_list.append(pd.read_csv(new_file.format(i-1)))
        save_train=pd.DataFrame(train_ind_list)
        save_train.to_csv(r"D:\PycharmProjects\cross_val data\cross_val\data_NO.{}\train.csv".format(i))
        save_test=pd.DataFrame(test_ind_list)
        save_test.to_csv(r"D:\PycharmProjects\cross_val data\cross_val\data_NO.{}\test.csv".format(i))

    for j in new_file.format(i)[-5]:
        if int(j)==data_index[i]:
         test_ind_list.append(pd.read_csv(new_file.format(j)))
         del train_ind_list[i]
print(len(train_ind_list))
print(len(test_ind_list))
