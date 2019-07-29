# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import csv
result_file=r'.\Excess data.csv'
word_list =[]
word_set = []
with open(r'.\cross_validation_train0.csv','r',encoding='utf_8') as f:
    lines = f.readlines()
    for line in lines:
        for w in line:
            word_list.append(w)
with open(r'.\cross_validation_test0.csv','r',encoding='utf_8') as f:
    lines = f.readlines()
    for line in lines:
        for w in line:
            word_set.append(w)
word_set = set(word_set)
oov= 0
file_new=open(result_file,'w',encoding='utf_8')
for w in word_set:
    if w not in word_list:
        oov+=1
        file_new.write(w + '\n')
file_new.close()
print(100.0*oov/len(word_set))
print(len(word_set))
print(len(set(word_list)))





