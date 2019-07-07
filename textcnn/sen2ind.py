#-*- coding: utf_8 -*-

import csv
import sys, io
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') #改变标准输出的默认编码

validFile = 'D:/VScode/work/textcnn/train'
wordLabelFile = 'D:/VScode/work/textcnn/wordLabel'
validDataVecFile = 'D:/VScode/work/textcnn/traindata_vec'
maxLen = 20

labelFile = 'D:/VScode/work/textcnn/label'

# def main():
label_w2n = {}
label_n2w = {}
data = open(labelFile,'r').read().split( )
for line in data:
    name_w = line.split('\t')[0]
    name_n = int(line.split('\t')[1])
    label_w2n[name_w] = name_n
    label_n2w[name_n] = name_w
return label_w2n, label_n2w



if __name__ == "__main__":
    main()
