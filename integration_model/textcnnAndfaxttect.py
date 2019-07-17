import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from fasttest import fasttext_classifier
from textcnn import sen2inds
from textcnn.test import get_testData, parse_net_result
import fastText.FastText as ff
import csv


classifier = ff.load_model('integration_model\\fastText_model')
net = torch.load('integration_model\\textcnn_model')
fasttext_file = 'integration_model\\test'
textcnn_file = 'integration_model\\testdata_vec.txt'

correct = 0
total_count = 0
with open(fasttext_file, 'r') as t:
     lines = t.readlines()
     total_count = len(lines)
     for line in lines:
         txt = line.split('\t')[0]#根据数据间的分隔符切割行数据
         txt = txt.strip('\n')#去掉每行最后的换行符'\n'
         predict = classifier.predict(txt)
         if predict[0][0] == line.split('\t')[1].strip('\n'):
            correct += 1

Accuracy1 = 100 * correct / total_count
print("fasttext_Accuracy:",Accuracy1)

numAll = 0
numRight = 0
testData = get_testData(textcnn_file)
for data in testData:
    numAll += 1
    data = data.split(',')
    label = int(data[0])
    sentence = np.array([int(x) for x in data[1:21]])
    sentence = torch.from_numpy(sentence)
    predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]
    label_pre, score = parse_net_result(predict)
    if label_pre == label and score > -100:
            numRight += 1
Accuracy2 = 100 * numRight / numAll
print("textcnn_Accuracy:",Accuracy2)

print("Accuracy:", (Accuracy1 + Accuracy2)/2)
























