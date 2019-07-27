import shutil
import csv

train__CsvFile = "enhance_train_data\enhance_train4.csv"
train_CsvFile = "data\cross validation\cross_validation_train4.csv"

testCsvFile = "data\cross validation\cross_validation_test4.csv"
trainCsvFile = "textCNN_chinese_enhance\model_save\\train4.csv"

testFile = "textCNN_chinese_enhance\model_save\\test4.txt"
trainFile = "textCNN_chinese_enhance\model_save\\train4.txt"

file_1 = open(train__CsvFile, 'r', encoding = 'utf_8')
file_2 = open(train_CsvFile, 'r', encoding = 'utf_8')

list1 = []
for line in file_1.readlines():
    ss = line.strip()
    list1.append(ss)
file_1.close()

list2 = []
for line in file_2.readlines():
    ss = line.strip()
    list2.append(ss)
file_2.close()

file_new = open(trainCsvFile,'w', encoding = 'utf_8')
for i in range(len(list1)):
    sline = list1[i]
    file_new.write(sline + '\n')
for i in range(len(list2)):
    sline = list2[i]
    file_new.write(sline + '\n')
file_new.close()


def get_data(path, Opath):
    with open(Opath, 'w', encoding='utf_8') as tr:
        with open(path, 'r', encoding='utf_8') as f:
            lines = csv.reader(f)
            for line in lines:
                line = line[0]
                context = line.split('\t')[1]
                label = line.split('\t')[2]
                tr.write(context + '\t' + label + '\n')

get_data(trainCsvFile, trainFile)
get_data(testCsvFile, testFile)
















