import shutil
import csv

testCsvFile = "data\Chinese\Chinese raw data\\re_seg_test.csv"
testFile = "textCNN_chinese\model_save\\test.txt"
trainCsvFile = "data\Chinese\Chinese raw data\\re_seg_train.csv"
trainFile = "textCNN_chinese\model_save\\train.txt"

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
















