import shutil
import csv

path = "D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\data\\train.csv"
train = "D:\\bert1\BERTProject\BERTProject\\textCNN_chinese\\train.txt"

with open(train, 'w', encoding='utf_8') as tr:
    with open(path, 'r', encoding='utf_8') as f:
        lines = csv.reader(f)
        lines = list(filter(None,lines))
        worddict = {}
        idx = 0
        for line in lines:
            context = line[1]
            w = line[2]
            if not w in worddict:
                worddict[w] = idx
                idx += 1
            tr.write(context + '\t' + str(worddict[w]) + '\n')

















