import csv

path = "D:/VScode/work/BERTProject/data/atis/valid/valid.csv"
train = "D:/VScode/work/textcnn/valid"
with open(train,'w') as tr:
    with open(path,'r') as f:
        lines = csv.reader(f)
        for line in lines:
            target = line[2]
            context = line[1]
            tr.write(context + '\t' + '__label__' + target + '\n')

