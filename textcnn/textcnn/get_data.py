import csv

path = "BERTProject/data/atis/train/train.csv"
train = "textcnn/train.txt"
with open(train,'w') as tr:
    with open(path,'r') as f:
        lines = csv.reader(f)
        for line in lines:
            target = line[2]
            context = line[1]
            tr.write(context + '\t' + target + '\n')


