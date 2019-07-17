import csv
import re

path = "integration_model\data\\atis\label2id.csv"
train = "integration_model\generation\label.txt"
with open(train,'w',encoding='utf_8') as tr:
    with open(path,'r',encoding='utf_8') as f:
        lines = csv.reader(f)
        for line in lines:
            context, label = line[1], line[2].strip('\n')
            context = re.sub(r"\s{2,}", " ", context)
            while context[-1] == ' ' or context[0] == ' ':
                context = context.strip(' ')
            tr.write(context + '\t' + label + '\n')


