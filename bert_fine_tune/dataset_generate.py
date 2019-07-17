import csv

label2id_dict = {}
label_list = []
with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/train/label', 'r') as f:
    for line in f:
        labelname = line.strip('\n')
        if labelname not in label_list:
            label_list.append(labelname)
with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/test/label', 'r') as f:
    for line in f:
        labelname = line.strip('\n')
        if labelname not in label_list:
            label_list.append(labelname)

with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/valid/label', 'r') as f:
    for line in f:
        labelname = line.strip('\n')
        if labelname not in label_list:
            label_list.append(labelname)
print(len(label_list))
csvFile = open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/train/label2id.tsv', "w")
writer = csv.writer(csvFile, delimiter='\t')
k = 0
for ele in label_list:
    writer.writerow([ele, k])
    k += 1
csvFile.close()


with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/train/label2id.tsv', 'r') as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        labelname, labelid = line[0].split('\t')
        label2id_dict[labelname] = int(labelid)
with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/valid/label', 'r') as f:
    label_list = []
    for line in f:
        line = line.strip('\n')
        label_list.append(line)
with open('/home/user/pycharmprojects/duanxuxiang/pytorch-pretrained-BERT-master/data/atis/valid/seq.in', 'r') as f:
    text_a_list = []
    for line in f:
        line = line.strip('\n')
        text_a_list.append(line)


csvFile = open("valid.tsv", "w")
writer = csv.writer(csvFile,  delimiter='\t')
# writer.writerow(["guid", "text-a", "label"])
if len(label_list) == len(text_a_list):
    for i in range(len(text_a_list)):
        guid = "v-" + str(i)
        text_a = text_a_list[i]
        label = label2id_dict[label_list[i]]
        writer.writerow([guid, text_a, label])
csvFile.close()
