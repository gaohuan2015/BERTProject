import fastText.FastText as ff
import csv


def transfercsv_to_fastText(csv_path,fastText_file):
    path = "D:\PycharmProjects\\fastText-fastText-latest-build43\pytorch-pretrained-BERT-master\data\snips\\test\\test.csv"
    with open('D:\PycharmProjects\\fastText-fastText-latest-build43\Date\snips\\test', 'w') as t:
        with open(path, 'r') as f:
            lines = csv.reader(f)
            for line in lines:
                target = line[2]
                content = line[1]
                t.write(content + '\t' + '_label_' + target + '\n')


#训练模型
classifier = ff.train_supervised('D:\PycharmProjects\\fastText-fastText-latest-build43\Date\snips\\train',label='_label_')
#储存模型
classifier.save_model('D:\PycharmProjects\\fastText-fastText-latest-build43\Date\snips\\fastText_model1')#保存模型
#加载模型
classifier = ff.load_model('D:\PycharmProjects\\fastText-fastText-latest-build43\Date\snips\\fastText_model1')
#测试模型
correct = 0
total_count = 0
with open('D:\PycharmProjects\\fastText-fastText-latest-build43\Date\\snips\\test','r') as t:
     lines = t.readlines()l
     total_count = len(lines)
     print(total_count)
     for line  in lines:
         txt = line.split('\t')[0]
         txt = txt.strip('\n')
         predict = classifier.predict(txt)
         if predict[0][0] == line.split('\t')[1].strip('\n'):
             correct += 1

print("Accuracy:", correct / total_count)
