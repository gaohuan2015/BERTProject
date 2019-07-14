import fastText.FastText as ff
import csv

#def transfercsv_to_fastText(csv_path,fastText_file):
path =r"D:\PycharmProjects\fastText-fastText-latest-build43\chinese data\re_seg_test.csv"
with open(r'D:\PycharmProjects\fastText-fastText-latest-build43\data_chinese\re_seg_test', 'w',encoding='utf_8') as t:
    with open(path, 'r',encoding='utf_8') as f:
        lines = csv.reader(f,delimiter='\t')
        for line in lines:
            target = line[2]
            content = line[1]
            t.write(content + '\t' + '_label_' + target + '\n')


#训练模型
classifier = ff.train_supervised(r'D:\PycharmProjects\fastText-fastText-latest-build43\data_chinese\re_seg_train',label='_label_')
#储存模型
classifier.save_model(r'D:\PycharmProjects\\fastText-fastText-latest-build43\data_chinese\\fastText_model3')#保存模型
#加载模型
classifier = ff.load_model(r'D:\PycharmProjects\\fastText-fastText-latest-build43\data_chinese\\fastText_model3')
#测试模型
correct = 0
total_count = 0
with open(r'D:\PycharmProjects\fastText-fastText-latest-build43\data_chinese\re_seg_test','r',encoding='utf_8') as t:
     lines = t.readlines()
     total_count = len(lines)
     print(total_count)
     for line in lines:
         txt = line.split('\t')[0]#根据数据间的分隔符切割行数据
         txt = txt.strip('\n')#去掉每行最后的换行符'\n'
         predict = classifier.predict(txt)
         if predict[0][0] == line.split('\t')[1].strip('\n'):
             correct += 1

print("Accuracy:", correct / total_count)
