
import fastText.FastText as ff
import csv


path = "integration_model\data\\atis\\test\\test.csv"
with open('integration_model\generation\\test', 'w', encoding='utf_8') as t:
    with open(path, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            target = line[2]
            content = line[1]
            t.write(content + '\t' + '_label_' + target + '\n')

#训练模型
classifier = ff.train_supervised("integration_model\generation\\train", label='_label_')
#储存模型
classifier.save_model('integration_model\generation\\fastText_model')#保存模型
#加载模型
classifier = ff.load_model('integration_model\generation\\fastText_model')
#测试模型
correct = 0
total_count = 0
with open('integration_model\generation\\test', 'r') as t:
     lines = t.readlines()
     total_count = len(lines)
     print(total_count)
     for line in lines:
         txt = line.split('\t')[0]#根据数据间的分隔符切割行数据
         txt = txt.strip('\n')#去掉每行最后的换行符'\n'
         predict = classifier.predict(txt)
         if predict[0][0] == line.split('\t')[1].strip('\n'):
             correct += 1

print("Accuracy:", 100 * correct / total_count)
