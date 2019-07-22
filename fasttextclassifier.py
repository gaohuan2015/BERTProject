import fasttext
import csv
import numpy as np
np.random.seed(5)


def transfercsv_to_fasttext(csv_path, fasttext_file):
    path = "D:/NLPTool/BERTProject/data/atis/test/test.csv"
    with open('test', 'w') as t:
        with open(path, 'r') as f:
            lines = csv.reader(f)
            for line in lines:
                target = line[2]
                content = line[1]
                t.write(content + '\t' + '__label__' + target + '\n')


# 训练模型
classifier = fasttext.train_supervised(
    'BERTProject\data\char_cross_validation_train4.csv',
    label_prefix="__label__",
    lr=0.00001,
    dim=100,
    ws=5,
    neg=10,
    wordNgrams=2,
    minCount=1,
    lrUpdateRate=10,
    epoch=700)
# 存储模型
classifier.save_model('fasttext_model')  # 保存模型

#加载模型
classifier = fasttext.load_model('fasttext_model')
#测试模型
correct = 0
total_count = 0
with open(
        'BERTProject\data\char_cross_validation_test4.csv', 'r',
        encoding='utf-8') as t:
    lines = t.readlines()
    total_count = len(lines)
    for l in lines:
        txt = l.split('\t')[0]
        txt = txt.strip('\n')
        predict = classifier.predict(txt)
        if predict[0][0] == l.split('\t')[1].strip('\n'):
            correct = correct + 1

print("准确率:", 100.0 * correct / total_count)
