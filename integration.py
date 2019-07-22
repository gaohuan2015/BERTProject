import numpy as np
import torch
import os
from textCNN_chinese.textcnn import get_wordlists
import fastText.FastText as ff
from textCNN_chinese.textcnn.test import parse_net_result
from textCNN_chinese.textcnn.model import textCNN
from textCNN_chinese.textcnn import sen2inds

classifier = ff.load_model("fasttext_save\\fastText_model")
testCsvFile = "data\Chinese\Chinese raw data\\re_seg_test.csv"
testFile = 'textCNN_chinese\model_save\\test.txt'
testDataVecFile = 'textCNN_chinese\model_save\\testdata_vec.txt'

word2ind, ind2word = get_wordlists.get_worddict()
label_w2n, label_n2w = sen2inds.read_labelFile('textCNN_chinese\model_save\label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 50,
    'class_num': len(label_w2n),
    "kernel_num": 20,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

datas = open(testFile, 'r', encoding='utf_8').read().split('\n')
datas = list(filter(None, datas))
word2ind, ind2word = get_wordlists.get_worddict()
net = textCNN(textCNN_param)
weightFile = 'textCNN_chinese\model_save\\19071915_model_iter_99_loss_3.03.pkl'
if os.path.exists(weightFile):
    print('load weight')
    net.load_state_dict(torch.load(weightFile))
else:
    print('No weight file!')
    exit()
print(net)

net.eval()

correct = 0
total_count = 0
total_count = len(datas)
for line in datas:
    txt = line.split('\t')[0]
    fasttext_predict = classifier.predict(txt)[1]
    label = line.split('\t')[1]
    label = label
    sen2id = [label]

    for word in txt:
        if word.strip() == '':
            continue
        else:
            sen2id.append(word2ind[word])

    if len(sen2id) < 91:
        sen2id.extend([0] * (90 - len(sen2id) + 1))
    else:
        sen2id = sen2id[:91]

    data = [str(w) for w in sen2id]
    lab = int(data[0])
    sentence = np.array([int(x) for x in data[1:91]])
    sentence = torch.from_numpy(sentence)
    textcnn_predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]
    predict = [(fasttext_predict[i] + textcnn_predict[i])/2 for i in range(len(fasttext_predict))]
    label_pre, score = parse_net_result(predict)

    if label_pre == lab:
        correct += 1

print("Accuracy:", 100 * correct / total_count)




