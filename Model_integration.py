import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from textCNN_chinese.textcnn import get_wordlists
import fastText.FastText as ff
from textCNN_chinese.textcnn.test import parse_net_result
from textCNN_chinese.textcnn.model import textCNN
from textCNN_chinese.textcnn import sen2inds
import torch
from torch.autograd import Variable
from dmn_atis.DmnModel import DMN
from dmn_atis.DmnLoader import label_to_index,word_to_index,flatten,ATIS_data_load,prepare_sequence,pad_to_fact,getBatch,pad_to_batch
import random
import numpy as np

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)


HIDDEN_SIZE = 80
BATCH_SIZE = 64
NUM_EPISODE = 3

classifier = ff.load_model("D:\VScode\BERTProject-master\\fasttext_save\\fastText_re_cut word_cross_validation_model0")
testCsvFile = "D:\VScode\BERTProject-master\data\cross validation\\re_cut word_cross_validation_test0.csv"
testFile = "textCNN_chinese\model_save\\re_cut word_cross_validation_test0.txt"
testDataVecFile = 'textCNN_chinese\model_save\\re_cut word_cross_validation_testvec0.txt'

train_data=ATIS_data_load("D:\VScode\BERTProject-master\data\cross validation\\re_cut word_cross_validation_train0.csv", encoding = "utf-8")
fact, q, a = list(zip(*train_data))
vocab = ['在','哪','里','办','理']
for lis in fact:
    for seq in lis:
        for word in seq:
            if word not in vocab:
                vocab.append(word)
word2index,index2word = word_to_index(vocab)

labels = []
for label in a:
    for la in label:
        if la not in labels:
            labels.append(la)
label2index,index2label = label_to_index(labels)

label_w2n, label_n2w = sen2inds.read_labelFile('textCNN_chinese\model_save\label.txt')
word2ind, ind2word = get_wordlists.get_worddict()

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 50,
    'class_num': len(label_w2n),
    "kernel_num": 20,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

net = textCNN(textCNN_param)
weightFile1 = 'D:\VScode\BERTProject-master\\textCNN_chinese\model_save\\19080117_re_cut word_cross_validation_model0_iter_99_loss_3.68.pkl'
if os.path.exists(weightFile1):
    print('load weight')
    net.load_state_dict(torch.load(weightFile1))
else:
    print('No weight file!')
    exit()
print(net)
net.eval()

model = DMN(len(word2index), HIDDEN_SIZE, len(label2index))
weight = 'D:\VScode\BERTProject-master\dmn_atis\dmn_atis3.pkl'
if os.path.exists(weight):
    print('load weight')
    model.load_state_dict(torch.load(weight))
else:
    print('No weight file!')
    exit()
print(model)

model.eval()

test_data=ATIS_data_load("D:\VScode\BERTProject-master\data\cross validation\\re_cut word_cross_validation_test0.csv", encoding = "utf-8")

accuracy = 0
for t in test_data:
    txt = t[0][0][:-1]
    sequence = ''.join(txt)
    sequence_ge = ' '.join(str(i) for i in txt)
    fasttext_predict = classifier.predict(sequence_ge)[1]
    labe = t[2][0]
    sen2id = [labe]
    for word in sequence:
        if word.strip() == '':
            continue
        else:
            sen2id.append(word2ind[word])

    if len(sen2id) < 91:
        sen2id.extend([0] * (90 - len(sen2id) + 1))
    else:
        sen2id = sen2id[:91]

    data = [str(w) for w in sen2id]
    sent= np.array([int(x) for x in data[1:91]])
    senten = torch.from_numpy(sent)
    textcnn_predict = net(senten.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]

    for i, fact in enumerate(t[0]):
        t[0][i] = prepare_sequence(fact, word2index).view(1, -1)
    t[1] = prepare_sequence(t[1], word2index).view(1, -1)
    t[2] = prepare_sequence(t[2], label2index).view(1, -1)
    fact, fact_mask = pad_to_fact(t[0], word2index)
    question = t[1]
    question_mask = Variable(torch.ByteTensor(
        [0] * t[1].size(1)), volatile=False).unsqueeze(0)
    answer = t[2].squeeze(0)
    pred = model([fact], [fact_mask], question,
                 question_mask, answer.size(0), NUM_EPISODE)
    pred_dmn = pred.detach().numpy()[0]

    predict = [(fasttext_predict[i] + textcnn_predict[i] +  pred_dmn[i])/3 for i in range(len(fasttext_predict))]

    label_pre, score = parse_net_result(predict)
    z = answer.data.tolist()[0]-1
    if label_pre == z:
        accuracy += 1

print("Accuracy:",100 * accuracy/len(test_data))