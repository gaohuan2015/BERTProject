import torch
import os
import torch.nn as nn
import numpy as np
import time

from model import textCNN
import sen2inds
import get_wordlists

word2ind, ind2word = get_wordlists.get_worddict()
label_w2n, label_n2w = sen2inds.read_labelFile('textCNN_chinese_CrosssValidation\model_save\label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 100,
    'class_num': len(label_w2n),
    "kernel_num": 100,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

def get_testData(file):
    datas = open(file, 'r').read().split('\n')
    datas = list(filter(None, datas))

    return datas


def parse_net_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]
    
    return label, score


def main():
    #init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = 'textCNN_chinese_enhance\model_save\\19072619_model4_iter_49_loss_0.38.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        print('No weight file!')
        exit()
    print(net)

    # net.cuda()
    net.eval()

    numAll = 0
    numRight = 0
    testData = get_testData('textCNN_chinese_enhance\model_save\\testdata_vec4.txt')
    for data in testData:
        numAll += 1
        data = data.split(',')
        label = int(data[0])
        sentence = np.array([int(x) for x in data[1:91]])
        sentence = torch.from_numpy(sentence)
        predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]
        label_pre, score = parse_net_result(predict)
        if label_pre == label and score > -100:
            numRight += 1
        if numAll % 10 == 0:
            print('Accuracy:{}({}/{})'.format(100 * numRight / numAll, numRight, numAll))


if __name__ == "__main__":
    main()