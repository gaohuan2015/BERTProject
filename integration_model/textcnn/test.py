import torch
import os
import torch.nn as nn
import numpy as np
import time

from textcnn.model import textCNN
from textcnn import sen2inds

word2ind, ind2word = sen2inds.get_worddict('integration_model\generation\wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('integration_model\generation\label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}


def get_testData(file):
    datas = open(file, 'r',encoding='UTF-8').read().split('\n')
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
    weightFile = 'integration_model\model_save\\19071719_model_iter_91_34_loss_0.11.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile,map_location='cpu'))
    else:
        print('No weight file!')
        exit()
    print(net)

    # net.cuda()
    net.eval()

    numAll = 0
    numRight = 0
    testData = get_testData('integration_model\generation\\testdata_vec.txt')
    for data in testData:
        numAll += 1
        data = data.split(',')
        label = int(data[0])
        sentence = np.array([int(x) for x in data[1:21]])
        sentence = torch.from_numpy(sentence)
        predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]
        label_pre, score = parse_net_result(predict)
        if label_pre == label and score > -100:
            numRight += 1
        if numAll % 70 == 0:    
            print('Accuracy:{}({}/{})'.format(numRight / numAll, numRight, numAll))


if __name__ == "__main__":
    main()