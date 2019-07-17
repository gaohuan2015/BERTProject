import torch
import os
import torch.nn as nn
import numpy as np
import time

from textcnn.model import textCNN
from textcnn import sen2inds
from textcnn import textCNN_data
# from textcnn.label import LabelSmoothSoftmaxCE

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
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True,
}


def main():
    #init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = 'integration_model\generation\weight.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        net.init_weight()
    print(net)

    # net.cuda()

    #init dataset
    print('init dataset...')
    dataLoader = textCNN_data.textCNN_dataLoader(dataLoader_param)
    traindata = textCNN_data.get_testdata()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    # criterion = LabelSmoothSoftmaxCE()

    # log = open('D:\pathon\work\integration_model\model_save\log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w',encoding='UTF-8')
    # log.write('epoch step loss\n')
    print("training...")
    for epoch in range(100):
        loss_total = 0
        for i, (clas, sentences) in enumerate(dataLoader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor)
            clas = clas.type(torch.LongTensor)
            # sentences = sentences.type(torch.LongTensor).cuda()
            # clas = clas.type(torch.LongTensor).cuda()
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()
            loss_total += loss
        print('Epoch [{}/{}]:\tLoss:{:.4f}'.format(epoch+1, 100, loss_total))

        torch.save(net.state_dict(), weightFile)
        torch.save(net.state_dict(), "integration_model\model_save\{}_model_iter_{}_loss_{:.2f}.pkl".format(time.strftime('%y%m%d%H'), epoch, loss_total.item()))  # current is model.pkl
        torch.save(net, 'integration_model\generation\\textcnn_model')


if __name__ == "__main__":
    main()