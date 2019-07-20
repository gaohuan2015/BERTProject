import torch
import os
import torch.nn as nn
import numpy as np
import time

from model import textCNN
import sen2inds
import textCNN_data
import get_wordlists

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

dataLoader_param = {
    'batch_size': 50,
    'shuffle': True,
}


def main():
    #init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = 'textCNN_chinese\model_save\weight.pkl'
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

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.NLLLoss()


    print("training...")
    for epoch in range(100):
        loss_total = 0
        for i, (clas, sentences) in enumerate(dataLoader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor)
            clas = clas.type(torch.LongTensor)
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()
            loss_total += loss
        print('Epoch [{}/{}]:\tLoss:{:.4f}'.format(epoch+1, 100, loss_total))

        torch.save(net.state_dict(), weightFile)
        torch.save(net.state_dict(), "textCNN_chinese\model_save\{}_model_iter_{}_loss_{:.2f}.pkl".format(time.strftime('%y%m%d%H'), epoch, loss_total.item()))  # current is model.pkl
        torch.save(net, 'textCNN_chinese\model_save\\textcnn_model')


if __name__ == "__main__":
    main()