from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


trainDataFile = 'textCNN_chinese\model_save\\traindata_vec.txt'
testDataFile = 'textCNN_chinese\model_save\\testdata_vec.txt'


def get_testdata(file=testDataFile):
    testData = open(testDataFile, 'r', encoding='utf_8').read().split('\n')
    testData = list(filter(None, testData))
    random.shuffle(testData)
    return testData


class textCNN_data(Dataset):
    def __init__(self):
        trainData = open(trainDataFile, 'r', encoding='utf_8').read().split('\n')
        trainData = list(filter(None, trainData))
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        data = self.trainData[idx]
        data = list(filter(None, data.split(',')))
        data = [int(x) for x in data]
        cla = data[0]
        sentence = np.array(data[1:])

        return cla, sentence



def textCNN_dataLoader(param):
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    dataset = textCNN_data()
    cla, sen = dataset.__getitem__(0)

    print(cla)
    print(sen)