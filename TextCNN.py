
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BasicModel(nn.Module):
    def __init__(self, vocab_size, number_labels):
        super(BasicModel, self).__init__()
        self.vocab_size = vocab_size
        # self.embedding = BertModel.from_pretrained('bert-base-chinese')
        self.embedding = nn.Embedding(vocab_size, 100)
        kernels = [2, 3, 4, 5]
        kernels_number = [100, 100, 100, 100]
        self.convs = nn.ModuleList([
            nn.Conv2d(1, number, (size, 100), padding=(size - 1, 0))
            for (size, number) in zip(kernels, kernels_number)
        ])
        self.line = nn.Linear(100*4, number_labels)

    def forward(self, input_ids):
        result = self.embedding(input_ids)
        result = result.unsqueeze(1)
        conv_value = [
            F.relu(conv(result)).squeeze(3) for conv in self.convs
        ]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_value]
        x = torch.cat(x, 1)
        x = self.line(x)
        return x


def build_word_dic(path, dic):
    with open(path, 'r', encoding = 'utf_8') as f:
        data = f.readlines()
        for d in data:
            d = d.split('\t')[1].strip()
            for w in d:
                if w.strip() == '':
                    continue
                if w not in dic:
                    dic[w] = len(dic)


def convert_label_to_id(path):
    labels = []
    with open(path, 'r', encoding = 'utf_8') as f:
        data = f.readlines()
        for d in data:
            d = d.split('\t')[2].strip()
            if d.strip() == '':
                continue
            labels.append(int(d))
    return labels


def convert_sentence_to_id(path, dic):
    sentences = []
    with open(path, 'r', encoding = 'utf_8') as f:
        data = f.readlines()
        for d in data:
            sen2id = []
            d = d.split('\t')[1].strip()
            for w in d:
                if w.strip() == '':
                    continue
                else:
                    sen2id.append(dic[w])
            if len(sen2id) < 80:
                length = 80 - len(sen2id)
                sen2id = sen2id + length*[0]
            sentences.append(sen2id)
    return sentences


if __name__ == "__main__":
    setup_seed(20)
    word2idx = {'pad': 0}
    build_word_dic(
        'data/Chinese/Chinese raw data/re_seg_train.csv', word2idx)
    build_word_dic(
        'data/Chinese/Chinese raw data/re_seg_test.csv', word2idx)
    training_data = convert_sentence_to_id(
        'data/Chinese/Chinese raw data/re_seg_train.csv', word2idx)
    label_data = convert_label_to_id(
        'data/Chinese/Chinese raw data/re_seg_train.csv')
    training_data = torch.tensor(training_data, dtype=torch.long)
    label_data = torch.tensor(label_data, dtype=torch.long)
    train_data = TensorDataset(training_data, label_data)
    train_dataloader = DataLoader(train_data, batch_size=8)
    model = BasicModel(len(word2idx)+1, 22)
    loss_func = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for _ in trange(int(100), desc="Epoch"):
        fc_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t for t in batch)
            input_ids, label_ids = batch
            pred = model(input_ids)  
            loss = loss_func(pred.view(-1, 22), label_ids.view(-1))
            fc_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(fc_loss)
    testing_data = convert_sentence_to_id(
        'data/Chinese/Chinese raw data/re_seg_test.csv', word2idx)
    testing_label_data = convert_label_to_id(
        'data/Chinese/Chinese raw data/re_seg_test.csv')
    testing_data = torch.tensor(testing_data, dtype=torch.long)
    testing_label_data = torch.tensor(testing_label_data, dtype=torch.long)
    test_data = TensorDataset(testing_data, testing_label_data)
    test_dataloader = DataLoader(test_data, batch_size=1)
    correct = 0
    total = 0
    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
        total += 1
        batch = tuple(t for t in batch)
        input_ids, label_ids = batch
        pred = model(input_ids)
        idx = torch.argmax(pred, dim=1)
        if idx == label_ids:
            correct += 1
        print(100.0*correct/total)
