# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to test bert series model
# --------------------------------****************************-----------------------------------------------------
import torch
import torch.nn as nn
import pandas as pd
import codecs
import csv
import numpy as np
import argparse
import random
from transformers import BertModel, AutoModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SSTDataset(Dataset):

    def __init__(self, tokerpath, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        self.tokenizer = BertTokenizer.from_pretrained(tokerpath)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentencesss = []
        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        if type(sentence) is not str:
            sentence = ''
        sentencesss.append(sentence)
        label = self.df.loc[index, 'label']
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long()
        return sentence, tokens_ids_tensor, attn_mask, label


class SentimentClassifier(nn.Module):

    def __init__(self, modelpath, shape, freeze_bert=True, num_labels=2):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(modelpath)
        self.num_labels = num_labels
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.drop = torch.nn.Dropout(0.1)
        self.cls_layer = nn.Linear(shape, self.num_labels)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        cont_reps = cont_reps[0]
        cls_rep = cont_reps[:, 0]
        # logits = self.cls_layer(cls_rep)
        logits = self.cls_layer(self.drop(cls_rep))
        return logits


def evaluate(net, val_set):

    val_loader = DataLoader(val_set, batch_size=1, num_workers=1)
    net.eval()
    mean_acc, mean_loss = 0, 0
    corre = 0
    with torch.no_grad():
        with open('error.csv', 'w', encoding='utf-8') as w:
            for it, (sentence, seq, attn_masks, labels) in enumerate(val_loader):
                seq, attn_masks, labels = seq.cuda(device), attn_masks.cuda(device), labels.cuda(device)
                logits = net(seq, attn_masks)
                probs = logits.unsqueeze(-1)
                soft_probs = torch.argmax(probs, 1).view(1, -1).long()
                pred, lab = soft_probs.squeeze().tolist(), labels.tolist()[0]
                if pred == lab:
                    corre += 1
                else:
                    w.write(sentence[0] + "\t" + str(pred) + "\t" + str(lab) + "\n")
    return corre / it, mean_loss / it


def create_examples(li_exam):
    labels = []
    for index, ele in enumerate(li_exam):
        if index == 0:
            continue
        label = int(ele[1])
        labels.append(label)
    return set(labels)


def read_csv(input_file):
    li_exam = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\n')
        for line in csv_reader:
            ind1 = line[0].index('\t')
            ind2 = line[0].rindex('\t')
            sentence = line[0][ind1+1:ind2]
            lab = line[0][ind2+1:]
            if list(sentence) == []:
                sentence = ''
            lin = (sentence, lab)
            li_exam.append(lin)
    return li_exam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
    parser.add_argument("--max_seq_length", default=40, type=int, help="字符串最大长度")
    parser.add_argument("--device_id", default=1, type=int, help="gpu id")
    parser.add_argument("--bert_model", default="./PretrianedModel/bert_base_chinese", type=str, help="选择bert模型的类型")
    parser.add_argument("--train_data_dir", type=str, help="训练数据读入的路径",
                        default="DataSet/train0628.csv")
    parser.add_argument("--test_data_dir", type=str, help="测试数据读入的路径",
                        default="test.csv")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:' + str(args.device_id))
    labels = create_examples(read_csv(args.train_data_dir))

    val_set = SSTDataset(tokerpath=args.bert_model, filename=args.test_data_dir, maxlen=args.max_seq_length)

    net = SentimentClassifier(modelpath=args.bert_model, shape=768, freeze_bert=False, num_labels=len(labels))
    net = net.to(device)

    model_save_pth = 'Models/Bert-chinese_871.dat'
    state_dict = torch.load(model_save_pth, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    net.to(device)

    val_acc, val_loss = evaluate(net, val_set)
    print("*****************************************")
    print("the test accuracy is:", val_acc)
    print("*****************************************")




