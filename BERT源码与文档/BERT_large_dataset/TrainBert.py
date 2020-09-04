# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to train bert series model
# --------------------------------****************************-----------------------------------------------------
import torch
import torch.nn as nn
import pandas as pd
import codecs
import csv
import numpy as np
import random
import argparse
import torch.optim as optim
from transformers import BertModel, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from apex import amp
import time


class SSTDataset(Dataset):

    def __init__(self, tokerpath, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokerpath)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        # if type(sentence) is not str:
        #     sentence = ''
        label = self.df.loc[index, 'label']

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        # Insering the CLS and SEP token in the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            # Padding sentences
            tokens = tokens + \
                ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            # Prunning the list to be of specified max length
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids)

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, attn_mask, label


class SentimentClassifier(nn.Module):

    def __init__(self, modelpath, freeze_bert=True, num_labels=2):
        super(SentimentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = AutoModel.from_pretrained(modelpath)
        # self.bert_layer = AutoModel.from_pretrained('bert-base-chinese')
        # self.bert_layer.init_weights()
        self.num_labels = num_labels
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.drop = torch.nn.Dropout(0.1)
        self.cls_layer = nn.Linear(768, self.num_labels)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        # cont_reps, _ = self.bert_layer(seq, attention_mask=attn_masks)
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        cont_reps = cont_reps[0]
        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(self.drop(cls_rep))
        # logits = self.cls_layer(cls_rep)
        # logits = self.drop(self.cls_layer(cls_rep))

        return logits


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        #  CCE
        ce = self.cross_entropy(pred, labels)
        #  RCE
        # pred = torch.softmax(pred, dim = 1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        #  Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


def get_accuracy_from_logits(logits, labels):
    accall = 0
    probs = logits.unsqueeze(-1)
    soft_probs = torch.argmax(probs, 1).view(1, -1).long()
    acc = (soft_probs.squeeze() == labels).float().tolist()
    for a in acc:
        accall += a
    return accall, accall / len(labels)


def evaluate(net, criterion, dataloader, num_labels):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, labels) in enumerate(dataloader):
            seq, attn_masks, labels = seq.cuda(device), attn_masks.cuda(device), labels.cuda(device)
            logits = net(seq, attn_masks)
            # mean_loss += criterion(logits.view(-1, num_labels), labels.view(-1)).item()
            mean_acc += get_accuracy_from_logits(logits, labels)[0]
            count += 1
    return mean_acc / 1000, mean_loss / 1000


def train(args, net, criterion, opti, scheduler, train_loader, val_loader, num_labels):
    start = time.clock()
    for ep in range(args.train_epochs):

        best_acc = 0
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            net.train()
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(device), attn_masks.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)
            # Computing loss
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))

            # Backpropagating the gradients
            # loss.backward()
            with amp.scale_loss(loss, opti) as scaled_loss:
                scaled_loss.backward()
            # Optimization step
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            opti.step()
            scheduler.step()
            opti.zero_grad()

            if (it + 1) % 10 == 0:
                acc, accbatch = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(
                    it+1, ep+1, loss.item(), accbatch))

        val_acc, val_loss = evaluate(net, criterion, val_loader, num_labels)
        end = time.clock()
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}, Time Cost : {}".format(
            ep, val_acc, val_loss, end - start))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(
                best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(),
                       'Models/sstcls_{}_freeze_{}.dat'.format(ep, False))


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
            # ind1 = line[0].index('\t')
            ind2 = line[0].rindex('\t')
            # sentence = line[0][ind1+1:ind2]
            sentence = line[0][:ind2]
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
    parser.add_argument("--train_batch_size", default=64, type=int, help="训练时batch大小")
    parser.add_argument("--eval_batch_size", default=200, type=int, help="验证时batch大小")
    parser.add_argument("--train_epochs", default=80, type=float, help="训练的epochs次数")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Adam初始学习步长")
    parser.add_argument("--bert_model", default="./PretrianedModel/bert_base_chinese", type=str, help="选择bert模型的类型")
    parser.add_argument("--train_data_dir", type=str, help="训练数据读入的路径",
                        default="DataSet/train0628.csv")
    parser.add_argument("--test_data_dir", type=str, help="测试数据读入的路径",
                        default="DataSet/test0628.csv")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:' + str(args.device_id))
    # Creating instances of training and validation set
    labels = create_examples(read_csv(args.train_data_dir))
    train_set = SSTDataset(tokerpath=args.bert_model, filename=args.train_data_dir, maxlen=args.max_seq_length)
    val_set = SSTDataset(tokerpath=args.bert_model, filename=args.test_data_dir, maxlen=args.max_seq_length)
    # Creating intsances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.train_batch_size, num_workers=1)
    net = SentimentClassifier(modelpath=args.bert_model, freeze_bert=False, num_labels=len(labels))
    net.cuda(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = SCELoss(alpha=0.7, beta=0.3, num_classes=len(labels))
    # criterion = LabelSmoothSoftmaxCE()
    # opti = optim.AdamW(net.parameters(), lr=2e-5)
    opti = AdamW(net.parameters(), lr=args.learning_rate, correct_bias=True)
    train_steps = int(len(train_set) / args.train_batch_size * args.train_epochs)
    scheduler = get_linear_schedule_with_warmup(opti, num_warmup_steps=0.1, num_training_steps=train_steps)
    net, opti = amp.initialize(net, opti)

    train(args, net, criterion, opti, scheduler, train_loader, val_loader, num_labels=len(labels))




