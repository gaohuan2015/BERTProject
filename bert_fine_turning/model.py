import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn


class Classifier_base_model(nn.Module):
    def __init__(self,  bert, bert_output_size=768, num_labels=26):
        super(Classifier_base_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        pred = self.lin(cls_outputs)
        return pred


class Classifier_pad_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(Classifier_pad_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size * 2, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        pad_outputs = hiddens_outputs[-1][:, -1, :]
        outs = torch.cat((pad_outputs, cls_outputs), dim=1)
        pred = self.lin(outs)
        return pred


class Classifier_sep_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(Classifier_sep_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size * 2, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        sep_outputs = self.sep_search(hiddens_outputs[-1], input_mask)
        outs = torch.cat((sep_outputs, cls_outputs), dim=1)
        pred = self.lin(outs)
        return pred

    def sep_search(self, hidden_output, input_mask):
        if int(hidden_output.size()[0]) == int(input_mask.size()[0]):
            sep_ids = input_mask.sum(dim=1).tolist()
            sep_outputs = hidden_output[0, sep_ids[0], :].view(1, int(hidden_output.size()[2])).cuda()
            for seq in range(1, int(hidden_output.size()[0])):
                sep_id = sep_ids[seq]
                sep_output = hidden_output[seq, sep_id, :].view(1, int(hidden_output.size()[2])).cuda()
                sep_outputs = torch.cat((sep_outputs, sep_output), dim=0)
            return sep_outputs


class TextCNN(nn.Module):
    def __init__(self, vocab_size, bert, number_labels):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = bert
        # self.embedding = nn.Embedding(vocab_size, 100)
        kernels = [2, 3, 4, 5]
        kernels_number = [768, 768, 768, 768]
        self.convs = nn.ModuleList([
            nn.Conv2d(1, number, (size, 768), padding=(size - 1, 0))
            for (size, number) in zip(kernels, kernels_number)
        ])
        self.line = nn.Linear(768*4, number_labels)

    def forward(self,  input_ids, segment_ids, input_mask):
        result, cls_outputs = self.embedding(input_ids, segment_ids, input_mask,output_all_encoded_layers=False)
        result = result.unsqueeze(1)
        conv_value = [
            F.relu(conv(result)).squeeze(3) for conv in self.convs
        ]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_value]
        x = torch.cat(x, 1)
        x = self.line(x)
        return x


class Classifier_joint_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=22):
        super(Classifier_joint_model, self).__init__()
        self.bert_base = Classifier_base_model(bert, bert_output_size, num_labels)
        self.textCNN = TextCNN(22000, bert, num_labels)
        self.bert_pad = Classifier_pad_model(bert, bert_output_size, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, segment_ids, input_mask):
        pred_base = self.bert_base(input_ids, segment_ids, input_mask)
        pred_pad = self.bert_pad(input_ids, segment_ids, input_mask)
        pre_textcnn = self.textCNN(input_ids, segment_ids, input_mask)
        return self.dropout(pred_base), pred_pad, pre_textcnn