import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()

        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        #         print('resnet:{}'.format(x.shape))
        return x


class DPCNN(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(DPCNN, self).__init__()
        self.channel_size = 250
        self.seq_len = 100
        self.word_embeddings = bert
        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(bert_output_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

        resnet_block_list = []
        while self.seq_len > 2:
            resnet_block_list.append(ResnetBlock(self.channel_size))
            self.seq_len = self.seq_len // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.linear_out = nn.Linear(
            self.seq_len * self.channel_size, num_labels
        )  # 改成输出一个值

    def forward(self, input_ids, segment_ids, input_mask):
        batch = input_ids.shape[0]
        result, cls_outputs = self.word_embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        # Region embedding
        x = result.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size(0), -1)
        out = self.linear_out(x)
        return out

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)
        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        # Short Cut
        x = x + px
        return x


class TextRNN(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(TextRNN, self).__init__()
        self.word_embeddings = bert
        self.lstm = nn.LSTM(768, 100, bidirectional=True, dropout=0.5)
        self.label = nn.Linear(200, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.word_embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        input = result.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda())
        c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        return self.label(output[:, -1])


class RCNN(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(RCNN, self).__init__()
        self.word_embeddings = bert
        self.dropout = 0.5
        self.lstm = nn.LSTM(768, 100, dropout=0.8, bidirectional=True)
        self.W2 = nn.Linear(2 * 100 + 768, 100)
        self.label = nn.Linear(100, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.word_embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        input = result.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda())
        c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)
        y = y.permute(0, 2, 1)
        y = F.max_pool1d(y, y.size()[2])
        y = y.squeeze(2)
        logits = self.label(y)
        return logits


class Classifier_base_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
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
            sep_outputs = (
                hidden_output[0, sep_ids[0], :]
                .view(1, int(hidden_output.size()[2]))
                .cuda()
            )
            for seq in range(1, int(hidden_output.size()[0])):
                sep_id = sep_ids[seq]
                sep_output = (
                    hidden_output[seq, sep_id, :]
                    .view(1, int(hidden_output.size()[2]))
                    .cuda()
                )
                sep_outputs = torch.cat((sep_outputs, sep_output), dim=0)
            return sep_outputs


class TextCNN(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(TextCNN, self).__init__()
        self.embedding = bert
        # self.embedding = nn.Embedding(vocab_size, 100)
        kernels = [2, 3, 4, 5]
        kernels_number = [768, 768, 768, 768]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, number, (size, 768), padding=(size - 1, 0))
                for (size, number) in zip(kernels, kernels_number)
            ]
        )
        self.line = nn.Linear(768 * 4, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.embedding(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        result = result.unsqueeze(1)
        conv_value = [F.relu(conv(result)).squeeze(3) for conv in self.convs]
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