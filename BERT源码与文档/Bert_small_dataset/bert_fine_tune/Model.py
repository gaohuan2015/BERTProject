# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file contains some bert fine tune models
# --------------------------------****************************-----------------------------------------------------

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear
import torch.nn as nn
import copy


class Seq2SeqAttention(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(Seq2SeqAttention, self).__init__()
        self.embeddings = bert
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=100, num_layers=2, bidirectional=True
        )
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * (1 + True) * 2, num_labels)
        self.softmax = nn.Softmax()

    def apply_attention(self, rnn_output, final_hidden_state):
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(
            2
        )  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(
            rnn_output.permute(0, 2, 1), soft_attention_weights
        ).squeeze(2)
        return attention_output

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        result = result[:, 1:-2, :].permute(1, 0, 2)
        lstm_output, (h_n, c_n) = self.lstm(result)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(2, True + 1, batch_size, 100)[-1, :, :, :]
        final_hidden_state = torch.cat(
            [h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1
        )

        attention_out = self.apply_attention(
            lstm_output.permute(1, 0, 2), final_hidden_state
        )
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = concatenated_vector
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)


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
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(inplace=True),
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
            # nn.Dropout(0.3),
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
        x = result[:, 1:-2, :].permute(0, 2, 1)
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
    def __init__(self, bert, bert_output_size=768, num_labels=26, device='cuda:0'):
        super(TextRNN, self).__init__()
        self.word_embeddings = bert
        self.device = device
        self.lstm = nn.LSTM(768, 100, bidirectional=True, dropout=0.5)
        self.label = nn.Linear(200, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.word_embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        input = result[:, 1:-2, :].permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
        c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        return self.label(output[:, -1])


class RCNN(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26, device='cuda:0'):
        super(RCNN, self).__init__()
        self.word_embeddings = bert
        self.device = device
        # self.dropout = 0.5
        self.lstm = nn.LSTM(768, 100, bidirectional=True, dropout=0.8)
        self.W2 = nn.Linear(2 * 100 + 768, 100)
        self.label = nn.Linear(100, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        result, cls_outputs = self.word_embeddings(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        input = result[:, 1:-2, :].permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
        c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))

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
        """
        :param bert: the pretrained bert model
        :param bert_output_size: the output size of pretrained bert model
        :param num_labels: the total number of labels
        """
        super(Classifier_base_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        """
        :param input_ids: the token id of input sentence
        :param segment_ids: the segment id of input sentence
        :param input_mask: the input mask of input sentence
        :return: the predict labels probability matix
        """
        hiddens_outputs, cls_outputs = self.bert(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False
        )
        pred = self.lin(cls_outputs)
        return pred


class Classifier_pad_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(Classifier_pad_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size * 2, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=True
        )
        pad_outputs = hiddens_outputs[-1][:, -1, :]
        outs = torch.cat((pad_outputs, cls_outputs), dim=1)
        pred = self.lin(outs)
        return pred


class Classifier_sep_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26, device='cuda:0'):
        super(Classifier_sep_model, self).__init__()
        self.num_labels = num_labels
        self.device = device
        self.bert = bert
        self.lin = Linear(bert_output_size * 2, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=True
        )
        sep_outputs = self.sep_search(hiddens_outputs[-1], input_mask)
        outs = torch.cat((sep_outputs, cls_outputs), dim=1)
        pred = self.lin(outs)
        return pred

    def sep_search(self, hidden_output, input_mask):
        if int(hidden_output.size()[0]) == int(input_mask.size()[0]):
            sep_ids = input_mask.sum(dim=1).tolist()
            sep_outputs = (
                hidden_output[0, sep_ids[0]-1, :]
                .view(1, int(hidden_output.size()[2]))
                .cuda(self.device)
            )
            for seq in range(1, int(hidden_output.size()[0])):
                sep_id = sep_ids[seq]-1
                sep_output = (
                    hidden_output[seq, sep_id, :]
                    .view(1, int(hidden_output.size()[2]))
                    .cuda(self.device)
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
        result = result[:, 1:-1, :].unsqueeze(1)
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
        # self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, segment_ids, input_mask):
        pred_base = self.bert_base(input_ids, segment_ids, input_mask)
        pred_pad = self.bert_pad(input_ids, segment_ids, input_mask)
        pre_textcnn = self.textCNN(input_ids, segment_ids, input_mask)
        return self.dropout(pred_base), pred_pad, pre_textcnn


class Classifier_last_four_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26):
        super(Classifier_last_four_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.lin = Linear(bert_output_size * 4, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=True)
        last_output = hiddens_outputs[-1][:, 0, :].view(-1, int(hiddens_outputs[-1].size()[2]))
        lastone_output = hiddens_outputs[-2][:, 0, :].view(-1, int(hiddens_outputs[-2].size()[2]))
        lasttwo_output = hiddens_outputs[-3][:, 0, :].view(-1, int(hiddens_outputs[-3].size()[2]))
        lastthree_output = hiddens_outputs[-4][:, 0, :].view(-1, int(hiddens_outputs[-4].size()[2]))
        outs = torch.cat((last_output, lastone_output, lasttwo_output, lastthree_output), dim=1)
        pred = self.lin(outs)
        return pred


class Classifier_base_rnncnn_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26, device='cuda:0'):
        super(Classifier_base_rnncnn_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.device = device
        self.linernn = nn.Linear(200, num_labels)
        self.linecnn = nn.Linear(768 * 4, num_labels)
        self.lin = Linear(bert_output_size + 325*2, num_labels)

        kernels = [2, 3, 4, 5]
        kernels_number = [768, 768, 768, 768]
        self.lstm = nn.LSTM(768, 100, bidirectional=True, dropout=0.5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, number, (size, 768), padding=(size - 1, 0)) for (size, number) in zip(kernels, kernels_number)]
        )

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        hiddens_outputscnn = hiddens_outputs[:, 1:-1, :]
        result_cnn = hiddens_outputscnn.unsqueeze(1)
        conv_value = [F.relu(conv(result_cnn)).squeeze(3) for conv in self.convs]
        result_cnn = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_value]
        result_cnn = torch.cat(result_cnn, 1)
        result_cnn = self.linecnn(result_cnn)
        # print(result_cnn.size())
        result_rnn = hiddens_outputs[:, 1:-2, :].permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
        c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
        result_rnn, _ = self.lstm(result_rnn, (h_0, c_0))
        result_rnn = result_rnn.permute(1, 0, 2)
        result_rnn = self.linernn(result_rnn[:, -1])
        # print(result_rnn.size())
        outs = torch.cat((cls_outputs, result_cnn, result_rnn), dim=1)

        pred = self.lin(outs)

        return pred

class Classifier_base_rnncnn_last_four_model(nn.Module):
    def __init__(self, bert, bert_output_size=768, num_labels=26, device='cuda:0'):
        super(Classifier_base_rnncnn_last_four_model, self).__init__()
        self.num_labels = num_labels
        self.bert = bert
        self.device = device
        self.linernn = nn.Linear(200, num_labels)
        self.linecnn = nn.Linear(768 * 4, num_labels)
        self.lin = Linear(bert_output_size + 294*8, num_labels)

        kernels = [2, 3, 4, 5]
        kernels_number = [768, 768, 768, 768]
        self.lstm = nn.LSTM(768, 100, bidirectional=True, dropout=0.5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, number, (size, 768), padding=(size - 1, 0)) for (size, number) in zip(kernels, kernels_number)]
        )

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs_all, cls_outputs = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=True)
        for i, hiddens_outputs in enumerate(hiddens_outputs_all[-4:]):
            hiddens_outputscnn = hiddens_outputs[:, 1:-1, :]
            result_cnn = hiddens_outputscnn.unsqueeze(1)
            conv_value = [F.relu(conv(result_cnn)).squeeze(3) for conv in self.convs]
            result_cnn = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_value]
            result_cnn = torch.cat(result_cnn, 1)
            result_cnn = self.linecnn(result_cnn)

            result_rnn = hiddens_outputs[:, 1:-2, :].permute(1, 0, 2)
            h_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
            c_0 = Variable(torch.zeros(2, input_ids.size(0), 100).cuda(self.device))
            result_rnn, _ = self.lstm(result_rnn, (h_0, c_0))
            result_rnn = result_rnn.permute(1, 0, 2)
            result_rnn = self.linernn(result_rnn[:, -1])
            if i == 0:
                result_cnn_all = result_cnn
                result_rnn_all = result_cnn
            else:
                result_cnn_all = torch.cat((result_cnn_all, result_cnn), 1)
                result_rnn_all = torch.cat((result_rnn_all, result_rnn), 1)
        outs = torch.cat((cls_outputs, result_cnn_all, result_rnn_all), dim=1)
        pred = self.lin(outs)

        return pred