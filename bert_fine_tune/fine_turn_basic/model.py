import torch
from torch.nn import Linear
from pretrained_bert.modeling import BertPreTrainedModel, BertModel
import numpy as np

class Classifiermodel(BertPreTrainedModel):
    def __init__(self, config, bert_output_size=768, num_labels=26):
        super(Classifiermodel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        pred = self.lin(cls_outputs)
        return pred


class Classifier_pad_model(BertPreTrainedModel):
    def __init__(self, config, bert_output_size=768, num_labels=26):
        super(Classifier_pad_model, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        pad_outputs = hiddens_outputs[-1][:, -1, :]
        outs = torch.cat((pad_outputs, cls_outputs), dim=1)
        pred = self.lin(outs)
        return pred


class Classifier_sep_model(BertPreTrainedModel):
    def __init__(self, config, bert_output_size=768, num_labels=26):
        super(Classifier_sep_model, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        hiddens_outputs, cls_outputs = self.bert(input_ids, segment_ids, input_mask)
        sep_outputs = self.sep_search(hiddens_outputs[-1], input_mask)
        outs = torch.cat((sep_outputs, cls_outputs), dim=1)
        out=np.sum(input_mask)
        pred = self.lin(outs)
        return pred

    def sep_search(self, hidden_output, input_mask):
        if len(hidden_output[0]) == len(input_mask[0]):
            sep_id_list = []
            for row in range(len(input_mask)):
                sep_id = input_mask[row, :].tolist().index(0).cuda()-1
                sep_id_list.append([sep_id])
            sep_id_zl = torch.LongTensor(sep_id_list)
            sep_out_put = torch.gather(hidden_output, dim=1, index=sep_id_zl)
            return sep_out_put

