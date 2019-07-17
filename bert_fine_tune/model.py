import torch
from torch.nn import Linear
from pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Classifiermodel(BertPreTrainedModel):
    def __init__(self, config, bert_output_size=768, num_labels=26):
        super(Classifiermodel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config=config)
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