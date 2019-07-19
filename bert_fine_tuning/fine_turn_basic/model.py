from torch.nn import Linear
from pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Classifiermodel(BertPreTrainedModel):
    def __init__(self, config, bert_output_size=768, num_labels=26):
        super(Classifiermodel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.lin = Linear(bert_output_size, num_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        _, bert_outputs = self.bert(input_ids, segment_ids, input_mask)
        pred = self.lin(bert_outputs)
        return pred