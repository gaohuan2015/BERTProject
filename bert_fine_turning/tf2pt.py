# coding=utf-8

"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)
    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, tf_checkpoint_path)
    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    tf_checkpoint_path = '/home/user/pycharmprojects/duanxuxiang/bert_fine_turning/publish/bert_model.ckpt.index'
    bert_config_file = '/home/user/pycharmprojects/duanxuxiang/bert_fine_turning/publish/bert_config.json'
    pytorch_dump_path = '/home/user/pycharmprojects/duanxuxiang/bert_fine_turning/publish/bert_model.bin'
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)


