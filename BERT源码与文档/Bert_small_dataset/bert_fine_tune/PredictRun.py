# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is the code of model testing
# --------------------------------****************************-----------------------------------------------------

import torch
import json
import argparse
import random
import numpy as np
from Test import val, test
from Data import MyPro
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from Model import (Classifier_base_model, Classifier_pad_model, RCNN, TextCNN, TextRNN)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
parser.add_argument("--do_lower_case", default=True, action="store_true", help="英文字符的大小写转换")
parser.add_argument("--no_cuda", default=False, action='store_true', help="用不用CUDA")
parser.add_argument("--device_id", default=0, type=int, help="gpu id")
parser.add_argument("--max_seq_length", default=70, type=int, help="字符串最大长度")
parser.add_argument("--eval_batch_size", default=184, type=int, help="验证时batch大小")
parser.add_argument("--classifier_model", default="Classifier_sep_model", type=str, help="选择bert的fine tune模型的类型")
parser.add_argument("--test_type", default='small', type=str, help="训练数据读入的路径", required=False)
parser.add_argument("--pretrained_model", default= '../TrainedModels/Small/besides/mtdnn/Rcnnbert.bin', type=str, help="选择待模型的类型", required=False)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:' + str(args.device_id))
bert_model = '../PretrainedModels/bert_base_chinese'

if args.test_type == 'small':
    args.train_data_dir = '../data/Small/train660.csv'
    args.test_data_dir = '../data/Small/test184.csv'
    bert_model = '../PretrainedModels/bert_base_chinese'
elif args.test_type == 'open':
    args.train_data_dir = '../data/Atis/train.txt'
    args.test_data_dir = '../data/Atis/test.txt'
    bert_model = 'bert-base-uncased'
else:
    print("you have not give the paramenter test_type")

processor = MyPro()
label_list = processor.get_labels(args.train_data_dir)
num_labels = len(label_list)
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=args.do_lower_case)
bertmodel = BertModel.from_pretrained(bert_model)
if 'Rcnnbert' in args.pretrained_model:
    model = RCNN(bertmodel, bert_output_size=768, num_labels=num_labels, device=device)
elif 'Bert_base' in args.pretrained_model:
    model = Classifier_base_model(bertmodel, bert_output_size=768, num_labels=num_labels)
elif 'Bert_pad' in args.pretrained_model:
    model = Classifier_pad_model(bertmodel, bert_output_size=768, num_labels=num_labels)
elif 'Textrnnbert' in args.pretrained_model:
    model = TextRNN(bertmodel, bert_output_size=768, num_labels=num_labels, device=device)
elif 'Textcnnbert' in args.pretrained_model:
    model = TextCNN(bertmodel, bert_output_size=768, num_labels=num_labels)
else:
    print('The pretrained model path you inputed could be false')
model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
model.to(device)
print("\n********************** Start Testing **********************")
test(model, processor, args, label_list, device, tokenizer, modelname='bert')
torch.cuda.empty_cache()
