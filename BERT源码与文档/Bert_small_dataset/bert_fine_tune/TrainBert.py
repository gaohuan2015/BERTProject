# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file can be used to train the single bert series model
# --------------------------------****************************-----------------------------------------------------
import os
import time
import torch
import random
import argparse
import numpy as np
from Test import test
from tqdm import tqdm, trange
from ForcalLoss import FocalLoss
from torch.nn import CrossEntropyLoss
from LabelSmoothing import LabelSmoothSoftmaxCE
from Data import MyPro, convert_examples_to_features
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from Model import Classifier_base_model

LossFunctions = {"cross_entropy": CrossEntropyLoss(),
                 "focal_loss": FocalLoss(),
                 "labelsmoothing": LabelSmoothSoftmaxCE()}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
parser.add_argument("--no_cuda", default=False, action="store_true", help="用不用CUDA")
parser.add_argument("--loss_function", default="labelsmoothing", type=str, help="损失函数类型")
parser.add_argument("--device_id", default=0, type=int, help="gpu id")
parser.add_argument("--do_lower_case", default=True, action="store_true", help="英文字符的大小写转换")
parser.add_argument("--bert_model", default="../PretrainedModels/bert_base_chinese", type=str, help="选择bert模型的类型")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of train to perf linear lr")
parser.add_argument("--classifier_model", default="Classifier_sep_model", type=str, help="选择bert的fine tune模型的类型")

parser.add_argument("--max_seq_length", default=70, type=int, help="字符串最大长度")
parser.add_argument("--eval_batch_size", default=184, type=int, help="验证时batch大小")

parser.add_argument("--train_batch_size", default=64, type=int, help="训练时batch大小")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Adam初始学习步长")
parser.add_argument("--num_train_epochs", default=50, type=float, help="训练的epochs次数")

parser.add_argument("--train_data_dir", type=str, help="训练数据读入的路径",
                    default="../data/Small/train588.csv")
parser.add_argument("--test_data_dir", type=str, help="测试数据读入的路径",
                    default="../data/Small/test184.csv")
args = parser.parse_args()

ti = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
checkpoints_dir = "../checkpoints/{}/".format(ti)
if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(checkpoints_dir))
os.makedirs(checkpoints_dir, exist_ok=True)
model_save_pth = checkpoints_dir

if args.local_rank == -1 or args.no_cuda:
    n_gpu = torch.cuda.device_count()
else:
    n_gpu = 1
    torch.distributed.init_process_group(backend="nccl")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:' + str(args.device_id))
processor = MyPro()
label_list = processor.get_labels(args.train_data_dir)
train_examples = processor.get_train_examples(args.train_data_dir)
num_labels = len(label_list)
num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
bertmodel = BertModel.from_pretrained(args.bert_model)
model = Classifier_base_model(bertmodel, bert_output_size=768, num_labels=num_labels)
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay_rate": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay_rate": 0.0}]

t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      warmup=args.warmup_proportion,
                      t_total=t_total,
                      weight_decay=0.001,
                     )
train_features = convert_examples_to_features(
    train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
print("\n********************** Running training *********************")
print("[Train: %d]" % len(train_examples))
print("[Batch size: %d]" % args.train_batch_size)
print("[Num steps: %d]" % num_train_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

print("\n********************** Preparing Data ***********************")
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
print("\n********************** Start Trainning **********************")
model.train()
loss_func = LossFunctions[args.loss_function]
for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        soft_labels = 0
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        pred = model(input_ids, segment_ids, input_mask)
        loss = loss_func(pred.view(-1, num_labels), label_ids.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.data
    print("\n[Trainning]\t[Epoch; %d]\t[Iteration: %d]\t[Loss: %f]" % (_, step, total_loss))
    pr = test(model, processor, args, label_list, device, tokenizer)
    model.train()
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, model_save_pth + 'Bert_base{}.bin'.format(_))
    total_loss = 0
checkpoint = {'state_dict': model.state_dict()}
torch.save(checkpoint, model_save_pth + 'Bert_base.bin')
test(model, processor, args, label_list, device, tokenizer, modelname='bert')
torch.cuda.empty_cache()
