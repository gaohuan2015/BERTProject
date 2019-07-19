import os
import torch
import random
import logging
import argparse
import numpy as np
from test import val, test
from model import Classifiermodel
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from pretrained_bert.optimization import BertAdam
from pretrained_bert.tokenization import BertTokenizer
from data_input import MyPro, convert_examples_to_features
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

logger = logging.getLogger(__name__)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default='../data/atis/', type=str, help="数据读入的路径")
parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="选择bert模型的类型")
parser.add_argument("--output_dir", default='checkpoints/', type=str, help="checkpoints的路径")
parser.add_argument("--model_save_pth", default='checkpoints/bert_classification.pth', type=str, help="模型保存的路径")
parser.add_argument("--max_seq_length", default=100, type=int, help="字符串最大长度")
parser.add_argument("--do_lower_case", default=True, action='store_true', help="英文字符的大小写转换")
parser.add_argument("--train_batch_size", default=32, type=int, help="训练时batch大小")
parser.add_argument("--eval_batch_size", default=1, type=int, help="验证时batch大小")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Adam初始学习步长")
parser.add_argument("--num_train_epochs", default=10.0, type=float, help="训练的epochs次数")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda", default=False, action='store_true', help="用不用CUDA")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus.")
parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
parser.add_argument("--loss_scale", default=128, type=float,
                    help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

args = parser.parse_args()

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')
logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0: torch.cuda.manual_seed_all(args.seed)

if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
os.makedirs(args.output_dir, exist_ok=True)
processor = MyPro()
label_list = processor.get_labels(args.data_dir)
train_examples = processor.get_train_examples(args.data_dir)
num_labels = len(label_list)
num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
bert_class_model = Classifiermodel.from_pretrained(args.bert_model, bert_output_size=768, num_labels=num_labels)
bert_class_model.to(device)
if args.local_rank != -1:
    bert_class_model = torch.nn.parallel.DistributedDataParallel(bert_class_model, device_ids=[args.local_rank], output_device=args.local_rank)
elif n_gpu > 1: bert_class_model = torch.nn.DataParallel(bert_class_model)

# Prepare optimizer
param_optimizer = list(bert_class_model.named_parameters())
no_decay = ['bias','LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]
t_total = num_train_steps
if args.local_rank != -1: t_total = t_total // torch.distributed.get_world_size()
optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=t_total)

global_step = 0
train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
logger.info("\n***** Running training *****")
logger.info("Num examples = %d", len(train_examples))
logger.info("Batch size = %d", args.train_batch_size)
logger.info("Num steps = %d", num_train_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args.local_rank == -1: train_sampler = RandomSampler(train_data)
else: train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

bert_class_model.train()
best_score = 0.7
flags = 0
loss_func = CrossEntropyLoss()
for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        pred = bert_class_model(input_ids, segment_ids, input_mask)
        loss = loss_func(pred.view(-1, num_labels), label_ids.view(-1))
        if n_gpu > 1: loss = loss.mean()
        if args.loss_scale != 1.0: loss = loss * args.loss_scale
        if step % 100 == 0:
            print("\n[trainning] the loss in is: %lf" % loss)
        loss.backward()
        if args.loss_scale != 1.0:
            for param in bert_class_model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data / args.loss_scale
        optimizer.step()
        copy_optimizer_params_to_model(bert_class_model.named_parameters(), param_optimizer)
        optimizer.step()
        bert_class_model.zero_grad()

    pr = val(bert_class_model, processor, args, label_list, tokenizer, device)
    if pr > best_score:
        best_score = pr
        print('precision is {}'.format(pr))
        flags = 0
        checkpoint = {'state_dict': bert_class_model.state_dict()}
        torch.save(checkpoint, args.model_save_pth)
    else:
        print('precision is {}'.format(pr))
        flags += 1
        if flags >= 6: break

bert_class_model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
test(bert_class_model, processor, args, label_list, tokenizer, device)
