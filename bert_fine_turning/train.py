import os
import time
import torch
import random
import argparse
import numpy as np
from test import val, test, ensembletest
from tqdm import tqdm, trange
import torch.nn.functional as F
from ForcalLoss import FocalLoss
from LookheadOptimizer import Lookahead
from torch.nn import CrossEntropyLoss
from LabelSmoothing import LabelSmoothSoftmaxCE
from data import MyPro, convert_examples_to_features
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model import (
    Classifier_base_model,
    Classifier_pad_model,
    Classifier_sep_model,
    Classifier_joint_model,
    RCNN,
    TextCNN,
    TextRNN,
    DPCNN,
)


class EMA:
    def __init__(self, gamma, model):
        super(EMA, self).__init__()
        self.gamma = gamma
        self.shadow = {}
        self.model = model
        self.setup()

    def setup(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = para.clone()

    def cuda(self):
        for k, v in self.shadow.items():
            self.shadow[k] = v.cuda()

    def update(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = (
                    1.0 - self.gamma
                ) * para + self.gamma * self.shadow[name]

    def swap_parameters(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                temp_data = para.data
                para.data = self.shadow[name].data
                self.shadow[name].data = temp_data

    def state_dict(self):
        return self.shadow


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


ti = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
checkpoints_dir = "checkpoints/{}/".format(ti)
if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
    raise ValueError(
        "Output directory ({}) already exists and is not empty.".format(checkpoints_dir)
    )
os.makedirs(checkpoints_dir, exist_ok=True)
model_save_pth = "checkpoints/{}/bert_classification.pth".format(ti)

BertModels = {
    "Classifier_base_model": Classifier_base_model,
    "Classifier_pad_model": Classifier_pad_model,
    "Classifier_sep_model": RCNN,
    "Classifier_joint_model": Classifier_joint_model,
}

LossFunctions = {
    "cross_entropy": CrossEntropyLoss(),
    "focal_loss": FocalLoss(),
    "labelsmoothing": LabelSmoothSoftmaxCE(),
}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
parser.add_argument("--max_seq_length", default=100, type=int, help="字符串最大长度")
parser.add_argument("--eval_batch_size", default=1, type=int, help="验证时batch大小")
parser.add_argument("--train_batch_size", default=8, type=int, help="训练时batch大小")
parser.add_argument("--no_cuda", default=False, action="store_true", help="用不用CUDA")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="Adam初始学习步长")
parser.add_argument(
    "--train_data_dir",
    default="data/cross validation/cross_validation_train4.csv",
    type=str,
    help="训练数据读入的路径",
)
parser.add_argument(
    "--test_data_dir",
    default="data/cross validation/cross_validation_test4.csv",
    type=str,
    help="测试数据读入的路径",
)
parser.add_argument("--num_train_epochs", default=100, type=float, help="训练的epochs次数")
parser.add_argument(
    "--do_lower_case", default=True, action="store_true", help="英文字符的大小写转换"
)
parser.add_argument("--loss_function", default="cross_entropy", type=str, help="损失函数类型")
parser.add_argument(
    "--bert_model", default="bert-base-chinese", type=str, help="选择bert模型的类型"
)
parser.add_argument(
    "--classifier_model",
    default="Classifier_sep_model",
    type=str,
    help="选择bert的fine tune模型的类型",
)
parser.add_argument(
    "--local_rank",
    default=-1,
    type=int,
    help="local_rank for distributed training on gpus.",
)
parser.add_argument(
    "--warmup_proportion",
    default=0.1,
    type=float,
    help="Proportion of train to perf linear lr",
)
args = parser.parse_args()

if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend="nccl")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

processor = MyPro()
label_list = processor.get_labels(args.train_data_dir)
train_examples = processor.get_train_examples(args.train_data_dir)
num_labels = len(label_list)
num_train_steps = int(
    len(train_examples) / args.train_batch_size * args.num_train_epochs
)
tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case
)

bertmodel = BertModel.from_pretrained(args.bert_model)
model1 = RCNN(bertmodel, bert_output_size=768, num_labels=num_labels)
model1.to(device)
model2 = Classifier_base_model(bertmodel, bert_output_size=768, num_labels=num_labels)
model2.to(device)
model3 = Classifier_pad_model(bertmodel, bert_output_size=768, num_labels=num_labels)
model3.to(device)
model4 = TextRNN(bertmodel, bert_output_size=768, num_labels=num_labels)
model4.to(device)
model5 = TextCNN(bertmodel, bert_output_size=768, num_labels=num_labels)
model5.to(device)
param_optimizer3 = list(model3.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters3 = [
    {
        "params": [
            p for n, p in param_optimizer3 if not any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer3 if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0,
    },
]

t_total = num_train_steps
optimizer3 = BertAdam(
    optimizer_grouped_parameters3,
    lr=args.learning_rate,
    warmup=args.warmup_proportion,
    t_total=t_total,
    weight_decay=0.001,
)
lookahead3 = Lookahead(optimizer3, k=5, alpha=0.5)
train_features = convert_examples_to_features(
    train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False
)
print("\n********************** Running training *********************")
print("[Train: %d]" % len(train_examples))
print("[Batch size: %d]" % args.train_batch_size)
print("[Num steps: %d]" % num_train_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in train_features], dtype=torch.long
)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
print("\n********************** Preparing Data ***********************")
train_data = TensorDataset(
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids
)
if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=args.train_batch_size
)
print("\n********************** Start Trainning **********************")
model1.train()
model2.train()
model3.train()
model4.train()
model5.train()
modellist = [model1, model2, model4, model5]
loss_func = LossFunctions[args.loss_function]
trainloss = AverageMeter()
updateEMA = EMA(0.2, model3)
updateEMA.setup()
for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        soft_labels = 0
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        for model_index, model in enumerate(modellist):
            pred = model(input_ids, segment_ids, input_mask)
            soft_labels += F.softmax(pred)
        pred = model3(input_ids, segment_ids, input_mask)
        loss = loss_func(pred.view(-1, num_labels), label_ids.view(-1))
        kd_loss = (
            F.kl_div(
                F.log_softmax(pred.view(-1, num_labels).float(), 1),
                soft_labels / len(modellist),
            )
            * num_labels
        )
        loss = loss + kd_loss
        trainloss.update(loss, pred.size(0))
        # if n_gpu > 1: loss = loss.mean()
        loss.backward()
        # if (step + 1) % 16 == 0:
        torch.nn.utils.clip_grad_norm_(model3.parameters(), 0.01)
        lookahead3.step()
        lookahead3.zero_grad()
        updateEMA.update()
        total_loss += loss.data
    print(
        "\n[Trainning]\t[Epoch; %d]\t[Iteration: %d]\t[Loss: %f]"
        % (_, step, total_loss)
    )
    total_loss = 0
    test(model1, processor, args, label_list, tokenizer, device)
    test(model2, processor, args, label_list, tokenizer, device)
    test(model3, processor, args, label_list, tokenizer, device)
    test(model4, processor, args, label_list, tokenizer, device)
    test(model5, processor, args, label_list, tokenizer, device)
    ensembletest(
        model1,
        model2,
        model3,
        model4,
        model5,
        processor,
        args,
        label_list,
        tokenizer,
        device,
    )
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    model5.train()
