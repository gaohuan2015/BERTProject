# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import codecs
import random
import logging
import argparse
from tqdm import tqdm, trange
from sklearn import metrics
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Linear
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


# 确定example构成
class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# 确定feature构成
class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def read_csv(cls, input_file):
        li_exam = []
        with codecs.open(input_file, 'r', 'utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                line = list(line[0].split('\t'))
                li_exam.append(line)
        return li_exam


class Minemodel(torch.nn.Module):
    def __init__(self, bert_output_size=768, class_size=26):
        super(Minemodel, self).__init__()
        self.lin = torch.nn.Linear(bert_output_size, class_size).cuda()

    def forward(self, bertmodel, input_ids, segment_ids, input_mask):
        _, bert_output = bertmodel(input_ids, segment_ids, input_mask)
        output = self.lin(bert_output)
        return output


class MyPro(DataProcessor):

    def get_train_examples(self, data_dir):
        return self.create_examples(
            self.read_csv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        return self.create_examples(
            self.read_csv(os.path.join(data_dir, "valid.tsv")))

    def get_test_examples(self, data_dir):
        return self.create_examples(
            self.read_csv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self, data_dir):
        label_list = []
        name_id_list = self.read_csv(os.path.join(data_dir, "label2id.tsv"))
        for name_id in name_id_list:
            label_list.append(int(name_id[1]))
        return label_list

    def create_examples(self, li_exam):
        examples = []
        for ele in li_exam:
            guid = ele[0]
            text_a = ele[1]
            label = int(ele[2])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    '''
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b: tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2: tokens_a = tokens_a[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("\n*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length: break
        if len(tokens_a) > len(tokens_b): tokens_a.pop()
        else: tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def val(minemodel, bert_model, processor, args, label_list, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	    processor: 数据读取方法
	    args: 参数表
	    label_list: 所有可能类别
	    tokenizer: 分词方法
	    device:

    Returns: F1值
    '''

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    minemodel.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output = minemodel(bert_model, input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
        # output = output.detach().cpu().numpy()
        # label_ids = label_ids.to('cpu').numpy()
    print("\n[valid] the correct num is:%ld" % i)
    pr = i / len(eval_data)
    print("[valid] the presion is: %lf" % pr)
    return pr


def test(minemodel, bert_model, processor, args, label_list, tokenizer, device):
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    minemodel.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output = minemodel(bert_model, input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
    print("\n[test] the correct num is: %ld" % i)
    pr = i / len(test_data)
    print("[test] the presion is: %lf" % pr)
    return pr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./data/atis/', type=str, help="数据读入的路径")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="选择bert模型的类型")
    parser.add_argument("--task_name", default='MyPro', type=str, help="自己任务的名字")
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
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu", default=False, action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16", default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale", default=128, type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()

    processors = {'mypro': MyPro}
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    task_name = args.task_name.lower()
    if task_name not in processors: raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(args.data_dir)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    bert_model = BertModel.from_pretrained(args.bert_model)
    if args.fp16: bert_model.half()
    bert_model.to(device)
    if args.local_rank != -1:
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1: bert_model = torch.nn.DataParallel(bert_model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) for n, param in bert_model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) for n, param in bert_model.named_parameters()]
    else: param_optimizer = list(bert_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
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

    minemodel = Minemodel(768, 26)
    minemodel.train()
    best_score = 0.7
    flags = 0
    loss_fct = CrossEntropyLoss()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            output = minemodel(bert_model, input_ids, segment_ids, input_mask)
            loss = loss_fct(output, label_ids)
            if n_gpu > 1: loss = loss.mean()
            if args.fp16 and args.loss_scale != 1.0: loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:  loss = loss / args.gradient_accumulation_steps
            if step % 100 == 0:
                print("\n[trainning] the loss in is: %lf" % loss)
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16 or args.optimize_on_cpu:
                    if args.fp16 and args.loss_scale != 1.0:
                        for param in bert_model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data / args.loss_scale
                    is_nan = set_optimizer_params_grad(param_optimizer, bert_model.named_parameters(), test_nan=True)
                    if is_nan:
                        logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                        args.loss_scale = args.loss_scale / 2
                        minemodel.zero_grad()
                        continue
                    optimizer.step()
                    copy_optimizer_params_to_model(bert_model.named_parameters(), param_optimizer)
                else: optimizer.step()
                minemodel.zero_grad()

        pr = val(minemodel, bert_model, processor, args, label_list, tokenizer, device)
        if pr > best_score:
            best_score = pr
            print('precision is {}'.format(pr))
            flags = 0
            checkpoint = {'state_dict': minemodel.state_dict()}
            torch.save(checkpoint, args.model_save_pth)
        else:
            print('precision is {}'.format(pr))
            flags += 1
            if flags >= 6: break

    minemodel.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
    test(minemodel, bert_model, processor, args, label_list, tokenizer, device)


if __name__ == '__main__':
    main()
