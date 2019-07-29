import time
import torch
import argparse
from test import test
from data import MyPro
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from model import Classifier_base_model, Classifier_pad_model, Classifier_sep_model, Classifier_joint_model

BertModels = {'Classifier_base_model': Classifier_base_model,
              'Classifier_pad_model': Classifier_pad_model,
              'Classifier_sep_model': Classifier_sep_model,
              'Classifier_joint_model': Classifier_joint_model}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
parser.add_argument("--max_seq_length", default=100, type=int, help="字符串最大长度")
parser.add_argument("--eval_batch_size", default=1, type=int, help="验证时batch大小")
parser.add_argument("--no_cuda", default=False, action='store_true', help="用不用CUDA")
parser.add_argument("--train_data_dir", default='/home/user/pycharmprojects/duanxuxiang/BERTProject/enhance_train data/enhance_train4.csv', type=str, help="训练数据读入的路径")
parser.add_argument("--test_data_dir", default='/home/user/pycharmprojects/duanxuxiang/BERTProject/data/cross validation/cross_validation_test2.csv', type=str, help="测试数据读入的路径")
parser.add_argument("--do_lower_case", default=True, action='store_true', help="英文字符的大小写转换")
parser.add_argument("--bert_model", default='bert-base-chinese', type=str, help="选择bert模型的类型")
parser.add_argument("--classifier_model", default='Classifier_sep_model', type=str, help="选择bert的fine tune模型的类型")
args = parser.parse_args()

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
processor = MyPro()
label_list = processor.get_labels(args.train_data_dir)
num_labels = len(label_list)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
bertmodel = BertModel.from_pretrained(args.bert_model)
bert_class_model = BertModels[args.classifier_model](bertmodel, bert_output_size=768, num_labels=num_labels)
bert_class_model.to(device)
bert_class_model.load_state_dict(torch.load('/home/user/pycharmprojects/duanxuxiang/BERTProject/bert_fine_turning/checkpoints/2019-07-27-15-48/bert_classification.pth')['state_dict'])
test(bert_class_model, processor, args, label_list, tokenizer, device)
