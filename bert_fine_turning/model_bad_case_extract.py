import time
import torch
import argparse
from test import val, test
from data import MyPro, convert_examples_to_features
from pretrained_bert.tokenization import BertTokenizer
from model import Classifiermodel, Classifier_pad_model, Classifier_sep_model


ti = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default='./chinese_data/', type=str, help="数据读入的路径")
parser.add_argument("--bert_model", default='bert-chinese-wwm', type=str, help="选择bert模型的类型")
parser.add_argument("--output_dir", default='checkpoints/{}/'.format(ti), type=str, help="checkpoints的路径")
parser.add_argument("--model_save_pth", default='checkpoints/{}/bert_classification.pth'.format(ti), type=str, help="模型保存的路径")
parser.add_argument("--max_seq_length", default=100, type=int, help="字符串最大长度")
parser.add_argument("--do_lower_case", default=True, action='store_true', help="英文字符的大小写转换")
parser.add_argument("--eval_batch_size", default=1, type=int, help="验证时batch大小")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Adam初始学习步长")
parser.add_argument("--num_train_epochs", default=20.0, type=float, help="训练的epochs次数")
parser.add_argument("--no_cuda", default=False, action='store_true', help="用不用CUDA")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
processor = MyPro()
label_list = processor.get_labels(args.data_dir)
num_labels = len(label_list)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
# bert_class_model = Classifiermodel.from_pretrained(args.bert_model, bert_output_size=768, num_labels=num_labels)
# bert_class_model = Classifier_pad_model.from_pretrained(args.bert_model, bert_output_size=768*2, num_labels=num_labels)
bert_class_model = Classifier_sep_model.from_pretrained(args.bert_model, bert_output_size=768*2, num_labels=num_labels)
bert_class_model.to(device)
bert_class_model.load_state_dict(torch.load('/home/user/pycharmprojects/duanxuxiang/bert_fine_turning/checkpoints/2019-07-26-10-10/bert_classification.pth')['state_dict'])
test(bert_class_model, processor, args, label_list, tokenizer, device)
