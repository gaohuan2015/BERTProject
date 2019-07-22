import torch
import numpy as np
from data import convert_examples_to_features
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def test(bert_model, processor, args, label_list, tokenizer, device):
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    bert_model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output = bert_model(input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
    pr = i / len(test_data)
    print("\n[Test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr


def val(bert_model, processor, args, label_list, tokenizer, device):
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

    bert_model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output = bert_model(input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
    pr = i / len(eval_data)
    print("\n[Valid]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr