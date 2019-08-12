import os
import torch
import numpy as np
import torch.nn.functional as F
from data import convert_examples_to_features
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def test(model, processor, args, label_list, tokenizer, device):
    test_examples = processor.get_test_examples(args.test_data_dir)
    with open(args.test_data_dir, "r", encoding='utf-8') as f:
        test_list = []
        for line in f:
            _, text_a, label = line.strip("\n").split("\t")
            test_list.append((text_a, label))
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in test_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in test_features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size
    )
    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    f = open("Error data.txt", "w")
    for text_id, (input_ids, input_mask, segment_ids, label_ids) in enumerate(
        test_dataloader
    ):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            if args.classifier_model == "Classifier_joint_model":
                output_base, output_pad, output_textcnn = model(
                    input_ids, segment_ids, input_mask
                )
                output = output_base
            else:
                output = model(input_ids, segment_ids, input_mask)
            labpre = output
            # labpre, ind = torch.max(output_base,dim=-1)
            # labpre1, ind1 = torch.max(output_pad,dim=-1)
            # if labpre >= labpre1:
            #     labpre = ind
            # else:
            #     labpre = ind1
            labpre = labpre.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
            else:
                f.write(
                    test_list[text_id][0]
                    + "\t"
                    + str(predict[-1])
                    + "\t"
                    + test_list[text_id][1]
                    + "\n"
                )
    pr = i / len(test_data)
    print("\n[Test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr


def ensembletest(
    model1, model2, model3, processor, args, label_list, tokenizer, device
):
    test_examples = processor.get_test_examples(args.test_data_dir)
    with open(args.test_data_dir, "r", 'utf-8') as f:
        test_list = []
        for line in f:
            _, text_a, label = line.strip("\n").split("\t")
            test_list.append((text_a, label))
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in test_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in test_features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size
    )
    model1.eval()
    model2.eval()
    model3.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    f = open("Error data.txt", "w")
    for text_id, (input_ids, input_mask, segment_ids, label_ids) in enumerate(
        test_dataloader
    ):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output1 = model1(input_ids, segment_ids, input_mask)
            output2 = model2(input_ids, segment_ids, input_mask)
            output3 = model3(input_ids, segment_ids, input_mask)
            labpre = F.softmax(output1, dim=-1) + F.softmax(output2,
                                                            dim=-1) + F.softmax(output3, dim=-1)
            # labpre, ind = torch.max(output_base,dim=-1)
            # labpre1, ind1 = torch.max(output_pad,dim=-1)
            # if labpre >= labpre1:
            #     labpre = ind
            # else:
            #     labpre = ind1
            labpre = labpre.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
            else:
                f.write(
                    test_list[text_id][0]
                    + "\t"
                    + str(predict[-1])
                    + "\t"
                    + test_list[text_id][1]
                    + "\n"
                )
    pr = i / len(test_data)
    print("\n[Test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr


def val(model, processor, args, label_list, tokenizer, device):

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    i = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1)
            predict = np.hstack((predict, labpre.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if predict[-1] == gt[-1]:
                i += 1
    pr = i / len(eval_data)
    print("\n[Valid]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr
