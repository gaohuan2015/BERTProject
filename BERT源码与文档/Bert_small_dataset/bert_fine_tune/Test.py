# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file contains the model evaluation and testing function
# --------------------------------****************************-----------------------------------------------------

import os
import torch
import random
import collections
import numpy as np
import torch.nn.functional as F
from Data import convert_examples_to_features
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def test(model, processor, args, label_list, device, tokenizer, modelname=''):
    '''
    :param model: the trained model
    :param processor: the data processor
    :param args: the command line parameters
    :param label_list: the list of all the labels
    :param device: device is "GPU" or "CPU"
    :param tokenizer: bert tokenizer
    :param modelname: modelname default is ""
    :return: the accuracy of the trained model
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    test_examples = processor.get_test_examples(args.test_data_dir)
    with open(args.test_data_dir, "r", encoding='utf-8') as f:
        test_list = []
        for line in f:
            _, text_a, label = line.strip("\n").split("\t")
            test_list.append((text_a, label))
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()
    i = 0
    f = open("../error_data/error_data.txt", "w")
    for text_id, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            if args.classifier_model == "Classifier_joint_model":
                output_base, output_pad, output_textcnn = model(input_ids, segment_ids, input_mask)
                output = output_base
            else:
                output = model(input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1).cpu().numpy().tolist()
            label_ids.cpu().numpy().tolist()
            if len(labpre) == len(label_ids):
                for ind in range(len(labpre)):
                    if labpre[ind] == label_ids[ind]:
                        i += 1
                    elif modelname == 'bert_pad':
                        f.write(test_list[text_id * args.eval_batch_size + ind][0] + "\t" + str(labpre[ind])
                            + "\t" + test_list[text_id * args.eval_batch_size + ind][1] + "\n")
    f.close()
    pr = i / len(test_data)
    print("\n[test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr


def ensembletest(model1, model2, model3, model4, model5,
                 processor, args, label_list, device, tokenizer):
    test_examples = processor.get_test_examples(args.test_data_dir)
    with open(args.test_data_dir, "r", encoding='utf-8') as f:
        test_list = []
        for line in f:
            _, text_a, label = line.strip("\n").split("\t")
            test_list.append((text_a, label))
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    i = 0
    f = open("Error data0.txt", "w")
    for text_id, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            output1 = model1(input_ids, segment_ids, input_mask)
            output2 = model2(input_ids, segment_ids, input_mask)
            output3 = model3(input_ids, segment_ids, input_mask)
            output4 = model4(input_ids, segment_ids, input_mask)
            output5 = model5(input_ids, segment_ids, input_mask)
            labpre = (
                F.softmax(output1, dim=-1)
                + F.softmax(output2, dim=-1)
                + F.softmax(output3, dim=-1)
                + F.softmax(output4, dim=-1)
                + F.softmax(output5, dim=-1)
            )
            labpre = labpre.argmax(dim=1).cpu().numpy().tolist()
            label_ids.cpu().numpy().tolist()
            gt = np.hstack((gt, label_ids.cpu().numpy()))
            if len(labpre) == len(label_ids):
                for ind in range(len(labpre)):
                    if labpre[ind] == label_ids[ind]:
                        i += 1
                else:
                    f.write(test_list[text_id * args.eval_batch_size + ind][0] + "\t" + str(labpre[ind])
                            + "\t" + test_list[text_id * args.eval_batch_size + ind][1] + "\n")
    pr = i / len(test_data)
    print("\n[test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr


def val(model, processor, args, label_list, device, tokenizer):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    test_examples = processor.get_test_examples(args.test_data_dir)
    with open(args.test_data_dir, "r", encoding='utf-8') as f:
        test_list = []
        for line in f:
            _, text_a, label = line.strip("\n").split("\t")
            test_list.append((text_a, label))
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, show_exp=False
    )
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()
    i = 0
    for text_id, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            if args.classifier_model == "Classifier_joint_model":
                output_base, output_pad, output_textcnn = model(input_ids, segment_ids, input_mask)
                output = output_base
            else:
                output = model(input_ids, segment_ids, input_mask)
            labpre = output.argmax(dim=1).cpu().numpy().tolist()
            label_ids.cpu().numpy().tolist()
            if len(labpre) == len(label_ids):
                for ind in range(len(labpre)):
                    if labpre[ind] == label_ids[ind]:
                        i += 1
    pr = i / len(test_data)
    print("\n[test]\t[Correct Num: %d]\t[Presion: %f]" % (i, pr))
    return pr