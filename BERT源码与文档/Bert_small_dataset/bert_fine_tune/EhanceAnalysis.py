# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is used to data enhancement analysis
# --------------------------------****************************-----------------------------------------------------
import csv
import jieba


# get the sentence in the file
def get_sentences(filepath):
    sentences = []
    parses = []
    f = csv.reader(open(filepath, 'r', encoding='utf-8'), delimiter='\t')
    for line in f:
        sentences.append(line[1].strip())
        parse = ','.join(jieba.cut(line[1].strip(), cut_all=False)).split(',')
        parses.append(parse)
    return sentences, parses


originpath = '\BERTProject\data\cross validation\cross_validation_train3.csv'
enhancepath = '\BERTProject\data\enhance_train data\enhance_train3.csv'
originsentences, originparses = get_sentences(originpath)
enhancesentences, enhanceparses = get_sentences(enhancepath)

f1 = open('1.txt', 'w', encoding='utf-8')
i = 1
if len(originparses) == len(enhanceparses):
    for ind1 in range(len(originparses)):
        changed_parses = ''
        if len(originparses[ind1]) == len(enhanceparses[ind1]):
            for ind2 in range(len(originparses[ind1])):
                if originparses[ind1][ind2] != enhanceparses[ind1][ind2]:
                    changed_parses += originparses[ind1][ind2] + ',' + enhanceparses[ind1][ind2] + '\t'
            f1.write(originsentences[ind1] + ',' + enhancesentences[ind1] + '\t\t' + changed_parses + '\n')
        else:
            print(originparses[ind1], '\t\t', enhanceparses[ind1])
            i += 1
