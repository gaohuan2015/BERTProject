# -*- coding: utf-8 -*-
import csv
from tqdm import tqdm

trainFile = 'D:/VScode/work/textcnn/train'
wordLabelFile = 'D:/VScode/work/textcnn/wordLabel'

def main():
    worddict = {}
    len_dic = {}
    with open (trainFile,'r') as f:
        lines = f.readlines()
        for line in lines:
            txt = line.split('\t')[0]
            txt = txt.split()
            length = 0
            for w in txt:
                if w in worddict:
                    worddict[w] += 1
                else:
                    worddict[w] = 1
                length += 1
            if length in len_dic:
                len_dic[length] += 1
            else:
                len_dic[length] = 1

    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='utf_8')
    ind = 0
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

if __name__ == "__main__":
    main()