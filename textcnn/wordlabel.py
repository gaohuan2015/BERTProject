
path = "D:/VScode/work/textcnn/train"
train = "D:/VScode/work/textcnn/train_pp"
with open(train,'w') as tr:
    with open(path,'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in lines:
            txt = line.split('\t')[0]
            print(txt)
            txt = txt.strip('\n')
            # tr.write(txt + ' ')
            
        