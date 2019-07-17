import pandas as pd
import numpy as np

file = r'chin_ dataset.csv'
new_file = r'segment_chin_dataset_{}.csv'
data = pd.read_csv(file, header=None, sep='\t')
indexs = np.arange(len(data))
np.random.shuffle(indexs)
indexs.resize((5, 132))  # 660 = 5 *132
for i, index in enumerate(indexs):
    with open(new_file.format(i), 'w', encoding='utf-8'):
        sub_data = data.iloc()[index]
        sub_data.to_csv(new_file.format(i), index=False, sep='\t', header=None)
    print('subdata %d segment complete.' % i)