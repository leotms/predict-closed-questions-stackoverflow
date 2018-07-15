'''
    Author: Leonardo Martinez.
    Last Updated: 13/07/2018.
    Creates folds for training data. Each fold is a combination of 90/10
    train and validation test from data/train.csv
'''

import pandas as pd
import os
from sklearn.model_selection import ShuffleSplit

def check_subdir(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

print('Importing data for folding......')
data = pd.read_csv('data/train.csv')

rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
rs.get_n_splits(data)
i = 1

print('Folding...')
for train_index, test_index in rs.split(data):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]

    path = 'data/folds/%i/'%(i)
    check_subdir(path)
    data_train.to_csv('data/folds/%i/train_%i.csv'%(i,i), index = False)
    data_test.to_csv('data/folds/%i/test_%i.csv'%(i,i), index = False)

    del data_train
    del data_test
    i += 1
