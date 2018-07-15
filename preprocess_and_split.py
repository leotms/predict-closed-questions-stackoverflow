'''
    Author: Leonardo Martinez.
    Last Updated: 13/07/2018.
    Preprocess data:
        Removes outliers
        Normalizes
    And splits it into train and test sets (90/10).
'''
from   sklearn.model_selection import ShuffleSplit
import pandas   as pd
import numpy    as np
import os

def check_subdir(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

print('Importing data...')
data = pd.read_csv('train_complete_preprocessed.csv')

print('Removing outliers from class.')
print('Total instances before removing: ', data.shape[0])

# Calculates outliers for each class separately. And removes instances from dataset.
aux_data = pd.DataFrame(columns=data.columns)

for i in range(5):
    subdata = data[data['OpenStatus'] == i]
    bounds  = {}
    attributes = data.columns.tolist()
    attributes.remove('OpenStatus')
    for attribute in attributes:
        percentiles = np.percentile(subdata[attribute], [25, 50, 75])
        bounds[attribute] = {'min' : percentiles[1] - percentiles[0]*2, 'max' : percentiles[1] + percentiles[2]*2 }
    for attribute in bounds:
        # avoiding empty ranges areas
        if bounds[attribute]['min'] < bounds[attribute]['max']:
            subdata = subdata.loc[(subdata[attribute] >= bounds[attribute]['min']) & (subdata[attribute] <= bounds[attribute]['max'])]
    aux_data = aux_data.append(subdata)
    del subdata

data = aux_data
del aux_data
print('Total instances after removing: ', data.shape[0])

rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

print('Spliting into Train and Test sets...')
for train_index, test_index in rs.split(data):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]

check_subdir('data/train.csv')
check_subdir('data/test.csv')
data_train.to_csv('data/train.csv', index = False)
data_test.to_csv('data/test.csv', index = False)

print('Total instances in training set by class:')
grouped = data_train.groupby('OpenStatus')
for group in grouped:
    print(group[0], group[1].shape)

print('Total instances in test set by class:')
grouped = data_test.groupby('OpenStatus')
for group in grouped:
    print(group[0], group[1].shape)

print('Saved in data/')
