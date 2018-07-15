import matplotlib.pyplot as plt
import pandas   as pd
import numpy    as np
from   datetime import datetime, timedelta
from   pandas.plotting import scatter_matrix

# previus data (with outliers)
# data = pd.read_csv('data/folds/train-sample-preprocessed.csv')
# data without outliers
data = pd.read_csv('data/folds/1/train_1.csv')

print('Removing indexes...')

# 1. Calculando los missing values.
print('Missing Values por columna...')
nulls = data.isnull().sum()
print (nulls[nulls > 0])

attributes = data.columns.tolist()
print('Calculating Histograms...')

for attribute in attributes:
    plt.figure()
    plt.hist(data[attribute], 20, color='#16a085')
    plt.title('Histograma %s'%(attribute))
    plt.xlabel(attribute)
    plt.ylabel('Frecuencia')
    # previus data (with outliers)
    # plt.savefig('graphs/hist/histograma_%s.png'%(attribute))
    # data without outliers
    plt.savefig('graphs/hist_2/histograma_%s.png'%(attribute))
    # plt.show()
    plt.close()

print('Calculating Boxplots...')
for attribute in attributes:
    plt.figure()
    plt.boxplot(data[attribute])
    plt.title('BoxPlot de %s'%(attribute))
    # previus data (with outliers)
    # plt.savefig('graphs/boxplots/boxplot_%s.png'%(attribute))
    # data without outliers
    plt.savefig('graphs/boxplots_2/boxplot_%s.png'%(attribute))
    #plt.show()
    plt.close()

print('Calculating Scatterplots...')
for attribute1 in attributes:
    for attribute2 in attributes:
        plt.figure()
        plt.scatter(data[attribute1], data[attribute2])
        plt.title('Scatterplot entre %s y %s'%(attribute1, attribute2))
        plt.xlabel(attribute1)
        plt.ylabel(attribute2)
        # previus data (with outliers)
        # plt.savefig('graphs/scatterplots/scatterplot_%s_%s.png'%(attribute1, attribute2))
        # data without outliers
        plt.savefig('graphs/scatterplots_2/scatterplot_%s_%s.png'%(attribute1, attribute2))
        #plt.show()
        plt.close()

new_names = [str(attributes.index(attribute)) for attribute in attributes]
data.columns = new_names

print('Calculating correlation...')
scatter_matrix(data, alpha = 0.2, figsize=(10, 10))
plt.title('Correlation Matrix')
# previus data (with outliers)
# plt.savefig('graphs/scatter_matrix.png')
# data without outliers
plt.savefig('graphs/scatter_matrix_2.png')
# plt.show()
plt.close()

fig = plt.figure(figsize=(10, 10))
plt.matshow(data.corr(), fignum=1)
plt.yticks(range(len(data.columns)), data.columns)
plt.xticks(range(len(data.columns)), data.columns, rotation='vertical')
plt.colorbar()
# previus data (with outliers)
# plt.savefig('graphs/matshow.png')
# data without outliers
plt.savefig('graphs/matshow_2.png')
# plt.show()
plt.close()
