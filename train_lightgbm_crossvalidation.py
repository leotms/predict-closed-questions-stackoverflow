'''
    Author: Leonardo Martinez.
    Last Updated: 13/07/2018.
    Trains LightGBM using crossvalidation.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import mean_squared_error, accuracy_score, confusion_matrix, log_loss, recall_score
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import lightgbm          as lgb

for index in range(1,11):

    print('Importing data %i...'%(index))
    train = pd.read_csv('data/folds/%i/train_%i.csv'%(index, index))
    valid = pd.read_csv('data/folds/%i/test_%i.csv'%(index, index))

    #Separate target variable
    y_train = train['OpenStatus']
    y_valid = valid['OpenStatus']
    del train['OpenStatus']
    del valid['OpenStatus']

    columns = train.columns.tolist()

    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    valid = scaler.transform(valid)


    train_data=lgb.Dataset(train, label=y_train, feature_name = columns)
    valid_data=lgb.Dataset(valid, label=y_valid, feature_name = columns)

    params = {'boosting_type': 'gbdt',
              'max_depth' : 10,
              'objective': 'multiclass',
              'num_class' : 5,
              'nthread': 5,
              'num_leaves': 64,
              'learning_rate': 0.05,
              'metric' : 'multi_logloss'
              }

    #Train model on selected parameters and number of iterations
    lgbm = lgb.train(params,
                     train_data,
                     2500,
                     valid_sets=valid_data,
                     early_stopping_rounds= 40,
                     verbose_eval= 10,
                     )

    # prints data to calculate feature importance
    print(lgbm.feature_name())
    print(lgbm.feature_importance(importance_type='gain'))

    print('Saving model...')
    lgbm.save_model('data/folds/%i/model_lightgbm_%i.txt'%(index, index))

    print('Start predicting...')

    print('Training results:')
    y_pred = lgbm.predict(train, num_iteration=lgbm.best_iteration)
    print('MULTI_LOGLOSS:', log_loss(y_train, y_pred))

    y_pred = [pred.tolist().index(max(pred)) for pred in y_pred]

    print('RMSE:',  mean_squared_error(y_train, y_pred) ** 0.5)
    print('ACCURACY:', accuracy_score(y_train, y_pred))
    print('CONFUSION MATRIX:', confusion_matrix(y_train, y_pred))
    print('RECALL:', recall_score(y_train, y_pred, average='macro'))

    print('----------------------------')
    print('Loading Test set...')
    test   = pd.read_csv('data/test.csv')
    y_test = test['OpenStatus']
    del test['OpenStatus']

    test   = scaler.transform(test)
    y_pred = lgbm.predict(test, num_iteration=lgbm.best_iteration)
    print('Test results:')
    print('MULTI_LOGLOSS:', log_loss(y_test, y_pred))

    y_pred = [pred.tolist().index(max(pred)) for pred in y_pred]

    print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)
    print('ACCURACY:', accuracy_score(y_test, y_pred))
    print('CONFUSION MATRIX:', confusion_matrix(y_test, y_pred))
    print('RECALL:', recall_score(y_test, y_pred, average='macro'))
    print('----------------------------')
