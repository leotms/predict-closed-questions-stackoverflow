'''
    Author: Leonardo Martinez.
    Last Updated: 13/07/2018.
    Trains RandomForestClassifier using crossvalidation.
'''
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import mean_squared_error, accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.externals       import joblib
from sklearn.ensemble        import RandomForestClassifier
import pandas            as pd
import numpy             as np

for index in range(1,11):

    print('Importing data of Fold %i...'%(index))
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

    clf = RandomForestClassifier(max_depth=10, random_state=0)

    clf.fit(train, y_train)

    print('Saving model...')
    joblib.dump(clf, 'data/folds/%i/model_rf_%i.pkl'%(index, index))
    # to open model use something like:
    # clf = joblib.load('pah.pkl')

    # prints data to calculate feature importance
    print(columns)
    print(clf.feature_importances_)

    print('Training results:')
    y_pred = clf.predict_proba(train)
    print('MULTI_LOGLOSS:', log_loss(y_train, y_pred))

    y_pred = clf.predict(train)

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
    y_pred = clf.predict_proba(test)

    print('Test results:')
    print('MULTI_LOGLOSS:', log_loss(y_test, y_pred))
    y_pred = clf.predict(test)
    print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)
    print('ACCURACY:', accuracy_score(y_test, y_pred))
    print('CONFUSION MATRIX:', confusion_matrix(y_test, y_pred))
    print('RECALL:', recall_score(y_test, y_pred, average='macro'))
    print('----------------------------')
