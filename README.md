# Predicting Closed Questions on Stack Overflow
## Data Mining - Universidad Simoon Bolivar Jul 2018

### Data Creation
Due to space, data is not present completely in this repository. To recreate
test and train sets and each fold, please run the following commands after uncompressing ```csv_data.tar.gz``` file
(shipped separately) inside repository folder.

There are 3 compressed files:
- ```train_complete_preprocessed.csv``` which contains all data after extracting
attributes.
- ```train-sample-preprocessed.csv``` a small sample of the previous file.
- ```train_sample_preprocessed_categorical.csv``` the same sample from above but with target class as categorical. Might be useful for Weka.


Commands are:
```bash
  $ python preprocess_and_split.py
  $ python extract_folds.py
```

First command creates train and testing datasets without outliers and the second creates the folds train and testing subsets inside ```data/``` directory.

For a copy of ```csv_data.tar.gz``` please [email me](mailto:martinezazuaje@gmail.com).

### Data Analysis

All data Analysis can be reviewed by running
Then run:
```bash
  $ python analyze_data.py
```

### Training

```train_lightgbm_crossvalidation.py``` and ```train_rf_crossvalidation.py``` will perform cross-validation using folds inside ```data/``` folder. Previously saved models will be overwritten.

Please note that this repository already contains previously trained models for both RandomForest and LightGBM algorithms inside each fold folder. 

### Last Updated
[Leonardo Martínez](https://github.com/leotms/) - 14/07/2018.
