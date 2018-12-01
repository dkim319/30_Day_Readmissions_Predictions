# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:12:17 2018

@author: dkim
"""

import psycopg2
from sqlalchemy import create_engine
import pandas as pd


### 
# create a database engine 
# to find the correct file path, use the python os library:
# import os
# print(os.getcwd())
#
###

#engine = create_engine('postgresql://postgres:Droid319@localhost/mimic')
#pd.read_sql("SELECT * FROM mimiciii.admissions", engine)

#from sqlalchemy import Table, MetaData, create_engine

#conn = psycopg2.connect(host="localhost",database="mimic", user="postgres", password="Droid319")

#conn = psycopg2.connect("dbname=postgres user=postgres password=Droid319")


# ETL

'''
Using the MIMICIII postgresql database, the inital dataset is generated that contains transformed dataset preppred for machine learning

The following steps were completed to generate this dataset
1.  see document

One observation of this dataset is that it is not as clean as I intital expected.  
However, I have run into similar issues with healthcare datasets in my prior work experience
'''

# Query the data to pull the intital dataset using the psycopg2 library
# Some data clean-up steps have been left to be completed after querying the data
conn = psycopg2.connect(host="localhost",database="mimic", user="postgres", password="Droid319", port=5433)

data_query = open('initial dataset.sql', 'r')
data = pd.read_sql_query(data_query.read(), conn)

secondary_diag_query = open('secondary diagnoses.sql','r')
sec_diag_data = pd.read_sql_query(secondary_diag_query.read(), conn)

secondary_proc_query = open('secondary procedures.sql','r')
sec_proc_data = pd.read_sql_query(secondary_proc_query.read(), conn)

# Review the dataset 
print(data.head(5))

# Remove the data that is no longer needed such as the subject_id.  Hadm_id will be used for joining
# Also, remove admittime and dischtime since they are not useful since the values have been scrambled
data = data.drop(['subject_id','admittime','dischtime'], axis = 1)

# Confirm that the columns have been removed
print(data.head(5))

# Review the statistics of some of the non-categorical variables
print(data.describe())

# The first problem is that there are length of stays that are negative
# The assumption that it is a byproduct of scrambling the data
print('The number of los_days with negative values:')
print(data[data['los_days'] < 0]['los_days'].count())
print(data[data['emergency_hours'] < 0]['los_days'].count())
data = data[data['los_days'] >=0]
#data = data[data['emergency_hours'] >=0]

print(data.shape)

# There are NaN values for emergency_hours, icu_days, and proc_count

print('Replace Nans with 0 for emergency_hours, icu_days, and proc_count')

data['emergency_hours'] = data['emergency_hours'].fillna(0)
data['icu_days'] = data['icu_days'].fillna(0)
data['proc_count'] = data['proc_count'].fillna(0)

print(data.describe())

# The max age is 311, which looks like an error or byproduct of data scrambling
# Any unrelatistic age (150) with the median
age_median = data[data['age'] < 150]['age'].median()
data['age'] = data['age'].apply(lambda x: age_median if x > 150 else x)

print(data.describe())

# remove speical characters from primary_diag and primary_proc
import re

data['admission_location'] = data['admission_location'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
data['discharge_location'] = data['discharge_location'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
data['drg'] = data['drg'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
data['primary_diag'] = data['primary_diag'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
data['primary_proc'] = data['primary_proc'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))

# remove invalid encounters such as newborn and expired

data = data[data['admission_type'] != 'NEWBORN']
data = data[data['hospital_expire_flag'] == 0]
data = data[data['cancer_flag'] == 0]

######################################################################################
# Data Analysis
admissiontype_groupby = data.groupby(['admission_type'])['readmitted_flag'].sum() / data.groupby(['admission_type'])['readmitted_flag'].count()

print(admissiontype_groupby.sort_values(ascending=0))

discharge_groupby = data.groupby(['discharge_location'])['readmitted_flag'].sum() / data.groupby(['discharge_location'])['readmitted_flag'].count()

print(discharge_groupby.sort_values(ascending=0))

insurance_groupby = data.groupby(['insurance'])['readmitted_flag'].sum() / data.groupby(['insurance'])['readmitted_flag'].count()

print(insurance_groupby.sort_values(ascending=0))

age_groupby = data.groupby(['age'])['readmitted_flag'].sum() / data.groupby(['age'])['readmitted_flag'].count()

print(age_groupby.sort_values(ascending=0))

maritalstatus_groupby = data.groupby(['marital_status'])['readmitted_flag'].sum() / data.groupby(['marital_status'])['readmitted_flag'].count()

print(maritalstatus_groupby.sort_values(ascending=0))

maritalstatus_groupby = data.groupby(['marital_status'])['readmitted_flag'].sum() / data.groupby(['marital_status'])['readmitted_flag'].count()

print(maritalstatus_groupby.sort_values(ascending=0))

#print(data.groupby(['language'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))
#print(data.groupby(['religion'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))
#print(data.groupby(['ethnicity'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))

print(data.groupby(['primary_diag'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))

print(data.groupby(['primary_proc'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))

# age/icu/emergency chart

data.groupby(['los_days'])['readmitted_flag'].sum()
data.groupby(['emergency_hours'])['readmitted_flag'].sum()
data.groupby(['icu_days'])['readmitted_flag'].sum()

print(data.groupby(['diag_count'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))
print(data.groupby(['proc_count'])['readmitted_flag'].sum().sort_values(ascending=0).head(10))


######################################################################################
# ML Data Preparation

#data_new = pd.concat((data.drop(['amenities','amenities_split'],axis=1), pd.get_dummies(data_new['amenities_split'].apply(pd.Series).stack(),prefix='a', prefix_sep='_').sum(level=0)),axis = 1)

print(data.head(5))

# remove ethnicity, religion, language
data = data.drop(['language','ethnicity','religion','diagnosis','dob'], axis = 1)

data_new = data

# create dummies variables for the categorical variables

cat_vars = ['admission_type','admission_location','discharge_location','insurance','marital_status','drg','primary_diag','primary_proc']

for c in cat_vars:
    # for each cat add dummy var, drop original column
    data_new = pd.concat([data_new.drop([c], axis=1), pd.get_dummies(data_new[c], prefix=c, prefix_sep='_', drop_first=True, dummy_na=True)], axis=1)


# generate matrices with all the secondary diagnoses and procedures and join then to the primary dataset
    
print('Convert the secondary diagnoses and procedures into matrices that can be joined')
print('Join in the secondary diagnoses and procedures')

# remove special characters
sec_diag_data['long_title'] = sec_diag_data['long_title'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
sec_diag_data['count'] = 1
sec_diag_data = sec_diag_data.pivot(index='hadm_id', columns = 'long_title', values='count')

sec_proc_data['long_title'] = sec_proc_data['long_title'].apply(lambda x : re.sub('[^a-zA-Z0-9-_*.]', '', str(x)))
sec_proc_data['count'] = 1
sec_proc_data = sec_proc_data.pivot(index='hadm_id', columns = 'long_title', values='count')

data_new.merge(sec_proc_data, left_on='hadm_id', right_index=True, how='left')
data_new.merge(sec_diag_data, left_on='hadm_id', right_index=True, how='left')

# fill all nans with 0
data_new = data_new.fillna(0)
data_new = data_new.reset_index()
# remove hadm_id.  it is no longer needed
data_new = data_new.drop(['hadm_id','index'], axis = 1)

# Machine learning 

# The inital dataset will use all the fields except for the secondary diagnoses and procedures
data_one = data_new.loc[:,'los_days':'primary_proc_nan']
data_two = data_new
# split the data into training and test

features = data_one.drop('readmitted_flag', axis = 1)
target = data_one['readmitted_flag']

features_two = data_two.drop('readmitted_flag', axis = 1)
target_two = data_two['readmitted_flag']

# check the target 
print(target.sum()/target.count())

# the target is heavily skewed since there is a small prorportion of readmissions
# use stratified shuffle split to ensure that the training and testing have the same proportion of readmissions

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 1, test_size=0.4, random_state =319)

for train_index, test_index in sss.split(features,target):
    #features_train = features.iloc[train_index]
    features_train, target_train = features.iloc[train_index], target.iloc[train_index]
    features_test, target_test = features.iloc[test_index], target.iloc[test_index]
    
    features_train_two, target_train_two = features.iloc[train_index], target.iloc[train_index]
    features_test_two, target_test_two = features_two.iloc[test_index], target_two.iloc[test_index]
    
# verify that the stratified 
print(target_train.sum()/target_train.count())
print(target_test.sum()/target_test.count())



# Machine Learning
# baseline
# since the readmission prorporation to non-readmission is so low, the baseline assumes that there will no readmissions 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def print_evaluation_metrics(target_train, pred_train, target_test, pred_test):
    #baseline = (1-(target_train.sum()/target_train.count()))
    baseline_test = (1-(target_test.sum()/target_test.count()))
    
    acc_train = accuracy_score(target_train, pred_train)       
    acc_test = accuracy_score(target_test, pred_test)
    auc_test = roc_auc_score(target_test, pred_test)
    con_fus_test = confusion_matrix(target_test, pred_test)
    
    print('baseline is ' + str(baseline_test))
    print('train accuracy: ' + str(acc_train))
    print('test accuracy: ' + str(acc_test))
    print('test auc_score: ' + str(auc_test))
    print('test confusion matrix: ')
    print(con_fus_test)

# log reg
lr_model = LogisticRegression(solver = 'saga', random_state=319)
lr_model = lr_model.fit(features_train, target_train)

pred_train = lr_model.predict(features_train)
pred_test = lr_model.predict(features_test)

print_evaluation_metrics(target_train, pred_train, target_test, pred_test)

# rf
rf_model = RandomForestClassifier()
rf_model = rf_model.fit(features_train, target_train)

pred_train = rf_model.predict(features_train)
pred_test = rf_model.predict(features_test)

print_evaluation_metrics(target_train, pred_train, target_test, pred_test)

ada_model = AdaBoostClassifier()
ada_model = ada_model.fit(features_train, target_train)

pred_train = ada_model.predict(features_train)
pred_test = ada_model.predict(features_test)

print_evaluation_metrics(target_train, pred_train, target_test, pred_test)

import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgboost = xgb_model.fit(features_train, target_train)

pred_train = xgboost.predict(features_train)
pred_test = xgboost.predict(features_test)

print_evaluation_metrics(target_train, pred_train, target_test, pred_test)


# parameter tuning using grid search
def tune_xgboost_parameters(features_train, target_train, features_test, target_test, est, lr, depth):
    # initalize variables
    acc_list_train = []
    acc_list_test = []
    
    est_list = []
    lr_list = []
    dep_list = []
    
    n_est = 0
    results = pd.DataFrame()

    for e in est:
        for l in lr:
            for d in depth:
                xgboost_model = xgb.XGBClassifier(n_estimators = e, learning_rate = l, max_depth = d)#, random_state = 319)
                
                xgboost_model.fit(features_train, target_train)
                pred_test = xgboost_model.predict(features_test)
                pred_train = xgboost_model.predict(features_train)
                
                acc_train = accuracy_score(target_train, pred_train)       
                
                acc_test = accuracy_score(target_test, pred_test)
    
                print (acc_train)
                print (acc_test)
                
                est_list.append(e)
                lr_list.append(l)
                dep_list.append(d)
                acc_list_train.append(acc_train)
                acc_list_test.append(acc_test)
    
    results = pd.DataFrame(
            {'est': est_list,
            'lr': lr_list,
            'depth': dep_list,
            'acc_train': acc_list_train,
            'acc_test': acc_list_test
            })
    
    results.to_csv('results.csv')
    return results, pred_train, pred_test

def select_xgboost_features(features_train, target_train, features_test, target_test, final_est, final_lr, final_depth, per):
    # feature selection
    # perform 
    from sklearn import feature_selection
    
    acc_list_train = []
    acc_list_test = []
    
    per_list = []
    
    percentile = per
    
    # identify the percentile that will produce the best results 
    for per in percentile:
        
        # intilaize SelectFromModel using thresh
        fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = per)
        feature_model =  fs.fit(features_train,target_train)
    
        features_train_new = feature_model.transform(features_train)
        features_test_new = feature_model.transform(features_test)
    
        xgboost_model = xgb.XGBClassifier(n_estimators = final_est, learning_rate = final_lr, max_depth = final_depth)#, random_state = 319)
        
        xgboost_model.fit(features_train_new, target_train)
        pred_test = xgboost_model.predict(features_test_new)
        pred_train = xgboost_model.predict(features_train_new)
        
        acc_train = accuracy_score(target_train, pred_train)       
        acc_test = accuracy_score(target_test, pred_test)
    
        print (acc_train)
        print (acc_test)
        
        per_list.append(per)
        acc_list_train.append(acc_train)
        acc_list_test.append(acc_test)
    
    per_results = pd.DataFrame(
            {'per': per_list,
            'acc_train': acc_list_train,
            'acc_test': acc_list_test
            })
    
    per_results.to_csv('per_results.csv')
    return per_results, pred_train, pred_test

#est = [10,20,30,40,50,50,100,150,200,500,1000]
#lr = [0.01,0.001,0.0001]
#depth = [4,5,6,7,8,9,10]


#est = [130,140,150,160,170]
#lr = [0.01]
#depth = [4,5,6]

est = [30]
lr = [0.1]
depth = [6]

data_one_tuned_results, data_one_pred_train, data_one_pred_test = tune_xgboost_parameters(features_train, target_train, features_test, target_test, est, lr, depth)
print(data_one_tuned_results)

print_evaluation_metrics(target_train, data_one_pred_train, target_test, data_one_pred_test)

final_est = 30
final_lr = 0.1
final_depth = 6

# very minimal increase in testing accuracy

per = range(1,100)
#per = [10]
    
data_one_features_selections_results, data_one_pred_train, data_one_pred_test = select_xgboost_features(features_train, target_train, features_test, target_test, final_est, final_lr, final_depth, per)
print(data_one_features_selections_results)

print_evaluation_metrics(target_train, data_one_pred_train, target_test, data_one_pred_test)
'''
# model two
# with secondary diagnosis and procedusre

est = [30]
lr = [0.1]
depth = [5]

data_two_tuned_results, data_two_pred_train, data_two_pred_test = tune_xgboost_parameters(features_train_two, target_train_two, features_test_two, target_test_two, est, lr, depth)
print(data_one_tuned_results)

print_evaluation_metrics(target_train, data_two_pred_train, target_test, data_two_pred_test)

final_est = 30
final_lr = 0.1
final_depth = 5

# very minimal increase in testing accuracy

#per = range(1,100)
per = [12]
    
data_two_features_selections_results, data_two_pred_train, data_two_pred_test = select_xgboost_features(features_train_two, target_train_two, features_test_two, target_test_two, final_est, final_lr, final_depth, per)
print(data_two_features_selections_results)

print_evaluation_metrics(target_train, data_two_pred_train, target_test, data_two_pred_test)

# this is a parameter tuning of a logisitic regression model using gridsearch CV
def fit_model_xgboostCV(X, y):
    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20)

    # initialize the model
    xgboost = xgb.XGBClassifier(silent=False)

    # tune and fit the model
    params = {'n_estimators': [50,60,70,80,90,100,150,200,300,400,500,1000,1500,2000]#[50,60,70,80,90,100,150,160,170,180,190,200,210,220,230,240,250,300,500,1000]#[60]
                    ,'learning_rate': [0.1,0.01,0.001,0.0001]#[0.001]
                    , 'max_depth': [4,5,6,7,8,9,10]#range(1,8)
                    , 'min_child_weight': [1]
                    #, 'max_delta_step': range(1,10)
                    #, 'n_jobs': [4]
                    #, 'gamma': range(0,3)
                   # , 'random_state':[319]
                    }

    # tune and fit the model
    grid = GridSearchCV(estimator = xgboost, param_grid = params, cv = cv_sets)
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_
'''

















#cursor = conn.cursor()

#cursor.execute('select * from mimiciii.admissions')

#test = pd.read_csv(cursor.fetchall())

#db_version = cur.fetchone()
#print(db_version)



#cursor.execute("SET search_path TO mimiciii, public")
#cursor.execute('select * from mimiciii.admissions')


#engine = create_engine("postgresql://postgres:Droid319@localhost/postgres")

#conn = engine.connect()

#conn.execute("SET search_path TO mimic, public")
#cursor = conn.cursor()
#cursor.execute('select * from mimiciii,admissions')





#engine = create_engine("postgresql://postgres:Droid319@localhost/")



#results = engine.execute("select * from mimiciii.admissions")

#conn = engine.connect()

#conn.execute("SET search_path TO mimic, public")

#conn = psycopg2.connect(host="localhost",database="mimic", user="postgres", password="Droid319")

#cursor = conn.cursor()

#cursor.execute('set search_path to mimic, public')
#cursor.execute('select * from admissions')

#cur = conn.cursor()

#cur.execute("select * from mimiciii.prescriptions")

#conn = psycopg2.connect("dbname=mimic user="" password=Droid319")

#try:
#    conn = psycopg2.connect("dbname='mimic' user='postgresql' host='localhost' password='Driod319'")
#except:
#    print ("I am unable to connect to the database")