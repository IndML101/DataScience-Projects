#! /usr/bin/env python3

import sys, os
import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sb

# from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

########## Load and explore Loan Prediciton dataset ##########
ML_path = '/othr/data/ML-Project/'
prob_name = 'Loan-prediction/'

dftrain = pd.read_csv(ML_path+prob_name+'data/train.csv')
dftest = pd.read_csv(ML_path+prob_name+'data/test.csv')

os.chdir(ML_path+prob_name+'data')

cols = dftrain.columns
cols.shape

numcols = [col for col in dftrain.columns if dftrain[col].dtype != 'object']

dftrain.info()
dftrain[cols[:7]].head(10)
dftrain[cols[7:]].head(10)

sb.set(color_codes=True)
sb.set_style('darkgrid')

# credit history value counts
df_ch = dftrain['Credit_History'].value_counts(ascending=True)

# probability of getting loan based on credit history
df_prob = dftrain.pivot_table(values='Loan_Status',index=['Credit_History']\
    ,aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

# crosstab of credit history and loan status
chVls = pd.crosstab(dftrain['Credit_History'], dftrain['Loan_Status'])

# crosstab of credit history and gender with loan status
ch_gen_ls = pd.crosstab([dftrain['Credit_History'], dftrain['Gender']], dftrain['Loan_Status'])

# crosstab of married and dependents with loan status
temp = pd.crosstab([dftrain.Married, dftrain.Dependents], dftrain.Loan_Status)

# pivot table of married vs dependents with probabiulity of getting loan as values
temp = dftrain.pivot_table(values='Loan_Status',index=['Married', 'Dependents'],\
    aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())


# count number of missing values in each columns
dftrain.apply(lambda x: sum(x.isnull()),axis=0)

# impute self employed missing values with distribution mode
dftrain['Self_Employed'].value_counts()
dftrain['Self_Employed'].fillna('No',inplace=True)

# pivot table self employed vs education with median loan amount as values
# table to impute missing values of laon amount
table = dftrain.pivot_table(values='LoanAmount',index='Self_Employed', \
    columns='Education', aggfunc=np.median)

# function that returns loan amount to be imputed
def Loan_median(x):
    return table.loc[x['Self_Employed'], x['Education']]

# impute loan amount
temp = dftrain[dftrain['LoanAmount'].isnull()].apply(Loan_median, axis=1)
dftrain['LoanAmount'].fillna(temp, inplace=True)

# scale feature to reduce outliers
dftrain['LoanAmount_log'] = np.log(dftrain['LoanAmount'])
dftrain['TotalIncome'] = dftrain['ApplicantIncome'] + dftrain['CoapplicantIncome']
dftrain['TotalIncome_log'] = np.log(dftrain['TotalIncome'])

# impute missing values of categorical data
dftrain['Gender'].fillna(dftrain['Gender'].mode()[0], inplace=True)
dftrain['Married'].fillna(dftrain['Married'].mode()[0], inplace=True)
dftrain['Dependents'].fillna(dftrain['Dependents'].mode()[0], inplace=True)
dftrain['Loan_Amount_Term'].fillna(dftrain['Loan_Amount_Term'].mode()[0], inplace=True)
dftrain['Credit_History'].fillna(dftrain['Credit_History'].mode()[0], inplace=True)

# generate new features
dftrain['LA/TI'] = dftrain['LoanAmount'] / dftrain['TotalIncome']

# label encode categorical data
var_mod = [col for col in dftrain.columns if dftrain[col].dtype == 'object']
var_mod.remove('Loan_ID')
le = LabelEncoder()
for col in var_mod:
    dftrain[col] = le.fit_transform(dftrain[col])\

# impute missing values in test data (output)

dftest['Self_Employed'].fillna(dftest['Self_Employed'].mode()[0], inplace=True)
dftest['Gender'].fillna(dftest['Gender'].mode()[0], inplace=True)
dftest['Married'].fillna(dftest['Married'].mode()[0], inplace=True)
dftest['Dependents'].fillna(dftest['Dependents'].mode()[0], inplace=True)
dftest['Credit_History'].fillna(dftest['Credit_History'].mode()[0], inplace=True)
dftest['Loan_Amount_Term'].fillna(dftest['Loan_Amount_Term'].mode()[0], inplace=True)

# Generating new features
dftest['LoanAmount_log'] = np.log(dftest['LoanAmount'])
dftest['TotalIncome'] = dftest['ApplicantIncome'] + dftest['CoapplicantIncome']
dftest['TotalIncome_log'] = np.log(dftest['TotalIncome'])
dftest['LA/TI'] = dftest['LoanAmount'] / dftest['TotalIncome']

# imputing feature Loan Amount
table = dftest.pivot_table(values='LoanAmount',index='Self_Employed', \
    columns='Education', aggfunc=np.median)

temp = dftest[dftest['LoanAmount'].isnull()].apply(Loan_median, axis=1)
dftest['LoanAmount'].fillna(temp, inplace=True)

########## preprocess categorical in test data (output) ##########

for col in var_mod:
    if col != 'Loan_Status':
        dftest[col] = le.fit_transform(dftest[col])

########## Training models ##########

# define x (independent) and y (dependent) variables
feature_lst = list(cols)
feature_lst.remove('Loan_ID')
feature_lst.remove('Loan_Status')
x = dftrain[feature_lst]
y = dftrain['Loan_Status']

# splitting data into training and cross validation set
xtrain, xcv, ytrain, ycv = train_test_split(x, y, test_size=0.3, random_state=10)

# training Logistic Regression
LR = LogisticRegression()
LR.fit(xtrain, ytrain)
yLR = LR.predict(xcv)

# function to calculate performance measure of different learning algorithms
def calculate_performance_metric(yvalidation, ypred):
    print('Classification Report:\n\n', classification_report(yvalidation, ypred))
    print('Confusion Matrix:\n\n', confusion_matrix(yvalidation, ypred))
    print('Accuracy:\n\n', accuracy_score(yvalidation, ypred))


calculate_performance_metric(ycv, yLR)
