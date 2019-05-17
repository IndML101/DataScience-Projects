#! /usr/bin/env python3


# Reference Tutorial link
# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/

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

# get list of columns
cols = dftrain.columns
cols.shape

# get info of dataframe structure
dftrain.info()

# get list of columns with float data type
numcols = [col for col in dftrain.columns if dftrain[col].dtype != 'object']

# get first ten records of data frame (total number of columns in data frame is 13)
dftrain[cols[:7]].head(10)
dftrain[cols[7:]].head(10)

########## Exploratory visualization ##########

sb.set(color_codes=True)
sb.set_style('dark')

# histogram of numerical features

dftrain.hist(bins=50, figsize=(7,7))
plt.tight_layout()
plt.savefig('feature-hist.png')
plt.close()

# to plot one feature at a time
'''
dftrain['feature-name'].hist(figsize=(7,7))
plt.tight_layout()
plt.savefig('feature-name-hist.png')
plt.close()
'''

# boxplot of features with numerical value

ax = dftrain.boxplot(figsize=(11,6), fontszie=9, vert=False)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('feature-boxplot-pd.png')
plt.close()

plt.figure(figsize=(10.5,6))
ax = sb.boxplot(data=dftrain, orient='h')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
plt.savefig('feature-boxplot-sb.png')
plt.close()

# boxplot groupby column
dftrain.boxplot(column='ApplicantIncome', by='Education')
plt.savefig('ApplicantIncome-boxplot-byEdu.png')
plt.close()

# boxplot Loan Amount group by Education and Self Employed
ax = dftrain.boxplot(column='LoanAmount', by=['Education', 'Self_Employed'], figsize=(7,7), fontsize=9)
ax.set_ylabel('Loan Amount')
ax.set_xticklabels(ax.getxticklabels(), rotation=45)

fig = ax.get_figure() # remove automatically generated title from pandas boxplot
fig.suptitle('')

plt.tight_layout()
plt.savefig('LoanAmount-grpby-Edu-and-SelfEmployed.png')
plt.close()

# boxplot in two parts
'''
dftrain.boxplot(column=['ApplicantIncome', 'CoapplicantIncome'])
plt.savefig('income-boxplot.png')
plt.close()

dftrain.boxplot(column=['LoanAmount', 'Loan_Amount_Term', 'Credit_History'])
plt.savefig('loan-boxplot.png')
plt.close()
'''

# boxplot with subplots
dftrain.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(10,6))
plt.tight_layout()
plt.savefig('feature-boxplot-subplot.png')
plt.close()

########## Categorical Variable analysis ##########

df_ch = dftrain['Credit_History'].value_counts(ascending=True)

# here probability is equal to arithmatic mean of loan status (binary data) for a credit history
df_prob = dftrain.pivot_table(values='Loan_Status',index=['Credit_History']\
    ,aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

# plotting number of records group by credit history
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
df_ch.plot(kind='bar',ax=ax1)

# plotting probability of getting loan based on credit history
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
df_prob.plot(kind='bar', ax=ax2)

plt.tight_layout()
plt.savefig('LoanStatus-and-CreditHistory.png')
plt.close()

# stacked bar plot showing Loan status(Y/N) based on credit history

chVls = pd.crosstab(dftrain['Credit_History'], dftrain['Loan_Status'])

ax = chVls.plot(kind='bar',stacked=True)
ax.set_xlabel('credit history')
ax.set_ylabel('Applicants')
plt.tight_layout()
plt.savefig('LoanStatus-vs-CreditHistory.png')
plt.close()

# stacked bar plot showing Loan status(Y/N) based on credit history and gender
ch_gen_ls = pd.crosstab([dftrain['Credit_History'], dftrain['Gender']], dftrain['Loan_Status'])
ax = ch_gen_ls.plot(kind='bar',stacked=True)
ax.set_xlabel('credit history and gender')
ax.set_ylabel('Applicants')
plt.tight_layout()
plt.savefig('LoanStatus-vs-CreditHistory-Gender.png')
plt.close()

# stacked bar plot showing loan status of applicants
# group by marriage status and number of dependants
# using pandas crosstab function

temp = pd.crosstab([dftrain.Married, dftrain.Dependents], dftrain.Loan_Status)
ax = temp.plot(kind='bar', stacked=True, figsize=(8,7))
ax.set_ylabel('Applicants')
ax.set_xlabel('Married,Dependents')
plt.tight_layout()
plt.savefig('Applicant-LoanStatus-Vs-Married-and-Dependents.png')
plt.close()

# barplot showing probability of getting loan
# based marriage satatus and number of dependents
# using using pandas pivot table function

temp = dftrain.pivot_table(values='Loan_Status',index=['Married', 'Dependents'],\
    aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

ax = temp.plot(kind='bar', figsize=(8,7))
ax.set_ylabel('Probability of getting loan')
ax.set_xlabel('Married,Dependents')
plt.tight_layout()
plt.savefig('Probability-Vs-Maried-and-Dependents.png')
plt.close()

# crosstab of gender vs married
genVmarr = pd.crosstab(dftrain['Gender'], dftrain['Married'])
# plot stacked bar plot for visualization
##### genVmarr.plot(kind='bar', stacked=True)
# for probability
genVmarr.div(genVmarr.sum(axis=1), axis = 0)

# crosstab like pivot table
mean_la = dftrain.pivot_table(values='LoanAmount', index='Credit_History', columns='Gender',\
     aggfunc=np.mean)

# gender vs dependents example of crosstab
genVdep = pd.crosstab(dftrain['Gender'], dftrain['Dependents'])
genVdep = pd.crosstab(index=dftrain['Gender'], columns=dftrain['Dependents'],\
     values=dftrain['ApplicantIncome'], aggfunc=np.mean)


########## Data munging (data prepration) for modeling ##########

# get number of missing values in each variable
dftrain.apply(lambda x: sum(x.isnull()),axis=0)

# example of how to fill missing values
# fill Loan Amount with mean value
# dftrain['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
# but there is a better solution

# Its better if we fill missing Loan Amount value with median values \
# of distribution group by Education and Self Employment

# fill missing values in Self Employnent with distribution mode
dftrain['Self_Employed'].value_counts()
dftrain['Self_Employed'].fillna('No',inplace=True)

##### fill median of distribution of loan amount group by education and self employed

# plot distribution of loan amount group by self employed and education
dftrain.boxplot(column='LoanAmount', by=['Selpf_Employed', 'Education'])
ax.set_ylabel('Loan Amount')
ax.set_xticklabels(ax.getxticklabels(), rotation=20)

fig = ax.get_figure() # remove automatically generated title from pandas boxplot
fig.suptitle('')

plt.tight_layout()
plt.savefig('LoanAmount-grpby-Edu-and-SelfEmployed-2.png')
plt.close()


# create reference table for median values of each group
table = dftrain.pivot_table(values='LoanAmount',index='Self_Employed', \
    columns='Education', aggfunc=np.median)

# function to return relevent value from this table
def Loan_median(x):
    return table.loc[x['Self_Employed'], x['Education']]

# input series (temp) to replace values
temp = dftrain[dftrain['LoanAmount'].isnull()].apply(Loan_median, axis=1)

#replace missing values
dftrain['LoanAmount'].fillna(temp, inplace=True)


########## Treatng extreme values of variables ##########

# take log of Loan Amount to scale feature values
dftrain['LoanAmount_log'] = np.log(dftrain['LoanAmount'])

# plot histogram of Loan Amount
ax = dftrain['LoanAmount_log'].hist(bins=20)
ax.set_title('Loan Amount Log Histogram')
plt.savefig('LoanAmount-log-hist.png')
plt.close()

# plot boxplot of Loan Amount log
ax = dftrain['LoanAmount_log'].plot(kind='box')
plt.savefig('LoanAmount-log-boxplot.png')
plt.close()

# calculate total income followed by log of total income
dftrain['TotalIncome'] = dftrain['ApplicantIncome'] + dftrain['CoaaplicantIncome']
dftrain['TotalIncome_log'] = np.log(dftrain['TotalIncome'])

# plot histogram of log of total income
ax = dftrain['TotalIncome_log'].hist(bins=20)
ax.set_title('Total Income Log Histogram')
plt.savefig('TotalIncome-log-hist.png')
plt.close()

# plot boxplot of Total Income log
ax = dftrain['TotalIncome_log'].plot(kind='box')
plt.savefig('TotalIncome-log-boxplot.png')
plt.close()


########## Exercise: impute missing values of Gender, Married, ##########
########## Dependents, Loan Amount Term, Credit History ##########
dftrain['Gender'].fillna(dftrain['Gender'].mode()[0], inplace=True)
dftrain['Married'].fillna(dftrain['Married'].mode()[0], inplace=True)
dftrain['Dependents'].fillna(dftrain['Dependents'].mode()[0], inplace=True)
dftrain['Credit_History'].fillna(dftrain['Credit_History'].mode()[0], inplace=True)
dftrain['Loan_Amount_Term'].fillna(dftrain['Loan_Amount_Term'].mode()[0], inplace=True)

########## Generate new features like (Loan Amount/Total Income) ########## 
dftrain['LA/TI'] = dftrain['LoanAmount'] / dftrain['TotalIncome']


########## preprocess categorical in training data ##########
# label encoding
var_mod = [col for col in dftrain.columns if dftrain[col].dtype == 'object']
var_mod.pop(0) # remove Loan ID from list of columns
# or use remove funciton
# var_mod.remove('Loan_ID')

le = LabelEncoder()
for col in var_mod:
    dftrain[col] = le.fit_transform(dftrain[col])

########## Data munging for sample test data (output) ##########
# Imputing missing values in sample test data (output)
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
    dftest[col] = le.fit_transform(dftest[col])


########## Training models ##########

# define x (independent) and y (dependent) variables
feature_lst = cols
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