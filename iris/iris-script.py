#! /usr/bin/env python3


import sys, os
import numpy as np
import padas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import copy as cp

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

########### performance measure of learning algorithm ###########

# function to calculate performance measure of different learning algorithms
def calculate_performance_metric(yvalidation, ypred):
    print('Classification Report:\n\n', classification_report(yvalidation, ypred))
    print('Confusion Matrix:\n\n', confusion_matrix(yvalidation, ypred))
    print('Accuracy:\n\n', accuracy_score(yvalidation, ypred))


datapath = '/othr/data/ML-Projects/iris/data'
os.chdir(datapath)


########### reading and manipulating data using data frames ###########

irisdata = pd.read_csv('iris-data.csv')
data = cp.deepcopy(irisdata)
data.head()
print('\n\n')
data.shape()
print('\n\n')
data.describe()
print('\n\n')
data.info()
print('\n\n')

numcol = []
specieslst = []

for col in data.columns:
    if data[col].dtype == 'float64':
        numcol.append(col)

for itm in data['class'].unique():
    specieslst.append(itm)

data.groupby('class').size()
print('\n\n')

########### visualizing(exploratory) data using data frames and matplotlib ###########

sb.set(color_codes=True)

# Visualization of data distribution boxplot and histograms
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('feature-boxplot-subplots.png')

data.plot(kind='box')
plt.savefig('feature-boxplot.png')

data.boxplot(by='class', figsize=(15,15))
plt.savefig('feature-boxplot-class.png')

data.hist()
plt.savefig('feature-hist.png')

# class wise feature boxplot
fig, axs = plt.subplots(nrows=len(numcol),ncols=len(specieslst),figsize=(15,15))

for i in range(len(numcol)):
    for j in range(len(specieslst)):  
        print(numcol[i]," : ",specieslst[j])
        axs[i,j].boxplot(df[numcol[i]][df['class']==specieslst[j]])
        axs[i,j].set_xticklabels([numcol[i]+"  "+specieslst[j]])
plt.savefig('classwise-feature-boxplot.png')

# pairplots which describe all the data
pd.scatter_matrix(data, figsize=(15,10))
plt.savefig('pd-scatter-plot.png')

sb.pairplot(data, size=2, diag_kind='hist')
plt.savefig('pair-plot.png')

sb.pairplot(data, size=2, diag_kind='kde')
plt.savefig('pair-plot-kde.png')

sb.pairplot(data, diag_kind='hist', hue='class')
plt.savefig('pair-plot-hist-hue.png')

sb.pairplot(data, diag_kind='kde', hue='class')
plt.savefig('pair-plot-kde-hue.png')

########### Split data into trai=ning and validation(test) set ###########

arr = data.values
x = arr[:,0:4]
y = arr[:,4]
validation_size = 0.2
seed = 7
xtrain, xvalidation, ytrain, yvalidation = train_test_split(x, y, test_size=validation_size, random_state=seed)


########### evaluation of training models ###########

# Test options and evaluation metric
numfolds = 10
numinstances = len(xtrain)
scoring = 'accuracy'

#Here we are testing various predictive algorithms from scikit-learn
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kf = KFold(n_splits=numfolds, random_state=seed, shuffle=True)
    kf = kf.split(x)
    kf = list(kf)
    # above two lines can be removed, results will be same.
    cv_results = cross_val_score(model, x, y, cv=kf, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


########### Training regression model ###########

LR = LogisticRegression()
LR.fit(xtrain, ytrain)
ylr = LR.predict(xvalidation)

svmModel = SVC()
svmModel.fit(xtrain, ytrain)
ysvm = svmModel.predict(xvalidation)


########### Performance metrics ###########

calculate_performance_metric(yvalidation, ylr)
calculate_performance_metric(yvalidation, ysvm)

# ROC curve for two class classification

'''
logit_roc_auc = roc_auc_score(yvalidation, ylr)
fpr, tpr, thresholds = roc_curve(yvalidation, LR.predict_proba(xvalidation)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
'''

'''
for multiclass classification refer following link:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
