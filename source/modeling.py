import pandas as pd
import numpy as np

import seaborn as sns
sns.set(style="white")
import matplotlib.pyplot as plt
import matplotlib as mpl


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold,RepeatedStratifiedKFold,RandomizedSearchCV,cross_val_score

from sklearn.metrics import(accuracy_score, roc_auc_score, f1_score, plot_confusion_matrix, precision_recall_curve, roc_curve,
                            recall_score, confusion_matrix, precision_score, classification_report)

from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC


#global variable
random_state=42

%matplotlib inline

#load data
df=pd.read_csv('./data/churn.csv').drop(columns=['RowNumber','Surname', 'CustomerId', 'Geography'], axis=1)
df.head(3)
#check for Nan
df.isnull().sum()

#data preprocessing

#convert categorical values to numerical
df['Gender']=df['Gender'].apply(lambda x: 0 if x=='Female' else 1)
#split data on test, val and train subset
y = df["Exited"]
X = df.drop(["Exited"], axis = 1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=random_state)
X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

#scale data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)

# count samples for each class
from collections import Counter
counter = Counter(y_train)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

#instantiate models
log_clf = LogisticRegression() 
rnd_clf = RandomForestClassifier() 
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='soft')

#compute accuracy for each model
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))
    
#evaluate models on roc_auc score:
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    print(clf.__class__.__name__, "Mean ROC AUC: %.5f " % (scores.mean()))

#hyperparameters tuning

# define grid
param_grid= {'max_iter':[10,50,100,300], 'C':[0.001,0.01,0.5], 'solver':['newton-cg','lbfgs','liblinear','sag','saga']}
model =LogisticRegression(random_state=67)
# define evaluation procedure
from sklearn.model_selection import GridSearchCV

# define grid search for logistic regressoin
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X_train, y_train)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# define grid search for SVM classifier
param_grid= {'max_iter':[10,50,100,-1], 'C':[0.05,0.01,1.0], 'gamma':['auto'],'kernel':['linear','poly','rbf','sigmoid']}
model =SVC(random_state=67, probability=True)
# define evaluation procedure
from sklearn.model_selection import GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X_train, y_train)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
# define grid search for random forest classsifier
param_grid= {'n_estimators':[100,300,500],'max_depth':[1,3,5,7], 'max_leaf_nodes':[5,15]}
model =RandomForestClassifier(random_state=random_state)
# define evaluation procedure
from sklearn.model_selection import GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X_train, y_train)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#plot roc curve on validation subset to evaluate model perfomance
from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_val, log_clf.predict_proba(X_val)[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_val, rnd_clf.predict_proba(X_val)[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_val, svm_clf.predict_proba(X_val)[:,1], pos_label=1)

#roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_val))]
p_fpr, p_tpr, _ = roc_curve(y_val, random_probs, pos_label=1)

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.figure(1)
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='RandomForest')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='SVC')

plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('AllModelsROC',dpi=200)
plt.show();

plt.figure(2)
plt.xlim(0, 0.4)
plt.ylim(0.4, 1)
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='RandomForest')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='SVC')

plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
#plt.savefig('ROCzoom',dpi=200)
plt.show();

# plot confusion matrix to evaluate Random Forest perfomance
plt.style.use('classic')
disp = plot_confusion_matrix(rnd_clf, X_val, y_val,cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix, without normalization")
plt.show()
plt.savefig('ConfusionMatrixBefore',dsi=200)
# plot precision vs recall on threshold
y_hat=rnd_clf.predict(X_val)
rf_proba=rnd_clf.predict_proba(X_val)
precisions,recalls, thresholds=precision_recall_curve(y_val, rf_proba[:,1])
plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="upper left") 
plt.ylim([0, 1])

#since labels inbalanced with ration 1:4 I need to set decision threshold
threshold=0.5
y_val_60=(rf_proba[:,1] > threshold)
# plot confusion matrix again 
plt.style.use('classic')
disp = plot_confusion_matrix(rnd_clf, X_val, y_val_60, cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix, without normalization")
plt.show()

# since model picked and threshold determinded I can make a prediction on test set
y_hat=rnd_clf.predict(X_test)
rf_proba=rnd_clf.predict_proba(X_test)

threshold=0.5
y_test_60=(rf_proba[:,1] > threshold)

disp = plot_confusion_matrix(rnd_clf, X_test, y_test_60,cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix, without normalization")

plt.show()

#######










