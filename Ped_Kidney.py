import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



##dec=pd.read_csv('C:\\Users\\gaibo\\OneDrive\\Documents\\Python\\Kidney_disc.csv', engine='python')
####if you wanna work with catergorical variables only

dec=pd.read_csv('C:\\Users\\gaibo\\OneDrive\\Documents\\Python\\Ped_Kidney.csv', engine='python')
##if you wanna work with the continous variables and categorical variables

dec=dec.dropna()
##remove all null values


##print (dec['fail'].value_counts())
######shows how many fail vs transplant success
##sns.countplot(dec['fail'])
##plt.show()
####if you wanna see graph of fail vs success

##print(dec.info())
####get info on all variables/predictors
##print(dec.isnull().sum())
####sum of null, make sure no null values present

result=LabelEncoder()
dec['fail']=result.fit_transform(dec['fail'])
####fit_transform to normalize data, not needed if fail gives binary/ discrete outcome only


####Work with data sets
X=dec.drop('fail', axis=1)
## these are all the predictors, might also wanna drop kdri as it bias the learning

y=dec['fail']
## outcome

##train/test
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33, random_state=42)
##randomize train and test sets, test_size=0.33 means 33% will be in the test set, 67 in training

##scale data using scalar, very important for continous, not needed for discrete data
sc=StandardScaler()

##fit transform
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
##not fit transform, don't wanna over fit the test set


#### 3 different models, pick your fighter!
#### random forest needs the imported classifier from sk learn
####MLPClassifier is a neural network classifier (multi layer)
#### svm is support vector model, which contains the classifier svc 

#### use confusion matrix/classification_report to bascially test how good model is with our split 

rfc=RandomForestClassifier(n_estimators=200)
## 'how many trees in forest', can play around with the n_estimator
rfc.fit(X_train, y_train)
## fit with the train model (X=input, y= output or fail)
pred_rfc=rfc.predict(X_test)
## predict with the test input
##print(pred_rfc[:20])
#### predicts how many fail vs success in the first 20 test profiles

##print(classification_report(y_test, pred_rfc))
#### shows accuracy, precision, recall, fl score, support
##print(confusion_matrix(y_test, pred_rfc))
####show tp, tn, fp, fn

###### build a regression with random forest
##rfc_dec=rfc.decision_function(X_test)
##
###### get roc with random forest
##rf_fp, rf_tp, threshold=roc_curve(y_test, rfc_dec)
##auc_rf=auc(rf_fp, rf_tp)


support=svm.SVC()
support.fit(X_train, y_train)
pred_support=support.predict(X_test)


print(classification_report(y_test,pred_support))
print(confusion_matrix(y_test, pred_support))

##build a regression with support vector
dec_support=support.decision_function(X_test)
##note .decision_function shows distance (pos/neg) of X_test from modeled plane

####get roc with support vector
sv_fp, sv_tp, threshold=roc_curve(y_test, dec_support)
auc_sv=auc(sv_fp, sv_tp)
print(auc_sv)

####build a regression with LogisticRegression
log=LogisticRegression()
log.fit(X_train, y_train)
pred_log=log.decision_function(X_test)


######get roc with logistic regression:
log_fp, log_tp, threshold=roc_curve(y_test, pred_log)

##this is our false pos, true positive, and threshold
auc_log=auc(log_fp, log_tp)
##print(auc_log)


######MLPC Classifier
##mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
##mlpc.fit(X_train, y_train)
##pred_mlpc=mlpc.predict(X_test)
####print(mlpc, classification_report(y_test, pred_mlpc))
####print(confusion_matrix(y_test, pred_mlpc))

##ml_dec=mlpc.decision_function(X_test)
##ml_fp, ml_tp, threshold=roc_curve(y_test, ml_dec)
##
##auc_mlp=auc(ml_fp, ml_tp)

plt.plot(sv_fp, sv_tp, linestyle='-', label='SVM (AUC=)'+ str(auc_sv))
plt.plot(log_fp, log_tp, marker='.', label='logistic (AUC=)'+ str(auc_log))
##plt.plot(ml_fp, ml_tp, marker='1', label='MLP AUC='+str(auc_mlp))

plt.title('C-Statistic for Machine Learning Model')
plt.xlabel('sensitivity -->>')
plt.ylabel('1 - specificity -->>')
plt.legend()
plt.show()




