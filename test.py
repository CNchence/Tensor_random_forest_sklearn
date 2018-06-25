from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn import tree

print('begin at '+ datetime.now().strftime('%H:%M:%S'))

X = np.load("pos_feature_data_array.npy")
print('finish load feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

neg_X = np.load("neg_feature_data_array.npy")
print('finish load neg_feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

y = np.array([1]*len(X) + [0]*len(neg_X))
X = np.concatenate((X, neg_X), axis = 0)

clf = joblib.load('first_tree.model')
y_result = clf.predict(X)





#y_result = clf.predict_proba(X)
pos_neg = 0
neg_pos = 0
for i in range(0,len(y_result)):
    if y[i] == 1 and y_result[i] == 0 :
        pos_neg += 1
    elif y[i] == 0 and y_result[i] == 1 :
        neg_pos += 1
print(pos_neg)
print(neg_pos)
#print(test_Y)

print('end at '+ datetime.now().strftime('%H:%M:%S'))