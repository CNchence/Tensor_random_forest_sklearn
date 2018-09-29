import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree
feature_index_num = []
now_index = 0
for i in range(575, -1, -1):
    now_index += i
    feature_index_num.append(now_index)
print(feature_index_num)
clf = joblib.load('F:/D2CO_dataset/detect_train_data/1538123396_pos_neg/classifier_forest.model')
feature_importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(feature_importances)[::-1]
print("Feature ranking:")
f = 0
feature_index = []
while feature_importances[indices[f]] != 0:
    feature_index.append(indices[f])
    f += 1
print("feature size: ",len(feature_index))

print(feature_index)

print('end at ' + datetime.now().strftime('%H:%M:%S'))
