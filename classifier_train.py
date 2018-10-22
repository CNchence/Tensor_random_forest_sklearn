import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree

print('begin at ' + datetime.now().strftime('%H:%M:%S'))
train_dir = "F:\\D2CO_dataset\\detect_train_data\\1539228326_pos_neg\\"
X = np.load(train_dir + "pos_feature_data_array.npy")
print('finish load feature_data_array at ' + datetime.now().strftime('%H:%M:%S'))

neg_X = np.load(train_dir + "neg_feature_data_array.npy")
print('finish load neg_feature_data_array at ' + datetime.now().strftime('%H:%M:%S'))

y = np.array([1]*len(X) + [0]*len(neg_X))
X = np.concatenate((X, neg_X), axis=0)

clf = RandomForestClassifier(n_estimators=200, max_depth=10, oob_score=True,
    min_samples_split=2, random_state=10, verbose=10)

#t_size=0.09#测试集所占比例，分数
#train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=t_size, random_state=10)

print('begin to train at ' + datetime.now().strftime('%H:%M:%S'))
clf.fit(X, y)
print('end train at ' + datetime.now().strftime('%H:%M:%S'))

feature_importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(feature_importances)[::-1]
print("Feature ranking:")
print(indices)
'''
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))

'''

#scores = cross_val_score(clf, X, y)
joblib.dump(clf, train_dir + 'classifier_forest.model')

print('end at ' + datetime.now().strftime('%H:%M:%S'))
