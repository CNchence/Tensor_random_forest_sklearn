import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree
import pydotplus

print('begin at '+ datetime.now().strftime('%H:%M:%S'))
data_dir = "J:\\D2CO_dataset\\images\\7.9\\"
train_dir = data_dir + 'train\\'
X = np.load(train_dir + "pos_feature_data_array.npy")
print('finish load feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

neg_X = np.load(train_dir + "neg_feature_data_array.npy")
print('finish load neg_feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

y = np.array([1]*len(X) + [0]*len(neg_X))
X = np.concatenate((X, neg_X), axis = 0)

clf = RandomForestClassifier(n_estimators = 200, max_depth = 10, oob_score = True,
    min_samples_split = 2, random_state = 10, verbose=10)

#t_size=0.09#测试集所占比例，分数
#train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=t_size, random_state=10)

print('begin to train at '+ datetime.now().strftime('%H:%M:%S'))
#clf.fit(train_X,train_Y)
clf.fit(X,y)
print('end train at '+ datetime.now().strftime('%H:%M:%S'))


#scores = cross_val_score(clf, X, y)
joblib.dump(clf,data_dir + 'classifier_trees.model')

print('end at '+ datetime.now().strftime('%H:%M:%S'))
