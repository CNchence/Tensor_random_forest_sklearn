import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree
import pydotplus
import random

def random_without_same(mi, ma, num):
	temp = list(range(mi, ma))
	random.shuffle(temp)
	return temp[0:num]

num = 10000

print('begin at '+ datetime.now().strftime('%H:%M:%S'))
train_array = np.load("L:\\Dataset\\images\\7.7\\train\\feature_data_array.npy")
test_array = np.load("L:\\Dataset\\images\\7.7\\test\\feature_data_array.npy")
#print(X)
used_features = random_without_same(30000,135600,num)
used_features.sort()

position_pose_index = list(range(165600,165610))
used_features = used_features + position_pose_index

print(used_features)

feature_select_train = train_array[:,used_features]
feature_select_test = test_array[:,used_features]


feature_select_train = feature_select_train.T
features_long = len(feature_select_train)
train_Y_T = feature_select_train[features_long - 6:]
train_X = feature_select_train[:features_long - 6]
train_y = train_Y_T.T
train_X = train_X.T

feature_select_test = feature_select_test.T
features_long = len(feature_select_test)
test_Y_T = feature_select_test[features_long - 6:]
test_X = feature_select_test[:features_long - 6]
test_y = test_Y_T.T
test_X = test_X.T



'''
print('finish load feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

clf = RandomForestRegressor(n_estimators = 500, max_depth = 10, oob_score = True,random_state = 10, verbose = 10)

print('begin to train at '+ datetime.now().strftime('%H:%M:%S'))
clf.fit(train_X,train_y)
#clf.fit(X,y)
print('end train at '+ datetime.now().strftime('%H:%M:%S'))
'''
clf = joblib.load('regression_forest_7_7.model')
#y_result = clf.predict_proba(test_X)
y_result = clf.predict(test_X)
print(y_result)
print(test_y)
np.savetxt('reg_estimation.txt', y_result, fmt='%10.4f', newline= '\t\n',delimiter = ' ' )
np.savetxt('reg_real.txt', test_y, fmt='%10.4f', newline= '\t\n',delimiter = ' ' )
'''
dot_data = tree.export_graphviz(clf, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("first_model.pdf")'''

#scores = cross_val_score(clf, X, y)
#joblib.dump(clf,'regression_forest_7_7.model')
#print(scores.mean())
print('end at '+ datetime.now().strftime('%H:%M:%S'))
