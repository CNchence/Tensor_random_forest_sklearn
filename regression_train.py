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
X = np.load("L:\\Dataset\\images\\7.3\\train\\feature_data_array.npy")
#print(X)
used_features = random_without_same(0,165600,num)
used_features.sort()

position_pose_index = list(range(165600,165610))
used_features = used_features + position_pose_index

feature_select_X = X[:,used_features]


feature_select_X = feature_select_X.T
features_long = len(feature_select_X)
Y_T = feature_select_X[features_long - 6:]
feature_select_X = feature_select_X[:features_long - 6]
y = Y_T.T
feature_select_X = feature_select_X.T


print('finish load feature_data_array at '+ datetime.now().strftime('%H:%M:%S'))

clf = RandomForestRegressor(n_estimators = 500, max_depth = 10, oob_score = True,random_state = 10, verbose = True)

t_size=0.1#测试集所占比例，分数
train_X, test_X, train_Y, test_Y = train_test_split(feature_select_X, y, test_size=t_size, random_state=10)

print('begin to train at '+ datetime.now().strftime('%H:%M:%S'))
clf.fit(train_X,train_Y)
#clf.fit(X,y)
print('end train at '+ datetime.now().strftime('%H:%M:%S'))
#y_result = clf.predict_proba(test_X)
y_result = clf.predict(test_X)
print(y_result)
print(test_Y)

'''
dot_data = tree.export_graphviz(clf, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("first_model.pdf")'''

#scores = cross_val_score(clf, X, y)
joblib.dump(clf,'regression_forest.model')
#print(scores.mean())
print('end at '+ datetime.now().strftime('%H:%M:%S'))
