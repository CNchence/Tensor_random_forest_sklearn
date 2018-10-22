import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

dir = "F:\\D2CO_dataset\\detect_train_data\\1538123396_pos_neg"
res = np.loadtxt(dir + "\\data.txt", delimiter=" ")
res = res[1:len(res)]
res = np.split(res, 2, axis=1)[1]
print(res)
data = np.loadtxt(dir + "\\position.txt", delimiter=" ")
data = data[1:len(data)]

clf = RandomForestRegressor(n_estimators = 500, max_depth = 10, oob_score = True,random_state = 10, verbose = 10)

t_size=0.1#测试集所占比例，分数
train_X, test_X, train_Y, test_Y = train_test_split(data, res, test_size=t_size, random_state=10)

clf.fit(train_X,train_Y)

y_result = clf.predict(test_X)
print(y_result)
print(test_Y)