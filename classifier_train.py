import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import tree
def classifier_train(X, neg_X, train_dir):
    print('begin at ' + datetime.now().strftime('%H:%M:%S'))
    y = np.array([1]*len(X) + [0]*len(neg_X))
    X = np.concatenate((X, neg_X), axis=0)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, oob_score=True,
        min_samples_split=2, random_state=10, verbose=10)

    #t_size=0.09#测试集所占比例，分数
    #train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=t_size, random_state=10)

    print('begin to train at ' + datetime.now().strftime('%H:%M:%S'))
    clf.fit(X, y)
    print('end train at ' + datetime.now().strftime('%H:%M:%S'))

    joblib.dump(clf, train_dir + 'classifier_forest.model')

    print('end at ' + datetime.now().strftime('%H:%M:%S'))
