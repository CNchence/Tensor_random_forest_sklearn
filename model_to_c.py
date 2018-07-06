from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn_porter import Porter
from sklearn import tree

def change_model_to_c(name):
    clf = joblib.load(name)
    porter = Porter(clf, language='c')
    output = porter.export(embed_data=True)
    print(output)
    f = open(name.split('.')[0]+".txt", 'a')
    f.truncate()
    f.write(output)
    f.close()
def generate_c_test_data():
    #X = np.load("array//pos_feature_data_array.npy")
    #print('finish load feature_data_array at ' + datetime.now().strftime('%H:%M:%S'))
    neg_X = np.load("array//neg_feature_data_array.npy")
    print('finish load neg_feature_data_array at ' + datetime.now().strftime('%H:%M:%S'))
    tmp_neg_x = neg_X[0]
    np.savetxt("c_test_neg_data.txt",tmp_neg_x, '%10.8f','  ')
    print(neg_X)

#generate_c_test_data()
change_model_to_c("6_25_five_pose.model")
