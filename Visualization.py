import pydotplus
from sklearn.externals import joblib
from sklearn import tree
import os

clf = joblib.load('first_tree.model')
print(len(clf.estimators_))
for i in range(0,200):
    dot = tree.export_graphviz(clf.estimators_[0], out_file='tree_dot//'+str(i)+'.dot')
    os.system('dot -Tpdf tree_dot//'+str(i)+'.dot -o tree_pdf//'+ str(i)+'.pdf')
