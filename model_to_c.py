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

def get_feature_index(clf):
    feature_importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(feature_importances)[::-1]
    f = 0
    feature_index = []
    while feature_importances[indices[f]] != 0:
        feature_index.append(indices[f])
        f += 1
    return ','.join(str(e) for e in feature_index)

def change_model_to_c(dir, name):
    clf = joblib.load(dir + name)
    porter = Porter(clf, language='c')
    output = porter.export(embed_data=True)
    print(output)
    cpp_file_name = dir + name.split('.')[0]+".cpp"
    f = open(cpp_file_name, 'w')
    f.truncate()
    f.write(output)                #写入模型转换后的原始c文件
    f.close()

    fp = open(cpp_file_name, "r")
    content = fp.read()
    fp.close()
    feature_add = []
    feature_add.append("#include <classifier_forest.h>")
    feature_add.append("void gene_used_features_index(vector<int>& needed_feature){")
    str_feature_index = get_feature_index(clf)
    feature_add.append("    needed_feature = vector<int>{" + str_feature_index + "};")
    feature_add.append("}\n")
    pos = content.find("int predict_0(float features[]) {")
    if pos != -1:
        content = content[:pos] + "\n".join(e for e in feature_add) + content[pos:]  #增加特征序号向量
        file = open("a.txt", "w")
        file.write(content)
        file.close()

    pos = content.find("int predict (float features[]) {")
    if pos != -1:
        file_add = open("predict_function.txt", "r")
        content_add = file_add.read()
        content = content[:pos] + content_add         #修改predict函数
        file_add.close()
    file = open(cpp_file_name, "w")
    file.write(content)
    file.close()





    #generate_c_test_data()
change_model_to_c("F:\\D2CO_dataset\\detect_train_data\\1538123396_pos_neg\\", 'classifier_forest.model')
