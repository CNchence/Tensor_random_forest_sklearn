import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

dir = "F:\\D2CO_dataset\\detect_train_data\\1539228326_pos_neg"

res = np.loadtxt(dir + "\\data.txt", delimiter=" ")
res = res[1:len(res)]
res = np.split(res, 2, axis=1)[1]
res = np.split(res, 3, axis=1)
print(res)
data = np.loadtxt(dir + "\\position.txt", delimiter=" ")
data = data[1:len(data)]
dic = {0: "tx", 1: "ty", 2: "tz"}
for i in range(0, len(res)):
    name_list =np.array([["%s" % dic[i], "minx", "miny", "maxx", "maxy"]])
    ranger_data = np.concatenate((res[i], data), axis=1)
    ranger_data = np.concatenate((name_list, ranger_data), axis=0)
    print("ranger_data:", ranger_data)
    np.savetxt(dir + '\\ranger_train_%s.dat' % dic[i], ranger_data, fmt='%s',)