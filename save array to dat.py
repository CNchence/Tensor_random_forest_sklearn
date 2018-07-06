import numpy as np
f = open('features.dat','w')
f.close()
f = open('features.dat','a')
X = np.load("feature_data_array.npy")
f.write('f0')
for i in range(1,len(X[0])):
    f.write(',f'+str(i))
f.write(',res\n')

f.write('111')


f.close()

#with open('features.dat',"w") as f:
#    f.write("\n".join(" ".join(map(str, x)) for x in X))