import numpy as np 

pre1 = np.loadtxt("error_1.txt")
pre2 = np.loadtxt("error_2.txt")
pre3 = np.loadtxt("error_3.txt")
pre4 = np.loadtxt("error_4.txt")
pre5 = np.loadtxt("error_5.txt")

pre = (pre1+pre2+pre3+pre4+pre5)/5

error = np.abs(pre)

mae = np.mean(error,axis=0)

acc_tmp = error
acc_tmp[acc_tmp<5] = 1
acc_tmp[acc_tmp>=5] = 0
acc = np.mean(acc_tmp,axis=0)

print(mae)
print(acc)
