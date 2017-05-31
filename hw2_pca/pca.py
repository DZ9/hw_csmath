import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt('data/optdigits.tra', delimiter=",", dtype='float')
threes = t[np.where(t[:, -1] == 3)]
threes = np.transpose(threes)
m = np.mean(threes, axis=0)

X = threes - m
print X.shape

U, S, V = np.linalg.svd(X)
print U.shape

Up = U[:, 0:2]
print Up.shape

Upt = np.transpose(Up)

re = np.dot(Up, Upt)
print re.shape

x_ = m + np.dot(re, X)

w = np.dot(Upt, X)
print w

plt.plot(w[1, :], w[0, :], color="blue", linestyle='None', marker='.')
plt.show()




