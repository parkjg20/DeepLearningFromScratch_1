import numpy as np

## 3.3.3 신경망 구축해보기
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

Y = np.dot(X, W)
print(Y)