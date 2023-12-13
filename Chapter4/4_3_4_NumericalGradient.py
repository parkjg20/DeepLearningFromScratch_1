# 수치 미분을 이용한 기울기 계싼
from Gradient.NumericalGradient import numerical_gradient
import numpy as np

def function(x):
    return x[0] ** 2.0 + x[1] ** 2.0

print(numerical_gradient(function, np.array([3.0, 4.0])))
print(numerical_gradient(function, np.array([0.0, 2.0])))
print(numerical_gradient(function, np.array([3.0, 0.0])))