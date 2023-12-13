# 편미분 구현 실습
# function_2 = y = (x_0)^2 + (x_1)^2)
# x_0 = 3, x_1 = 4일 때, x_1에 대한 편미분

import numpy as np
import matplotlib.pylab as plt

def function_2(x1):
    return 3.0 ** 2.0 + x1 * x1

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

print(numerical_diff(function_2, 4.0))

