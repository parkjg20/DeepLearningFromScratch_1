# 편미분 구현 실습
# function_2 = y = (x_0)^2 + (x_1)^2)
# x_0 = 3, x_1 = 4일 때, x_0에 대한 편미분

def function_1(x0):
    return x0 * x0 + 4.0 ** 2.0

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

print(numerical_diff(function_1, 3.0))

