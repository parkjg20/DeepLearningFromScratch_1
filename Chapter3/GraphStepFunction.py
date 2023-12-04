# 계단함수(Step Function), 인자로 np array(numpy array)를 지원하도록 구현
# 구간이 명확하게 구분되는 함수.

def step_function(x):
    y = x > 0
    return y.astype(np.int)

# 위처럼 함수를 선언하는게 가능한 이유는? 브로드캐스트 테스트(numpy)
import numpy as np

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0 # 브로드캐스트 발생
print(y)

y = y.astype(np.int)  # 브로드캐스트 발생
print(y)

import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위
plt.show()
