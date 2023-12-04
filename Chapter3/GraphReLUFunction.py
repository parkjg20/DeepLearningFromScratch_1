# ReLU(Rectified Linear Unit, 렐루) 함수
# 시그모이드 함수의 대체재로 최근 많이 사용
# 입력이 0 이하면 0을 출력, 0을 넘으면 그대로 출력

# # 함수 보기 전에 혼자 구현
# def relu_function(x):
#     return np.array(x if (x >= 0) else 0, dtype=int)
# 이건 책에 나온 예제.. max라는 좋은 방법이 있었다.
def relu_function(x):
    return np.maximum(0, x)

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
y = relu_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 5)
plt.show()
