# 수치 미분 limit (n -> 0) 이 아닐 때의 미분
# 특정 점과 미세한 차이가 나는 다른 점 사이의 기울기
import numpy as np


def numerical_gradient(f, x):
    h = 1e-4  # 시간(0.0001) 완전 0 일 수 없기 때문에 float32내에서 표현할 수 있는 작은 값을 사용함.
    gradient = np.zeros_like(x)  # x와 형상이 같은 배열 생성

    for index in range(x.size):
        tmp_val = x[index]
        # f(x + h) 계산
        x[index] = tmp_val + h
        fxh1 = f(x)

        # f(x - h) 계산
        x[index] = tmp_val - h
        fxh2 = f(x)

        gradient[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tmp_val  # 값 복원

    return gradient
