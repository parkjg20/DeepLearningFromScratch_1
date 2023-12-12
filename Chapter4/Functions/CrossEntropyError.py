import numpy as np


# y = 출력 데이터
# t = 정답 데이터
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # batch_size(N)
    batch_size = y.shape[0]

    # 원 핫 인코딩의 경우 아래 코드
    # - ∑_(N) t_n log y_n / N
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

    # 숫자 레이블로 주어질 경우의 구현
    # return -np.sum(nplog(y[np.arange(batch_size), t] + 1e-7)) / batch_size
