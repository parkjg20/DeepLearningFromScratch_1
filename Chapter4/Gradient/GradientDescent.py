# 경사하강법
# lr = 학습률(learning rate) - 너무 크지도 작지도 않은 값으로 적절하게 설정
from NumericalGradient import numerical_gradient

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad # 학습률 만큼 변화시키면서 기울기 검사

    return x