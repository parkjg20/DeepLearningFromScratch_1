import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt
from samples.dataset.mnist import load_mnist

TwoLayeredNeuralNetwork = __import__('4_5_1_TwoLayeredNeuralNetwork').TwoLayeredNeuralNetwork

# train = 학습 입력 데이터
# test = 시험 입력 데이터
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 하이퍼 파라미터 (유저가 설정해야하는 파라미터)
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayeredNeuralNetwork(input_size = 784, hidden_size = 50, output_size = 10)
for i in range(iters_num):

    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # numerical_gradient < gradient 성능 개선 버전

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc: " + str(train_acc) + " | test acc: " + str(test_acc))

# 책에서 표시하고있는 그래프를 똑같이 그려보자.

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(np.arange(1, len(train_acc_list)), train_acc_list, label = "Train")
plt.plot(np.arange(1, len(train_acc_list)), test_acc_list, label = "Test")
plt.show()
