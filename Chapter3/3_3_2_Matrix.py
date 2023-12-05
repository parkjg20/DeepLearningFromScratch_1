import numpy as np

# 3.3 ~ 3.3.2 행렬 다루기
oneDimension = np.array([1, 2, 3, 4, 5, 6])
twoDimension = np.array([[1, 2], [3, 4], [5, 6]])
threeDimension = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

# 행렬 매트릭 정보 출력 - 일관성 있는 출력을 보장하고자 1차원도 튜플의 형태로 내보낸다. ex) (6, )
print(oneDimension.shape)
print(twoDimension.shape)
print(threeDimension.shape)

otherOneDimension = np.array([1, 2, 3, 4, 5, 6])
otherTwoDimension = np.array([[1, 3, 5], [2, 4, 6]])

# 행렬 곱
# 1차원 행렬의 곱은 스칼라 값(91)을 반환한다.
print(np.dot(oneDimension, otherOneDimension))
# 2차원 행렬의 곱은 차수에 해당하는 새로운 행렬이 만들어진다.
print(np.dot(twoDimension, otherTwoDimension))

