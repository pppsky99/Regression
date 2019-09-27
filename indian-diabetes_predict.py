# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:36:06 2018

@author: is2js
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#랜덤 시드 고정
np.random.seed(7)

# 넘파이의 loadtxt로 csv파일을 불러올 수 있다. 
# 대신 구분자(delimiter)를 ","로 지정
dataset = np.loadtxt('./data/pima-indians-diabetes.csv', delimiter=",")

X = dataset[:, 0:8] # 첫번째부터 8번째 칼럼까지 - 0번 ~ 7번 칼럼
y = dataset[:, 8]   # 9번째 칼럼만 target


model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#verbose=2 는 막대가 없다.
model.fit(X, y, epochs=50, batch_size=10, verbose=2)

# 학습모델 평가 = evaluate => 손실율, loss가 어느정도 되는지 알려준다.
scores = model.evaluate(X, y)
print('\n %s : %.2f%%' %(model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# 예측값은 0 or 1이 안나오고 확률로 나오기 때문에,
# for문을 돌려서, 각 예측값을 반올림 해서  0or1을 만든다.
rounded = [round(x[0]) for x in predictions]

print(rounded)


## 0과 0.05 사이에 랜덤한 숫자를 정규분포화 시켜서 weight를 초기화한다.
## initializer = 'uniform'
model = Sequential()
model.add(Dense(12, input_dim = 8,init = 'uniform', activation='relu'))
model.add(Dense(8,init = 'uniform', activation='relu'))
model.add(Dense(1, activation='sigmoid',init = 'uniform'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10, verbose=2)
scores = model.evaluate(X, y)
print('\n initializer = \'uniform\'사용시 %s : %.2f%%' %(model.metrics_names[1], scores[1]*100))