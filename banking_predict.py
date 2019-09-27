# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 02:49:38 2018

@author: is2js
"""

import pandas as pd

# (10000, 14)
dataset = pd.read_csv('./data/Churn_Modelling.csv')

# index Location = 칼럼의 index 를 가지고 인덱싱한다.

# 4번째 칼럼부터 13번째 칼럼까지 --> [3:13]으로 예측근거 칼럼이라고 가정하자.
# 이 때, 문제가 있는데, geography(국가)과 gender(성별) 칼럼은 문자로 되어있어 컴퓨터가 인식하지 못한다. ( 이전에는 mapping )
# 여기서는 groupby를 통한 그룹별통계를 얻기 위해서는, 숫자로 되어있어야 하기 때문에, labelEncoder나 OneHotEncoder클래스가 필요하게 된다.
# 문자열 데이터 칼럼을 포함하여 칼럼인덱싱하면, object를 가지므로, spyder의 Variable explorer에서도 일단 안보인다.
# 13번 칼럼만 빼고 다 x에 
X = dataset.iloc[:, 3:13].values 
# 마지막 13번 칼럼(Exited칼럼)만 타겟으로서 y에 담는다.
y = dataset.iloc[:, 13].values


# 위의 문제점 해결을 위해서
# LabelEncoder : 인스턴스를 이용해 국가명이라는 label(분류, 정답)을 --> 코드 0, 1, 2로 encoding(암호화)를 자동으로 맞추어서 변형해주는 fit_transfrom함수를 사용한다.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # 라벨인코더 클래스의 인스턴스 생성
# X의 2번째 칼럼(읽을 때 +1)을 라벨인코더를 이용해서 0, 1, 2로 변형시킨다
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
# 3번째 칼럼을 라벨인코딩(0, 1, 2..) 할 때도 새로운 인스턴스가 필요한가보다.
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])

# 하지만 0, 1, 2라는 것은 크다/작다의 개념으로서 잘못 연산될 수 있기 때문에
# OneHotEncoder를 통해서 labelencoder(문자열->숫자)로 변형된 것을 단 하나의 요소만 1로 만들어준다.
# 먼저, 인스턴스 생성. 라벨인코더와 다르게, 여기서는 인자를 하나 넣어준다.
onehotencoder = OneHotEncoder(categorical_features=[1]) #입력이 벡터인 경우에 특정한 열만 카테고리 값이면 categorical_features 인수를 사용하여 인코딩이 되지 않도록 지정
# 원핫인코딩 된 데이터는 .toarray()를 통해 array형태로 만들어줘야 딥러닝 모델에 넣을 수 있다. 안그러면 행렬형태?
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #2번째 칼럼부터 쓰도록 한다.(?)


# 이전에는 from sklearn.cross_validation import train_test_split를 통해서 train/test -> train/valid 를 나누었지만
#trainX, testX, trainY, testY = train_test_split(X, y, test_size= 0.2, random_state=42)
#trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
# 여기서는 cross validation을 사용하지 않으면서 & random_state라는 seed도 주지않고 0.2만큼 잘라낸다.
# sklearn.model_selection 에서 train_test_split 클래스를 사용한다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# kerasyoutube 0-1. Preprocessing Data에서 
# 참고1. (2100,) shape의 데이터는 MinMaxScaler인스턴스로 fit_transform 함수가 먹히지 않는다. reshape(-1, 1)을 통해 (2000,1)로 2차원 형태여야한다.
# 참고2. MinMaxScaler(feature_range=(0,1))을 통해  0과 1사이의 범위로 scaling한 적이 있다.
# 참고3. list가 가지지 못한 .shape는 array가 가지고 있다. 케라스는 input_shape로서 .shape를 가지는 array를 원한다.

# StandardScaler는 수치적으로는 평균은 0, 분산(표준편차) 1을 만들어주는  Z = X-m/a를 적용하여
# 값의 범위가 어떤 값을 가지든지 간에 표준 정규분포 안의 곡선으로 들어가도록 잡아준 것이다.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # 스케일러의 인스턴스 생성
X_train = sc.fit_transform(X_train) #X_train와 X_test 데이터에 Scaling 적용
X_test = sc.fit_transform(X_test) # 강의에선 그냥 transform만 사용..


#딥러닝 시작
from keras.models import Sequential #선형 회귀 분류
from keras.layers import Dense # FC layer

classifier = Sequential()
# 첫번째 add에서는 
# Input Layer(input_dim) + first hidden layer(output_dim)을 같이 입력하자!
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu',
                     input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

#compile에서는 optimizer와 loss만 따진다.
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# 학습시키기
# verbose = 0 을 안해줘도 메세지들이 엄청 나와서 생략했다. (verbose=1이 default라서 그게 나왔다.)
classifier.fit(X_train, y_train, batch_size= 10, nb_epoch = 100) # 배치 10개씩, epochsss = number of epoch =  100

# 학습된 것으로 test데이터를 예측하기
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # 0.5보다 큰것만 True라서 1   /   0.5보다 작거나 같으면 False라서 0이 대입된다!

# 실제값과 예측값을 가지고 cm만들기
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
#accuracy 대각선 / 전체
Accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0])
print(Accuracy)


