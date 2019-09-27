# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:22:42 2018

@author: is2js

"""

import numpy as np
import pandas as pd

from keras.models import Sequential #선형회귀분석용 가설 모델
from keras.layers import Dense
from sklearn.cross_validation import train_test_split

import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.metrics import confusion_matrix


# PlotLosses는 매 학습마다 plt를 띄워주는 클래스를 만들려는 것인데,
# 케라스의 콜백함수를 ***상속***해서 학습시 콜백인자로 들어갈 수 있다. 케라스의 콜백함수를 상속하여 추가기능을 가진 콜백함수를 만드는 것이다.
# (클래스는 필드와, 함수를 내부에 넣어서, 같은 필드와 함수를 가지는 개별 인스턴스를 여러개 만들어서 쓸 수 있는 틀이다)
# 메서드의 첫번째 인자에 self에는 인스턴스(객체)들이 전달된다. a1, a2, ... 여러 인스턴스가 클래스의 필드와 메서드를 들어와서 사용한다고 생각하자.
# 케라스 콜백함수를 클래스의 인자로 넣어 만들면 딥러닝 모델 학습시 model.fit( callbacks =[]) 의 콜백인자로 클래스의 인스턴스를 넣을 수 있다.
# 들어가는 순간, 학습과정에서 클래스의 내부 함수(메서드) 2개의 함수가 실행된다.
class PlotLosses(keras.callbacks.Callback): 
    # train set init
    # 함수1. 훈련이 시작될 때 초기화 작업을 하는 함수
    def on_train_begin(self, logs={}): # 함수의 인자로는 self(각 객체들)와 logs={}라는 빈 딕셔너리가 들어간다.
        
        # 객체.객체안에 변수 = 멤버 변수 = 인스턴스 변수
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig=plt.figure()
        
        self.logs= []
        
    # 함수2. 학습 한번 끝날때마다 호출되는 함수들
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs) #들어오는 로그메세지를 logs{}딕셔너리에 넣어줌
        self.x.append(self.i) #카운팅 되는 값들 따로 받아줌
        self.losses.append(logs.get('loss')) #logs딕셔너리에서 loss라는 키값을 가지는 value 꺼내서 losses에 어펜드
        self.val_losses.append(logs.get('val_losses')) #검증의 손실값
        self.i += 1
        
        clear_output(wait=True) # 학습이 끝나면 콘솔지우고 그래프 그리기
        plt.plot(self.x, self.losses, label='loss')
        plt.legend()
        plt.show()
        

# 케라스 모델 만들기 -> return
def getModel(arr): #함수호출과함께 들어오는 데이터->데이터 크기로 활용할 것임
    model = Sequential() # 선형분류 모델 생성
    for i in range(len(arr)): # 데이터 크기만큼 루프를 돌면서 layer를 쌓는다.
        if i != 0 and i != len(arr)-1:
            if i==1: # i가 1인, 첫번째 계층만 input_dim을 입력해준다. 로스와 activation은 normal과 relu를 사용
                model.add(Dense(arr[i], input_dim=arr[0], kernel_initializer='normal', activation='relu'))
            else: # 중간의 계층들은 은닉층으로서 initializer가 필요없고 activation만 relu로 지정
                model.add(Dense(arr[i],activation='relu'))
    #마지막층 출력층만 따로 쌓는다.
    model.add(Dense(arr[-1], kernel_initializer='normal', activation='sigmoid'))
    # 모델 학습과정 설정하기
    # loss함수와 옵티마이저를 설정해준다.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
    return model

# male, female의 정보를 숫자로 바꾸는 매핑작업
# 매핑 딕셔너리의 생성을 for문을 이용한다.
def mapping(data, feature): #데이터셋과 특징칼럼의 특정값에 해당하는 것을 넘겨주면,
    featureMap = dict() # 딕셔너리를 하나 생성하고
    count = 0 
    
    # sorted( data, 정렬기준, reverse여부)이므로 
    # data 중 feature칼럼만 unique()하게 배열해놓은 상태에서 역순(z->a)으로 정렬한 뒤,
    # 역순으로 정렬된 unique한  feature칼럼의 데이터를 하나씩 받아와 하나씩 뿌려준다.
    for i in sorted(data[feature].unique(), reverse=True): 
        # 받은 unique데이터를 key값으로 넣는데, value값은 역순으로 0부터 주어진다.
        # 새로운 unique데이터가 들어온다면, 1, 2, 3 순으로 채워질 것이다.
        featureMap[i] = count
        count = count+1
    # 만들어진 딕셔너리 featureMap을 이용해서 data중 feature칼럼을 mapping하여
    # 문자열 --> mapping된 숫자로 바꾼다.
    data[feature] = data[feature].map(featureMap)
    return data

# 데이터를 로드할 때는, pandas를 이용해 dataframe으로 가져온다.
# 여기서 필요없는 칼럼을 삭제하고, head()와 shape를 출력해준다.
def dataLoad():
    dataset = pd.read_csv('./data/breast-cancer-wisconsin-data.csv')    
    print(dataset.head())
    
    #필요없는 칼럼을 삭제한다.
    dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)
    print(dataset.head())
    print(dataset.shape)
    return dataset
    


dataset = dataLoad()
# 칼럼 중 sum한 것이 null인 것 찾아보기
pd.isnull(dataset).sum()

# 매핑함수를 이용해서 feature칼럼 diagnosis를 이용해서 null값이 없다고 검증된 것을
# 직접 세어보고, 매핑된 데이터를 받는다.
dataset=mapping(dataset, feature='diagnosis')
# 매핑된 데이터의 샘플 5개만 받아보자.
sample_list = dataset.sample(5)
print(sample_list)


# diagnosis를 예측하는 것으로 문제를 만들자.
# 학습에서 예측에 사용되는 데이터들인 X는 diagnosis칼럼을 지운 뒤 X로 받고
X = dataset.drop(['diagnosis'], axis =1)
# dataset에서 정답인 diagnosis칼럼을 y로 받자.
y = dataset['diagnosis']


# cross_validation을 만들기 위해서, X와 y를 train_test_split 함수를 통해 짤라보자.
# 짜를 때, test데이터의 비중은 0.2로 줄 것이다.  
# 짜를 때, 랜덤으로 짜르게 되지만, seed를 42로 임의로 주어서 고정되게 한다.
# 1. 먼저 전체를 train X, y와    test X, y로 나눈다. test는 20%만 가지게 한다.
# 2. train X, y를 다시 한번 train X, y 와  valid X, y로 나눈다. valid는 20%만 가지게 한다.
trainX, testX, trainY, testY = train_test_split(X, y, test_size= 0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


# 모델을 불러오는데 getModel의 인자로는 arr(리스트)가 들어갔다. 
# 이 arr는 len()만큼 for문이 돌면서 layer를 쌓는데
# arr[0]이 첫번째 layer의 input_dim (input node인듯?)이었고, 
# arr[i]가 add되는 각 층의 output수(node수) 였고
# arr[-1]이 마지막층의 output수였다.
firstModel = getModel([30, 50, 1]) # 3개의 층 

#앞에서 정의한 PlotLosses클래스를 생성자로 사용하여 변수로 넘겨받는다.
plot_losses  = PlotLosses()
# 학습시킬 때는, np.array형태로 건네준다.
# 나머지 칼럼들을 가지고  diagnosis인 남자0, 여자1을 예측하도록 학습시킨다.
firstModel.fit(np.array(trainX), np.array(trainY), epochs=40, callbacks = [plot_losses])

# 학습시킨 firstModel을 validset으로 평가해준다.
# 모델이 valX의 정보를 가지고 valY에 있는답을 예측한 것과 결과가 어떻게 되는지 scores변수로 받아준다.
scores = firstModel.evaluate(np.array(valX), np.array(valY))
print('Loss : ', scores[0])
print('Accuracy : ', scores[1]*100)


# 2번째 모델은 3개층이지만, 가운데 층이 100개의 node를 가진다.
#secondModel = getModel([30, 100, 1]) # 3개의 층 
#secondModel.fit(np.array(trainX), np.array(trainY), epochs=40, callbacks = [plot_losses])
#scores2 = secondModel.evaluate(np.array(valX), np.array(valY))
#print('Loss : ', scores2[0])
#print('Accuracy : ', scores2[1]*100)

# 3번째 모델은 5개 층을 가지도록 해보자.
#thirdModel = getModel([30, 100, 1]) # 3개의 층 
#thirdModel.fit(np.array(trainX), np.array(trainY), epochs=40, callbacks = [plot_losses])
#scores3 = thirdModel.evaluate(np.array(valX), np.array(valY))
#print('Loss : ', scores3[0])
#print('Accuracy : ', scores3[1]*100)



# 결과적으로 firstModel가 검증시 정확도가 좋았다.
# 이제 firstmodel을 가지고 testX를 이용해 testY를 가지고 예측해보자.
# 검증(evaluate)는 loss와 accuracy를 scores변수에 return해줬지만,
# 예측(predict)는 예측한 Y값을 도출한다.
predY = firstModel.predict(np.array(testX))
# 예측한값은 np.round()로 반올림한 뒤, astype(int)로 형변환해주자
# reshpae을 통해 차원을 바꿔준다. 1행이면서, 나머지 데이터는 2차원으로 무한(-1)개로 던져준다
# 어차피 1차원에만 있는 첫번째 데이터 [0]만 가져온다.
predY =np.round(predY).astype(int).reshape(1, -1)[0]


# 예측값(predY)과 실제값(testY)를 이용해 컨퓨젼 매트릭스를 만든다.
m = confusion_matrix(predY, testY)
print('Confusion Matrix')
print(m)
# 컨퓨전 매트릭스의 개별값들을 받을 때는, .ravel()을 붙혀서 받는다.
tn, fn, fp, tp = confusion_matrix(predY, testY).ravel()

# 판다스의 crosstab을 이용해서 예측값과 실제값을 비교해보자.
ct = pd.crosstab(predY, testY)
print('Cross tab')
print(ct)

#민감도, 특이도, precision을 각각 구해보자
sens = tp / (tp + fn)  # = 진짜 행 전체 분에 진짜배기 / tp : 올바른것을 진짜 수락한 경우 + fn 거절해야할 것을 진짜도 거절못한 경우 = 진짜 행 전체
spec = tn / (tn + fp)  # tn : 거절해야할 것을 진짜 거절한 경우 + fp 올바른 것을 진짜는 거절해버린 경우 = 아닌 컬럼 전체사각형 + 진짜컬럼 중 아닌 놈들
prec = tp / (tp + fp)  # = 진짜 열 전체 분에 진짜배기 / 
print('Sensitivity(민감도) : ', sens )
print('Specificity(특이도) : ', spec )
print('Precision(정밀도) : ', prec )