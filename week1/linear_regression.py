'''
[Week1 예측(선형회귀)]
- csv 데이터 불러오기
- 데이터 -> 훈련데이터와 테스트데이터로 구분
- sklearn 라이브러리 -> (경사하강법) 모델 학습
- 테스트데이터 -> 학습모델 성능평가 (Mean Squared Error)
'''

''' <라이브러리 소개>
* pandas
  : 행과 열로 구성된 2차원 데이터 처리
  : data frame 객체에 행/열 데이터 저장
  - 데이터 읽기 : read_csv
  - 컬럼명 확인 : print(dataset.describe())
  - 컬럼명으로 데이터 접근 : print(dataset['column name'])
  - 인덱싱 방법
    : iloc[]] -> 행/열 숫자값을 지정하여 인덱싱
    : loc[] -> 행은 인덱스, 열은 열이름
  
* matplotlib
  : 그래프 등을 통해 데이터를 가시화하는 라이브러리
  
* sklearn
  : 각종 기계학습 모델 구현 라이브러리
'''

''' 2. 훈련데이터와 학습데이터
- 선형회귀 : 학습데이터 -> 모델의 인자(w값)를 탐색
- 학습모델의 성능검증 : 모델 학습과정에서 사용하지 않은 데이터
  -> 모델의 예측값을 산출한 후, 참값과 비교
- `train_test_split` : 학습데이터를 훈련데이터와 테스트데이터로 분할
- 훈련데이터와 테스트 데이터에 중복되는 개체가 존재하지 않도록 주의
'''

''' 3. fit과 predict
- fit() : 모델 학습
- predict() : 학습된 모델을 이용하여 회귀 값을 예측
'''

''' 단순 선형 회귀(Simple Linear Regression) '''
# 라이브러리 불러오기 (importing the libraries)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# 데이터 불러오기 (importing the datas)
dataset = pd.read_csv('/Users/sunyeonjeong/dev/github/computerVision/week1/Salary_Data.csv')
X = dataset.iloc[:, :-1].values # iloc 함수 -> csv 데이터를 X에 저장
y = dataset.iloc[:, 1].values # values -> 선택된 데이터를 Numpy 배열형태로 변환

# 훈련데이터 / 테스트데이터 셋 구분 (Splitting the dataset into the Training set and Test set)
from sklearn.model_selection import train_test_split
# - X(특징벡터), y(정답값)
# - test_size : 테스트/학습 데이터 양의 비율 설정 (1/3 -> 1/3은 테스트로, 2/3는 학습으로)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 선형 회귀 모델 불러오기
from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # 객체생성

# 모델 학습하기
# - fit() -> 가중치 초기화/손실함수 계산/가중치 업데이트 포함
# - 학습데이터만 사용
regressor.fit(X_train, y_train)

print(regressor.coef_) # 단순 선형 회귀의 w_1 개념 (기울기)
print(regressor.intercept_) # 단순 선형 회귀의 w_0 개념 (절편)

# 학습된 모델 대상 테스트 데이터로 성능평가
# - 테스트데이터만 사용
# - shpae : 배열의 형태를 반환 (배열의 포인트)
y_pred = regressor.predict(X_test)

print(y_test.shape)
print(y_pred.shape)

# R2 score 평가지표를 통해 학습 성과 확인
# - 모델이 종속변수의 변동성을 얼마나 잘 표현하는지 나타냄
# - R2 점수는 0~1로, 1에 가까울수록 모델이 데이터에 잘 맞다는 의미
from sklearn.metrics import r2_score

result = r2_score(y_test, y_pred)

print(result)