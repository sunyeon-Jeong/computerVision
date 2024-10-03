'''
[week2 퍼셉트론(scikit-learn perceptron)]
- 라이브러리 import
- 데이터 불러오기
- sklearn 라이브러리의 퍼셉트론 모델 불러오기
- 모델 학습
- 테스트데이터를 이용한 모델 성능평가 및 시각화
'''

''' 1. 사용데이터
- scikit-learn에서 제공하는 필기체 숫자 데이터
- 0~9까지 총 10개의 숫자 필기체 이미지를 수집한 데이터
- 개별 이미지는 8*8 = 64개의 특성(feature vector)
- 개별 특성은 픽셀을 의미하며 0-16까지의 정수값이 부여됨
- 퍼셉트론은 학습을 통해 숫자 필기체 이미지를 인식 -> 주어진 이미지의 소속 클래스를 예측하는 분류작업
'''

''' 2. 모델의 학습과 예측
- train_test_split -> 훈련용/학습용 데이터 확보
- fit() -> 모델 학습
- predict() -> 예측 작업 수행
'''

''' 3. 분류 성능평가와 시각화
- classification_report
  : sklearn.matrics 패키지에서 제공하는 기능
  : precision, recall, f1-score 등 다양한 측도를 이용하여 클래스 별로 분류 성능 확인 가능
- Confusion matrix
  : 학습모델의 분류성능을 시각적으로 확인
  : 목표클래스 성공/실패 수를 알려줌 -> 모델의 분류 성능을 한눈에 파악 가능
'''

# 라이브러리 import
from sklearn.datasets import load_digits # 학습 데이터
from sklearn.linear_model import Perceptron # 퍼셉트론 모델
from matplotlib import pyplot as plt # 데이터 시각화 라이브러리

# 학습데이터 로딩 -> X(특징벡터), y(클래스 레이블)로 구분
X, y = load_digits(return_X_y=True)

rows = 2; cols = 5 # 플롯 행/열 선언
fig = plt.figure() # 새로운 플롯 객체 생성

for i in range(10): # 필기체 0~9
    ax = fig.add_subplot(rows, cols, i+1) # 2행 5열 플롯에서 인덱스에 해당하는 새로운 서브플롯 생성
    ax.imshow(X[i].reshape(8, 8)) # 64차원 벡터 -> 8*8 이미지 형태로 변형
    ax.set_title(str(i)) # 서브플롯 타이틀을 인덱스 값으로 
plt.show()