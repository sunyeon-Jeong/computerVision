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

# 학습/테스트 데이터 분리
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True) # X = 데이터, y = 레이블
# 25% -> 테스트, 75% -> 훈련용
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# 퍼셉트론 모델 객체 생성
# - tol(tolerance) : 허용오차. 모델가중치(w)가 tol 값보다 작을 경우 학습 종료 (더이상 개선 가능성X)
clf = Perceptron(tol=1e-3, random_state=0) # 1e-3 : 0.001

# 퍼셉트론 모델 학습
clf.fit(X_train, y_train)

# 테스트 데이터 결과 예측
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

target_names = ['digit 0', 'digit 1', 'digit 2', 'digit 3', 'digit 4',
                'digit 5', 'digit 6', 'digit 7', 'digit 8', 'digit 9']

# 참값(y_test)과 예측값(y_pred)을 대상으로 classfication_report 생성
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import ConfusionMatrixDisplay

# 참값(y_test)과 예측값(y_pred)을 대상으로 confusion matrix 생성
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

rows = 2; cols = 5
fig = plt.figure()

fig.tight_layout(w_pad=100)

for i in range(10):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(X_test[i].reshape(8, 8))
    ax.set_title("y_hat = " + str(y_pred[i]))

plt.show()