'''
[week3 다층퍼셉트론(multi layer perceptron)]
- 퍼셉트론의 한계인 XOR문제의 해결방법
- 논리회로를 통한 구현(numpy)
'''

'''
0. XOR 문제의 해결방법 (MLP XOR Solution)
- XOR 문제 또한 0 또는 1 이진 분류 문제에 해당하며, 단층 퍼셉트론으로는 해결할 수 없는 케이스임.
- 0 : 두 입력 값이 서로 같을 경우
- 1 : 두 입력 값이 서로 다를 경우
'''

'''
1. XOR 게이트를 3개의 퍼셉트론을 이용해 표현하기
   : input(2개의 feature vector) -> NAND(AND게이트의 반대), OR 실행 후 NAND와 OR의 AND값 -> Y (output)
- NAND, OR, AND 게이트를 나타내기 위해 각 단층 퍼셉트론은 3개의 입력노드(바이어스 포함)와 1개의 출력노드로 구성
- 출력노드는 3개의 입력노드와 각 대응되는 가중치의 곱의 합인 가중치의 곱의 합인 가중합과 활성화함수를 거쳐 출력함
'''
import numpy as numpy

# NAND 게이트 - 퍼셉트론 가중치
w_nand = numpy.array([-2, -2])
# OR 게이트 - 퍼셉트론 가중치
w_or = numpy.array([2, 2])
# AND 게이트 - 퍼셉트론 가중치
w_and = numpy.array([1, 1])

# NAND 바이어스
bias_nand = 3
# OR 바이어스
bias_or = -1
# AND 바이어스
bias_and = -1

# 퍼셉트론 순전파
def Perceptron(x, w, bias):
    
    z = numpy.sum(w * x) + bias
    y_hat = z
    
    if y_hat <= 0:
        return 0
    else:
        return 1
    
# NAND 게이트
def NAND(x1, x2):
    return Perceptron(numpy.array([x1, x2]), w_nand, bias_nand)

# OR 게이트
def OR(x1, x2):
    return Perceptron(numpy.array([x1, x2]), w_or, bias_or)

# AND 게이트
def AND(x1, x2):
    return Perceptron(numpy.array([x1, x2]), w_and, bias_and)

# XOR 게이트
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

for x in [(0,0), (1,0), (0,1), (1,1)]:
    y_hat = XOR(x[0], x[1])
    print("입력값 : " + str(x) + "출력 값 : " + str(y_hat))
    
y = [(0, 1, 1, 0)]
y = numpy.array(y)
print("실제 값 : " + str(y))

numpy.array([[0], [1], [1], [0]])
numpy.array([0, 1, 1, 0])