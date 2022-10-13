# 지진예측을 위한 머신러닝 기법 설계

## 1. 프로젝트 소개
* 프로젝트 개요<br>
최근 딥러닝은 우리 사회의 다양한 응용분야에 적용되고 있고 이의 중요성이 더욱 주목받고 있다. 기상과 자연재해 등 예측 불가능하던 상황을 예측할 수 있고, 그에 따라 미리 대응하여 피해를 줄일 수 있다는 점에서 해당 분야에서는 점점 더 중요성이 커지고 있다. 특히 본 연구에서는 지진에 대해 다룰 것이다. 최근 지진으로 인한 피해가 많이 발생하는 만큼, 지진 예측에 있어서 높은 정확도와 예측 속도가 빠른 모델의 개발을 목표로 한다.

* 프로젝트 목표<br>
기존에 개발된 지진 예측 모델인 EQTransformer와 이를 경량화한 LEQNet의 구조를 변경하여 한계를 극복하고자 한다. EQTransformer는 높은 정확도를 갖는 반면, 모델의 크기가 크고 추론 속도가 느리다는 한계가 있다. 이를 실시간 지진 예측에 활용할 수 있도록 빠른 추론 속도와 작은 모델 사이즈로 사용 가능성을 확대하는 것을 목표로 한다.

## 2. 팀 소개
정지호, jjh990329@pusan.ac.kr, 데이터 분석 및 실시간 예측 기능 구현<br>
최지환, wlghks407@pusan.ac.kr, 모델 최적화 및 성능 개선<br>
김주은, bbbvovo123@pusan.ac.kr, 모델 구조 분석 및 개선된 모델 구조 개발<br>

## 3. 구성도
우리가 개발한 모델의 구조는 다음과 같다.

![LSTM,GRU](https://user-images.githubusercontent.com/84946412/195488921-7593bf3e-1af7-4b81-a4c6-97ad30ad83ac.png)
<br>

### 각 구조의 역할

* Encoder : 데이터를 압축해서 표현<br>
* CNN : 데이터의 특징을 학습<br>
* LSTM : 시계열 데이터의 특징을 학습<br>
* Transformer : 여러개의 Encoder와 Decoder를 연결<br>
* Decoder : 압축된 데이터를 다른 시계열 데이터로 변환<br>

### 기존 모델과 차이점

* 기존 모델의 Decoder 상위 부분에서 사용된 LSTM대신 GRU구조를 사용 

* CNN(Deeper Bottleneck), RNN(LSTM,GRU) Block의 깊이를 ½ 감소 

## 4. 소개 및 시연 영상

## 5. 설치 및 사용법

* Links
  * STEAD - 학습에 사용된 DataSet
    * https://github.com/smousavi05/STEAD
  * EQTransformer - 설치법 참고
    * https://github.com/smousavi05/EQTransformer
  * LEQNet - EQTransformer 경량화 모델
    * https://github.com/LEQNet/LEQNet

* 실행 환경<br>
'python==3.8.10'<br>
'tensorflow==2.2.0'  # tensorflow <2.7.0 needs numpy <1.20.0<br>
'tensorflow-estimator==2.0.0'<br>
'keras==2.3.1'<br>
'scipy==1.4.1'<br>
