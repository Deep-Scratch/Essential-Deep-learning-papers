# S00-02. AlexNet

[AlexNet: ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Alex Krizhevsky et al.

**reviewed by [Heekyung Park](https://github.com/HeeKyung-Park)**

## 1. 요약

- 2012년도 ImageNet ILSVRC에서 우승한 모형
- 딥러닝 발전에 있어서, turning point라고 볼 수 있다.
- 5개의 Convolutional Layer와 3개의 Fully Connected Layer로 구성되어 있다.
- Contribution
	+ ReLU Nonlinearity
	+ Training on Multiple GPUs
	+ Local Response Normalization
	+ Overlapping Pooling
- Reducing Overfitting
	+ Data Augmentation
	+ Drop out


## 2. 전문 

### 0. Abstract
![figure01](https://i.imgur.com/os7hG2Pm.png?1)


ImageNet LSVRC-2010 학습, 테스트 결과 top-1, top-5 에러가 각각 37.5%와 17.0%로 이전 최신의 방법의 방법보다 좋은 퍼포먼스를 보였다.
사용된 인공신경망은 5개의 convolutional layer와 3개의 fully-connected layer로 구성되어 있으며, 6천만개의 parameter와 65만개의 뉴런이 사용되었다.
오버피팅을 줄이기 위해 dropout을 사용하였고, 그 효과가 큼을 증명했다. ImageNet ILSVRC-2012에서 top-5 에러가 15.3%를 달성하며 2등과 꽤 큰 차이를 가졌다. 

### 1.Introduction
수 만개의 이미지에서 수천개의 object를 학습하기 위해서는, capacity가 큰 모델이 필요하다.
Convolutional neural networks(CNN)은 기존의 feedforward neural networks에 비하면 더 적은 connection과 parameter로 구성된다.
CNN의 매력적인 성질에도 불구하고, 큰 scale의 고해상도 이미지에 적용하기에는 한계가 있다.
하지만, GPU에 발전으로 ILSVRC-2010, 2012의 데이터셋에서 CNN의 구현이 가능해졌다. 

### 2. The Dataset
![figure2](https://i.imgur.com/rWT9YOT.png)
ImageNet은 22,000 카테고리로 구성된 1,500만개의 고해상도 이미지 데이터셋으로, ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)는 ImageNet 데이터 중, 1,000개의 카테고리만을 사용한다. 
120만개의 training data, 5만개의 validation data, 15만개의 test data로 이루어져 있다.

<br/>
ILSVRC에서는 top-1, top-5 에러를 평가 척도로 사용한다. 
- top-1 : 이미지 하나에 가장 큰 확률을 갖은 예측 label과 실제 label이 맞으면 정답, 이외에는 오답
- top-5 : 이미지 하나에 가장 큰 5개의 확률을 갖은 예측 label에 실제 label이 포함되면 정답, 이외에는 오답

![figure3](https://i.imgur.com/B05g6Kt.png)
ImageNet은 여러 스케일의 이미지로 구성되어 있기 때문에 다음과 같은 방식으로 이미지를 256×256 사이즈로 고정하였다.
1. 이미지에서 짧은 면을 256으로 사이즈를 조정한다.
2. rescaling된 이미지에서 가온데 만을 추출 (256×256) 


### 3. The Architecture
![figure4](https://i.imgur.com/JOS2mHU.png)
5개의 convolutional layer와 3개의 fully-connected layer로 수어

#### 3.1 ReLU Nonlinearity
![figure5](https://i.imgur.com/fUojTVO.png)
- Sigmoid, tanh 활성화 함수는 saturating nonlinearity로서, gradient vanishing 문제(gradient 손실, -∞, +∞ 에서의 기울기가 0에 가까움)를 야기시킨다. (위의 왼쪽 그래프) 
- ReLU(Rectified Linear Units)는 non-saturating nonlinearity로서 gradient vanishing을 어느정도 해소하며, tanh 함수에 비해 약 6배나 빠른 속도로 학습한다. (위의 오른쪽 그래프)

#### 3.2 Training on Multiple GPUs
![figure6](https://i.imgur.com/OA0VmPb.png)
- GPU 2개를 parallelization을 하여 사용. 
- GPU1은 color-agnostic, GPU2는 color-specific을 주로 학습하였다. 
- GPU는 parallelization하게 사용되지만, 3번째 convolutional layer에서는 communication을 한다. 
- top-1, top-5 에러를 각각 1.7%, 1.2% 상승
- (Reviewer think) AlexNet이 나온 2012년도에 비해, GPU의 성능이 어마어마하게 좋아졌기 때문에 요즘은 GPU를 parallelization해서 사용하지는 않는것 같다.

#### 3.3 Local Response Normalization
![figure7](https://i.imgur.com/MkSrmFi.png)
- ReLU는 입력값에 대한 normalization이 필요없다.
- 하지만, 출력값이 양수인 경우, 굉장히 큰 값을 갖을 수 있다. 
- 이를 normalization하기 위해 local Response Normalization을 사용한다.
- $k, n, \alpha, \beta$는 hyper-parameter이다. 
- top-1, top-5 에러를 각각 1.4%, 1.2% 상승
- (Reviewer think) 최신의 CNN에서는 위와 같은 normalization 대신, batch normalization을 주로 사용한다. 


#### 3.4 Overlapping Pooling 
![figure8](https://i.imgur.com/336CWYI.png)
- pooling layer에서 stride($s$)보다 pooling window($z$)를 크게 사용하여, pooling window를 겹쳐서 사용하였다.
- top-1, top-5 에러를 각각 0.4%, 0.3% 상승
- (Reviewer think) 이후의 CNN에서는 주로 Pooling을 겹치지 않으며, 심지어 Pooling layer를 안쓰는 경우도 많다.


#### 3.5 Overall Architecture
![figure9](https://i.imgur.com/zdNsLyR.png)
※ 논문에서는 input size가 224×224로 되어있지만, 실제로 11×11 filter size, 4 stride를 사용하면 output이 55×55로 성사되지 않는다. 이는 아마 padding을 이용하여 input을 227×227 맞췄을 것이라는 추측이 있다. (http://cs231n.github.io/convolutional-networks/)
※ 또한, 논문 figure2 에서 첫번째 layer의 뉴런수가 253,440으로 되어 있지만, 이는 무언가 실수가 있는 듯 싶다. neuron의 개수는 55×55×96=290,400개 이다. (https://stackoverflow.com/questions/36733636/the-number-of-neurons-in-alexnet)

![figure10](https://i.imgur.com/YWU9hUX.png)

![figure11](https://i.imgur.com/SGB2oj4.png)

![figure12](https://i.imgur.com/8PXJgrR.png)

![figure13](https://i.imgur.com/myqK0cX.png)


### 4. Reducing Overfitting
Alexnet은 6천만개의 parameter로 구성되어 있기 때문에, overfitting의 가능성이 있다. 이를 위해 크게 2가지 방법을 사용하였다.

#### 4.1 Data Augmentation
![figure14](https://i.imgur.com/NfVN3vd.png)
1. Generating image translations and horizontal reflections
	+ 이미지를 좌우로 대칭시켜, 256×256 size의 이미지에서 224×224 patch를 추출(한 이미지에서 32×32××2=2,048개의 patch를 추출할 수 있다.)
	+ test할때는, 이미지의 4개의 모퉁이와 가운데의 patch를 추출(5×2=10)하여, 10개의 softmax를 평균하여 예측한다.
![figure15](https://i.imgur.com/62UlbXH.png)

2. Altering the intensities of the RGB channels
![figure15](https://i.imgur.com/c0PTB5S.png)
	+ PCA를 이용한 color jittering

#### 4.2 Dropout
![figure16](https://i.imgur.com/cORdgUk.png)

0.5의 확률로 뉴런을 0으로 셋팅하는 dropout을 시행하여 학습한다. test할때는 모든 뉴런을 사용하지만, 각 출력값에 0.5를 곱하여 사용.


### 5. Details of learning
- 128 배치 사이즈
- Stochastic gradient descent, 0.9 momentum, 0.0005 weight decay
- zero-mean Gaussain(0.01 standard deviation) 초기화
- 2, 4, 5번째의 convolutional layer와 fully-connected layer에서는 bias를 1로 초기화(나머지는 0)는 학습을 가속화한다.(ReLU의 양수부분을 줌으로써)
- 초기 learning rate은 0.01이고, validation error가 줄어들지 않는다면 10으로 나누어 준다.
- NVIDIA GTX 580 3GB GPU 사용


### 6. Results
![figure17](https://i.imgur.com/vJVgsFj.png)
: ILSVRC-2010, 2012에서 모두 우수한 퍼포먼스를 보여준다.  

### 7. Discussion
- 중간의 hidden layer가 하나라도 빠지면, top-1 에러가 2% 상승한다. 즉, CNN에서의 depth는 매우 중요하다.
