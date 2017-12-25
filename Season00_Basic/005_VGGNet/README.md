# 005. VGGNet, VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

https://arxiv.org/abs/1409.1556

Oxford의 Visual Geometry Group(VGG)에서 개발해서 VGGNet이라고도 불린다.

by Karen Simonyan, Andrew Zisserman

## 1. 요약

- 매우 단순한 구조라서 지금도 널리 쓰이고 있다.
- 다른 configuration을 fix시키고 Depth가 미치는 영향에 대해서 연구. depth를 깊게 했을 때 성능도 좋아진다. 아쉽게도 19 layers 쌓으니 saturated.
- layers
    + conv layer, FC layer 뒤에 무조건 ReLU 따라옴
    + pooling 하면서 resolution을 반으로 줄이고, feature map depth는 2배로 늘림(512까지)
- small filter(3x3) 사용
    + depth를 늘릴 수 있음
    + 3x3 filter를 2개 layer로 겹치면 5x5 filter와 같고, 3개 겹치면 7x7 filter와 같다.
    + parameter를 줄여서 regularization 효과가 나타남
- 1x1 filter를 사용한 것은 FC Layer를 추가한 것과도 같으며(input-output 차원이 같음), 레이어 뒤에 ReLU를 붙일 수 있기 때문에 비선형성 강화
- Training: 224x224 고정 이미지를 받아들이기 위해 이미지 스케일링 사용
- Test: 다양한 input image size를 처리하기 위해 알렉스넷에선 multiple crop을 썼지만 여기선 dense evalutation을 사용

## 2. 전문 번역

### 0. Abstract

In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3 × 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

> 이 논문은 CNN의 depth가 대용량 이미지 인식의 정확도에 어떤 영향을 끼치는지 살펴본다. 논문은 매우 작은 convolution filter(3 x 3)를 사용해서 depth를 증가시키며 network를 철저하게 평가한다. 이전의 state-of-the-art에 비하여 매우 큰 성능 향상을 보이며 depth를 16-19개의 weight layer들로 확장할 수 있다. 이러한 발견은 2014 ImageNet 챌린지를 기반으로 하고, 우리 팀은 localisation과 classification 트랙에서 각가 1, 2위를 기록했다. 또한 우리의 representation은 다른 데이터셋에서도 좋은 generalization과 state-of-the-art 결과를 보인다. 우리는 두 개의 최고의 성능을 내는 ConvNet 모델을 공개적으로 사용할 수 있도록 했고, 컴퓨터 비전에서 deep 레이어를 사용하는 연구에 도움이 될 것이다.

### 1. Introduction

Convolutional networks (ConvNets) have recently enjoyed a great success in large-scale image and video recognition (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014) which has become possible due to the large public image repositories, such as ImageNet (Deng et al., 2009), and high-performance computing systems, such as GPUs or large-scale distributed clusters (Dean et al., 2012). In particular, an important role in the advance of deep visual recognition architectures has been played by the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) (Russakovsky et al., 2014), which has served as a testbed for a few generations of large-scale image classification systems, from high-dimensional shallow feature encodings (Perronnin et al., 2010) (the winner of ILSVRC-2011) to deep ConvNets (Krizhevsky et al., 2012) (the winner of ILSVRC-2012).

> ConvNets는 최근 대용량 이미지와 영상 인식에서 큰 성공을 거뒀다. 이는 ImageNet 같은 대용량의 공개 이미지 저장소와, GPU나 분산처리 같은 고성능의 컴퓨팅 시스템의 덕분이다. 특히 ILSVRC 챌린지가 deep visual recognition 아키텍처 발전에 중요한 역할을 했는데 새로운 대용량 이미지 분류 시스템의 생성에 테스트베드 역할을 했다.(2010년 high-dimensional~~ 논문부터 2012년 deep ConvNets까지)

With ConvNets becoming more of a commodity in the computer vision field, a number of attempts have been made to improve the original architecture of Krizhevsky et al. (2012) in a bid to achieve better accuracy. For instance, the best-performing submissions to the ILSVRC- 2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014) utilised smaller receptive window size and smaller stride of the first convolutional layer. Another line of improvements dealt with training and testing the networks densely over the whole image and over multiple scales (Sermanet et al., 2014; Howard, 2014). In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3 × 3) convolution filters in all layers.

> ConvNets가 CV 필드에서 점점 많은 역할을 함에 따라 더 나은 정확도를 위해 기존 아키텍처(알렉스넷 같은)를 발전시키려는 노력이 많이 생겨났다. 예를 들어 ILSVRC 2013의 최고 성능 작품은 window size와 첫 convolutional layer의 stride를 작게 사용함으로서 달성했다. 다른 우수작은 전체 이미지를 여러 scale로 밀도있게 네트워크를 train, test하는 방식이었다. 우리 논문은 이와 다르게 ConvNets 아키텍처에서 depth라는 중요한 부분을 다룰 것이다. 다른 parameter들을 고정하고 네트워크의 depth를 Convolutional layer를 추가하는 방식으로 점진적으로 늘릴텐데 이는 모든 레이어에서 매우 작은 convlutional filter(3 x 3)를 사용하기 때문에 가능하다.

As a result, we come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets, where they achieve excellent performance even when used as a part of a relatively simple pipelines (e.g. deep features classified by a linear SVM without fine-tuning). We have released our two best-performing models to facilitate further research.

> 결과적으로 우리는 ConvNet 아키텍처에서 매우 높은 정확도를 달성했다. ILSVRC 뿐만 아니라 다른 이미지 인식 데이터셋에 대해서도 상대적으로 단순한 파이프라인을 사용하면서 매우 훌륭한 결과를 냈다. 우리는 다른 연구를 위해 두 가지 가장 좋은 성능을 낸 모델들을 공개했다.

The rest of the paper is organised as follows. In Sect. 2, we describe our ConvNet configurations. The details of the image classification training and evaluation are then presented in Sect. 3, and the configurations are compared on the ILSVRC classification task in Sect. 4. Sect. 5 concludes the paper. For completeness, we also describe and assess our ILSVRC-2014 object localisation system in Appendix A, and discuss the generalisation of very deep features to other datasets in Appendix B. Finally, Appendix C contains the list of major paper revisions.

> 아래 부분은 다음처럼 구성돼있다. Section 2엔 ConvNet의 설정 내용들, Section 3엔 이미지 분류 학습과 평가에 대한 자세한 내용들, Section 4엔 다른 모델들과의 비교를, Section 5엔 결론을 담았다. 논문의 완결을 위해 부록 A엔 모델에 대한 평가 결과를, B엔 다른 데이터셋에 대한 generalization에 대한 생각들, C엔 논문의 수정 기록을 담았다.

### 2. ConvNet Configurations

To measure the improvement brought by the increased ConvNet depth in a fair setting, all our ConvNet layer configurations are designed using the same principles, inspired by Ciresan et al. (2011); Krizhevsky et al. (2012). In this section, we first describe a generic layout of our ConvNet configurations (Sect. 2.1) and then detail the specific configurations used in the evaluation (Sect. 2.2). Our design choices are then discussed and compared to the prior art in Sect. 2.3.

> ConvNet depth를 증가시키는 것에 대한 성능 향상을 공정히 측정하기 위해 모든 ConvNet layer의 configuration은 같은 원칙에 의해 설계되었다. 먼저 2.1에서 ConvNet configuration의 기본적인 레이아웃을 설명하고, 2.2에서 특별한 부분을 자세하게 다루겠다. 2.3에선 이전 최고 성능 작품과 비교해서 왜 우리가 이렇게 configuration을 설정했는지 다루겠다.

#### 2.1 Architecture

During training, the input to our ConvNets is a fixed-size 224 × 224 RGB image. The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel. The image is passed through a stack of convolutional (conv.) layers, where we use filters with a very small receptive field: 3 × 3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations we also utilise 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv.layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1 pixel for 3 × 3 conv.layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.layers (not all the conv.layers are followed by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.

> 학습할 때 ConvNet의 입력은 224 x 224 RGB(3 dim)의 고정 사이즈 이미지다. 전처리는 단 한가지만 했는데 train 데이터셋 이미지의 모든 픽셀에 대해 평균 RGB 값을 뺐다. 이미지는 중첩된 Conv layer를 통과하는데 아주 작은 3 x 3 필터를 사용했다. 이 사이즈는 어떤 의미를 뽑아낼 때 가능한 가장 작은 사이즈다. 또한 1 x 1 Convolution filter도 사용했는데 input 채널에서 linear transformation으로서 쓰인다. convolution stride는 1 픽셀로 고정되어서 resolution이 유지된다. 즉 패딩도 1 픽셀이다. spatial pooling은 5개의 맥스 풀링 레이어로 수행되며 몇 개의 convolution layer의 뒤에 존재한다.(모든 conv layer 뒤에 맥스 풀링 layer가 있는건 아니다.) 맥스 풀링은 2 x 2 윈도우에 2 stride로 이루어져있다

A stack of convolutional layers (which has a different depth in different architectures) is followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000 way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

> 서로 다른 depth와 다른 아키텍처를 가지고 있는 convolutional layer들의 층 뒤에는 Fully-Connected 레이어 3개가 있다. 처음 두 레이어는 4096 채널이고, 세 번째는 ILSVRC가 1000개 클래스를 분류하므로 1000 채널을 가진다. 이후의 마지막 레이어는 soft-max 레이어다. fully connected layer의 설정들은 다른 모든 네트워크들과 동일하다.

All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity. We note that none of our networks (except for one) contain Local Response Normalisation (LRN) normalisation (Krizhevsky et al., 2012): as will be shown in Sect. 4, such normalisation does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time. Where applicable, the parameters for the LRN layer are those of (Krizhevsky et al., 2012).

> 모든 히든 레이어는 ReLU를 사용한다. 네트워크 중 단 하나만 Local Response Normalization을 사용하고 나머지는 사용하지 않는다. 이런 정규화는 ILSVRC 데이터셋에서 성능을 향상시키지 않고 메모리 소비와 연산 시간만 늘인다.

#### 2.2 Configurations

![config-table1](http://i.imgur.com/JcOcAU5.png)

> Table 1 : A에서 E로 갈수록 depth가 깊어지고, notation은 `conv[receptive field size]-[number of channels]`이다. ReLU activation function은 간결함을 위해 표시하지 않았다.

The ConvNet configurations, evaluated in this paper, are outlined in Table 1, one per column. In the following we will refer to the nets by their names (A–E). All configurations follow the generic design presented in Sect. 2.1, and differ only in the depth: from 11 weight layers in the network A (8 conv. and 3 FC layers) to 19 weight layers in the network E (16 conv. and 3 FC layers). The width of conv.layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

> ConvNet의 configurations는 위 Table 1에 적혀있다. 컬럼이 하나의 모델이다. 다음부터 각각의 컬럼을 A-E로 지칭할 것이다. A부터 E까지 모든 네트워크의 configuration은 2.1에서 설명한 기본 디자인을 따르고 오직 depth만 달라진다. Convolutional layer들의 width, 즉 채널의 개수는 다소 작다. 첫 번째 레이어에서 64개로 시작하고 맥스 풀링 레이어를 지날 때마다 2배씩 증가하는데 512에 도달하면 멈춘다.

![config-table2](http://i.imgur.com/om9UTWn.png)

In Table 2 we report the number of parameters for each configuration. In spite of a large depth, the number of weights in our nets is not greater than the number of weights in a more shallow net with larger conv.layer widths and receptive fields (144M weights in (Sermanet et al., 2014)).

> Table 2는 각 configuration마다 parameter가 몇 개인지를 보여준다. 깊은 네트워크의 weights 수는 더 얕은 depth와 넓은 conv를 가진 네트워크보다 생각보다 그렇게 많지 않다.

#### 2.3 Discussioin

Our ConvNet configurations are quite different from the ones used in the top-performing entries of the ILSVRC-2012 (Krizhevsky et al., 2012) and ILSVRC-2013 competitions (Zeiler & Fergus, 2013; Sermanet et al., 2014). Rather than using relatively large receptive fields in the first conv.layers (e.g. 11 × 11 with stride 4 in (Krizhevsky et al., 2012), or 7 × 7 with stride 2 in (Zeiler & Fergus, 2013; Sermanet et al., 2014)), we use very small 3 × 3 receptive fields throughout the whole net, which are convolved with the input at every pixel (with stride 1). It is easy to see that a stack of two 3 × 3 conv.layers (without spatial pooling in between) has an effective receptive field of 5 × 5; three such layers have a 7 × 7 effective receptive field. So what have we gained by using, for instance, a stack of three 3 × 3 conv.layers instead of a single 7 × 7 layer? First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative. Second, we decrease the number of parameters: assuming that both the input and the output of a three-layer 3 × 3 convolution stack has C channels, the stack is parametrised by `3(3^2*C^2) = 27*C^2` weights; at the same time, a single 7 × 7 conv.layer would require `7^2*C^2 = 49*C^2` parameters, i.e. 81% more. This can be seen as imposing a regularisation on the 7 × 7 conv. filters, forcing them to have a decomposition through the 3 × 3 filters (with non-linearity injected in between).

> 우리 ConvNet의 configuration은 2012년 최고의 성능을 냈던 알렉스넷과 2013년 최고작과 많이 다르다. 첫 번째 convolution layer에서 큰 receptive field(filter size)를 사용하기보다(알렉스넷: 11 x 11, 4 stride / 2013: 7 x 7, 2 stride) 전체 네트워크에서 더 작은 3 x 3 filter와 1 stride를 사용했다. 맥스 풀링이 없는 2개의 3 x 3 convolution layer가 연속해서 등장하면 5 x 5를, 3개 레이어라면 7 x 7와 같은 효과를 내는 것을 볼 수 있다. 그렇다면 우리가 이 실험을 통해 얻은 결론은 무엇일까. 예를 들어 작은 필터의 3개 레이어를 겹치면 더 큰 필터 사이즈의 레이어 1개와 같다는 것? 첫째로 1개의 비선형 ReLU를 사용하는 대신 3개의 ReLU를 사용함으로서 더욱 모델을 discriminative하게 만들었다. 둘째로 parameter의 개수를 줄였다. 레이어 3개에 3 x 3 사이즈라면 총 개수는 `27*C^2`가 될 것이고, 레이어 하나에 7 x 7 사이즈라면 `49*C^2`개의 parameter일 것이다. 81% 더 많다. 이것은 7 x 7 filter를 사이 사이에 non-linearity가 들어간 3 x 3 filter로 분해하는 뛰어난 정규화라고 볼 수 있다.

The incorporation of 1 × 1 conv.layers (configuration C, Table 1) is a way to increase the non-linearity of the decision function without affecting the receptive fields of the conv.layers. Even though in our case the 1 × 1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function. It should be noted that 1 × 1 conv.layers have recently been utilised in the “Network in Network” architecture of Lin et al. (2014).

> Table 1의 C 네트워크를 보면 1 x 1 filter가 존재한다. 이것은 receptive field에 영향을 주지 않으면서 non-linearity 속성을 크게 만드는 방법이다. 1 x 1 convolution이 결국 같은 차원으로의 선형 projection일 뿐이지만 ReLU가 붙기 때문에 달라진다. 이는 최근 Network in Network 아키텍처에서 사용되었다.

Small-size convolution filters have been previously used by Ciresan et al. (2011), but their nets are significantly less deep than ours, and they did not evaluate on the large-scale ILSVRC dataset. Goodfellow et al. (2014) applied deep ConvNets (11 weight layers) to the task of street number recognition, and showed that the increased depth led to better performance. GoogLeNet (Szegedy et al., 2014), a top-performing entry of the ILSVRC-2014 classification task, was developed independently of our work, but is similar in that it is based on very deep ConvNets(22 weight layers) and small convolution filters (apart from 3 × 3, they also use 1 × 1 and 5 × 5 convolutions). Their network topology is, however, more complex than ours, and the spatial resolution of the feature maps is reduced more aggressively in the first layers to decrease the amount of computation. As will be shown in Sect. 4.5, our model is outperforming that of Szegedy et al. (2014) in terms of the single-network classification accuracy.

> 작은 사이즈의 convolution filter는 이전에도 사용되었지만 너무 얕은 네트워크였다. 그리고 ILSVRC같은 대량의 데이터셋에 적용되지 않았다. Goodfellow는 11 weight layer를 가지는 deep ConvNets를 도로 숫자 인식에 사용했고, 깊은 depth가 성능 향상을 이끈다는 것을 보여줬다. 2014년 최고작인 GoogLeNet은 우리의 작업과 독립적으로 개발되었지만 깊은 depth(22)와 작은 filter 사이즈(3, 1, 5)를 사용했다는 점에서 비슷하다. 하지만 GoogLeNet은 우리에 비해 매우 복잡하고, 연산 비용 절감을 위해 resolution을 줄이는 과정이 너무 공격적이다. 4.5 섹션에서 우리 모델이 GoogLeNet보다 single-network 분류에선 더 뛰어남을 보여줄 것이다.

### 3. Classification Framework

In the previous section we presented the details of our network configurations. In this section, we describe the details of classification ConvNet training and evaluation.

> 이전 섹션에서 네트워크 configuration에 대해 설명했다. 이번 섹션에선 ConvNet의 학습과 평가에 대해 상세하게 다룰 것이다.

#### 3.1 Training

The ConvNet training procedure generally follows Krizhevsky et al. (2012) (except for sampling the input crops from multi-scale training images, as explained later). Namely, the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent (based on back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 256, momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5 · 10−4) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving. In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs). We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv. filter sizes; (b) pre-initialisation of certain layers.

> ConvNet 학습 과정은 나중에 설명할 이미지 샘플링 방식을 제외하고는 알렉스넷을 따른다. 학습은 다변량 로지스틱 회귀를 사용하는데 mini-batch gradient descent 알고리즘과 momentum을 이용한다. batch 사이즈는 256이고 momentum은 0.9다. L2 regularizor를 사용하는데 상수는 `5*10e-4` 값이다. 처음 두 개의 fully-connected layer에서 dropout을 사용하는데 비율은 0.5다. learning rate은 0.01부터 시작해서 validation 정확도가 더이상 좋아지지 않을 때마다 10분의 1씩 한다. learning rate은 3번 감소했으며 37만 번 반복(74 epochs) 후 학습이 종료됐다. 알렉스넷과 비교해 더 깊고 많은 parameter 수에도 불구하고 더 적은 epochs를 가지는 이유를 두 가지로 추측한다. 첫째로 깊은 depth와 작은 filter 사이즈로 인한 자연스러운 정규화, 둘째로 특정 레이어에 대한 사전 initialization 덕분이다.

The initialisation of the network weights is important, since bad initialisation can stall learning due to the instability of gradient in deep nets. To circumvent this problem, we began with training the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when training deeper architectures, we initialised the first four convolutional layers and the last three fully-connected layers with the layers of net A (the intermediate layers were initialised randomly). We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning. For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean and 10−2 variance. The biases were initialised with zero. It is worth noting that after the paper submission we found that it is possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).

> weight의 초기화는 중요하다. 나쁜 초기화는 딥 네트워크에서 gradient를 불안정하게 만들기 때문이다. 이런 문제를 우회하기 위해 랜덤 가중치로 초기화해도 괜찮을만큼 얕은 A 네트워크(VGG 11: 8 conv, 3 fc)를 학습시켰다. 그리고 더 깊은 네트워크를 학습할 때 처음 4개의 convolutional layer와 마지막 3개의 fully-connected layer들을 A에서 만들어진 가중치로 초기화했다. 미리 학습된 가중치로 초기화된 레이어들에 대해서는 learning rate를 줄이지 않았다. 랜덤 초기화에 대해선 평균 0, `10^-2` 분산인 정규분포에서 가중치를 샘플링했다. bias는 모두 0으로 시작했다. 논문을 제출한 후 Glorot과 Bengio가 쓴 논문에서 미리 학습된 가중치가 아니라 그냥 랜덤 초기화를 해도 괜찮다는 사실을 알았다.

To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration). To further augment the training set, the crops underwent random horizontal flipping and random RGB colour shift (Krizhevsky et al., 2012). Training image rescaling is explained below.

> 224x224로 고정된 이미지를 얻기 위해 크기가 조정된 학습 이미지로부터 랜덤하게 해당 크기로 잘랐다. 학습 데이터셋을 좀 더 조정하기 위해 알렉스넷에서처럼 랜덤으로 수평 뒤집기도 하고, RGB 컬러를 바꾸기도 했다. 크기를 조절하는 것은 아래에 설명되어있다.

Training image size. Let S be the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped (we also refer to S as the training scale). While the crop size is fixed to 224 × 224, in principle S can take on any value not less than 224: for S = 224 the crop will capture whole-image statistics, completely spanning the smallest side of a training image; for S ≫ 224 the crop will correspond to a small part of the image, containing a small object or an object part.

> 학습 이미지를 원본 비율과 동일하게 크기를 조절한 후 가로 세로 중 더 작은 쪽을 S라고 하자. 224 x 224가 crop할 사이즈이므로 이미지의 S는 224보다 작아선 안된다. 만약 S가 224라면 이미지 전체를 이용할 수 있고, S가 224보다 크다면 이미지의 일부분만 사용하는게 된다.

We consider two approaches for setting the training scale S. The first is to fix S, which corresponds to single-scale training (note that image content within the sampled crops can still represent multi- scale image statistics). In our experiments, we evaluated models trained at two fixed scales: S = 256 (which has been widely used in the prior art (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014)) and S = 384. Given a ConvNet configuration, we first trained the network using S = 256. To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 10−3.

> S의 스케일을 조절하기 위해 2가지 방법을 사용한다. 첫 번째는 S를 고정하는 Single-scale training이다. 2가지 고정 S(256, 384)를 가지고 모델을 평가했다. 256은 이전에 많이 사용됐던 것이다. 먼저 256을 가지고 학습을 한다. 384 네트워크의 학습 속도를 높이기 위해 256에서 학습된 가중치를 사용해서 초기화하고 0.001의 작은 learning rate로 학습한다.

The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range `[Smin,Smax]`(we used Smin = 256 and Smax = 512). Since objects in images can be of different size, it is beneficial to take this into account during training. This can also be seen as training set augmentation by scale jittering, where a single model is trained to recognise objects over a wide range of scales. For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 384.

> 두 번째 방식은 Multi-scale 방식으로 이미지가 S의 최소, 최대값 사이의 랜덤값으로 스케일 조정이 되는 것인데 최소는 256, 최대는 512를 사용했다. 이미지의 사물이 다양한 크기가 될 수 있으므로 이 방식을 사용하면 도움이 된다. 단일 모델이 다양한 스케일에 대해 인식을 해야할 때 scale jittering을 활용한 augmentation을 하는데 이와 비슷하다. 속도를 위해 384 고정 single scale 모델로 학습하고 이 가중치를 모든 레이어에서 fine-tuning하는 방식으로 사용한다.

#### 3.2 Testing

At test time, given a trained ConvNet and an input image, it is classified in the following way. First, it is isotropically rescaled to a pre-defined smallest image side, denoted as Q (we also refer to it as the test scale). We note that Q is not necessarily equal to the training scale S (as we will show in Sect. 4, using several values of Q for each S leads to improved performance). Then, the network is applied densely over the rescaled test image in a way similar to (Sermanet et al., 2014). Namely, the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled). We also augment the test set by horizontal flipping of the images; the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores for the image.

> test 할 때 트레이닝된 네트워크와 인풋 이미지가 주어지고 다음처럼 분류된다. 먼저 미리 정해진 이미지의 작은 쪽 크기를 같은 비율로 크기 조정한다. 이를 "Q" 또는 "test scale"이라 한다. Q는 training의 S scale과 같은 크기가 될 필요는 없다. 섹션 4에서 각 S값에 대하여 여러 Q값을 활용하니 성능이 좋아짐을 보일 것이다. 둘째로 네트워크가 rescale된 이미지에 dense하게 적용된다. 즉 먼저 FC layer로 이미지를 받아들이고, 이것이 conv layer로 전환된다. 처음 FC layer는 7x7 conv layer로 바뀌고, 마지막 2개의 FC layer는 1x1 conv layer로 바뀐다. 이러한 네트워크가 crop되지 않은 전체 이미지에 적용된다. 결과는 class score map이고 dimension은 전체 채널 수(1000개) by input image resolution에 따른 다양한 값의 매트릭스로 나타내어진다. 이 매트릭스는 고정된 class score vector를 갖기 위해서 sum-pool을 한다. 또한 test 이미지를 수평 flip 등을 이용해 augment하고, 원본 이미지와 soft-max를 하여 최종 예측값으로 평균낸다.

Since the fully-convolutional network is applied over the whole image, there is no need to sample multiple crops at test time (Krizhevsky et al., 2012), which is less efficient as it requires network re-computation for each crop. At the same time, using a large set of crops, as done by Szegedy et al. (2014), can lead to improved accuracy, as it results in a finer sampling of the input image compared to the fully-convolutional net. Also, multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, while in the case of dense evaluation the padding for the same crop naturally comes from the neighbouring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured. While we believe that in practice the increased computation time of multiple crops does not justify the potential gains in accuracy, for reference we also evaluate our networks using 50 crops per scale (5 × 5 regular grid with 2 flips), for a total of 150 crops over 3 scales, which is comparable to 144 crops over 4 scales used by Szegedy et al. (2014).

> 전체 이미지가 네트워크에 적용되기 때문에 테스트할 때 multiple crop을 할 필요가 없다. AlexNet에서처럼 multiple crop을 할 경우 각각의 crop마다 computation을 처음부터 돌려야하기 때문에 비효율적이다. 하지만 동시에 대량의 crop set을 사용하하는 것은 세밀한 샘플링을 하는 것이므로 정확도를 높일 수 있다. 또한 multi-crop evaluation은 다른 convolution boundary 조건 때문에 dense evaluation과 상보적 관계다. ConvNet을 crop에 적용할 때 가장자리의 픽셀들은 zero-padding되어 사용된다. 반면에 dense evaluation에선 자연스럽게 전체 이미지에서 바로 근처의 이웃 픽셀로 padding되기 때문에 더 많은 context를 잡아낼 수 있다. 비교를 위해 5x5 grid와 2 flip을 사용해서 scale당 50개 crop(3개 scale이므로 150개)을 사용해서 AlexNet의 144개 crop과 비교해봤고, multiple crop을 사용해서 연산 시간을 늘리는 것이 더 나은 정확도를 도출하지 않는다고 결론지었다.
