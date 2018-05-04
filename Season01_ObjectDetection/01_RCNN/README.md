# S01-01. R-CNN

[R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) by Ross Girshick, et al.

- **reviewed by [Gyubin Son](https://github.com/gyubin)**  
- **summarized by [Boseop Kim](https://github.com/aisolab)**  

# R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation
본 논문 (R-CNN)은 ***Object detction에 Convolutional Neural Network를 feature extractor로 사용한 논문*** 으로 이후 Fast R-CNN, Faster R-CNN 등 여러가지 논문의 기반이 되는 논문입니다. 저술되어 공개된 지 오래된 논문이 만큼 여러 report가 존재하며, 본 포스트는 [Rich feature hierarchies for accurate object detection and semantic segmentation
Tech report (v5)](https://arxiv.org/pdf/1311.2524.pdf)에 기초하여 작성하였으며, 중요한 idea만 다루고 있습니다. 상세한 내용은 논문을 보시면 좋을 듯 합니다.
- - -
## Abstract
본 논문에서는 ***mAP (mean Avereage Precision)*** 를 기준으로 VOC 2012의 best result와 비교하여, 30% 이상의 성능 향상을 (mAP : 53.3%) 이루었다고 말하며, 그 기반이 되는 아이디어는 아래의 두 가지입니다.

- *One can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposal in order to localize and segment objects.*
- *When labeled traning data is scarce, supervised pre-training for an auxiliary task. followed by domain-specific fine-tuning, yields a significant performance boost.*

위의 두 가지 아이디어는 실제 논문에서 다음과 같이 반영됩니다. ***어떤 Region proposal algorithm이 주는 결과인 region에 대해서 classification 문제에 대해 학습된 CNN을 적용하여 feature를 추출합니다.*** (CNN을 feature extractor로 사용)

## 1. Introduction
본 논문의 Introduction에서는 다음과 같은 내용을 이야기 합니다.
-  Visual recognition 분야에서 CNN 이전에 SIFT 또는 HOG로 feature를 추출하고, classification 문제를 푸는 방식은 2010 ~ 2012 동안 발전속도가 매우 느렸다.
-  조금씩 이루어진 성능 향상은 Ensemble을 이용하거나, 과거의 성공적이었던 방법들의 minor variants에 불과하다.

이후 2012년에 ***Alex Krizhevsky*** 가 CNN 기반의 AlexNet으로 ILSVRC (ImageNet Large Scale Visual Recognition Challenge)에서 Visual recognition에서 큰 성능 향상을 달성하였고, 이후 CNN의 classification result (for ImageNet)를 어떻게 Object detection (for PASCAL VOC Challenge)으로 확장할 수 있을 지를 연구자들이 논의하였으며, 본 논문에서 아래의 문제 상황을 두 가지 방법으로 적절히 대응함으로써 문제를 풀어냅니다.

문제상황 : ***Localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data***

- image에 대해서 ***Region proposal algorithm*** 이 주는 각각의 region에 CNN을 적용하여 ***fixed-length feature vector*** 를 추출하고, 이에 ***Linear svm (class specific, classification model)*** 을 적용하여 classification을 한다. 이 때 localization의 경우는 마찬가지로 ***fixed-length feature vector*** 를 input으로 하는 ***Bounding-box regression (class specifitc, regression model)*** 를 구성하여 localization을 한다.
	- ***class specific*** 은 class 마다 ***Linear svm*** 과 ***Bounding-box regression*** 이 존재한다는 의미
	- ***Bounding-box regression*** 의 경우는 사용해도되고 사용하지않아도 무관, 단 사용하는 편이 localization 성능이 좋음

- 위에서 사용된 CNN은 ILSVRC 데이터 중 Visual recognition을 위한 데이터에 대해서 ***supervised pre-training*** 이 되어있고, PASCAL VOC의 Object detection를 위한 데이터에 ***fine-tuning*** 을 합니다.
	- ***fine-tuning*** 결과 ***fine-tuning*** 을 안했을 경우에 비해, 8 퍼센트 포인트 정도 성능 향상이 있다고 함


<p align="center"><img src = 'https://i.imgur.com/lEgrJ1o.png' width = '500'></p>


## 2. Object detection with R-CNN
### 2.1 Module design
R-CNN은 세 가지 모듈로 구성되어 있으며, 각각의 모듈 별 역할은 아래와 같습니다.
- **Region proposal module** : class에 무관하게 image에서 region을 추출하는 모듈
    - R-CNN은  ***Region proposal algorithm*** 중 하나인 ***selective-search*** 를 활용 (이전 논문과의 비교를 위해)
    - ***class-agnostic*** 하다면 어떤 ***Region proposal algorithm*** 이라도 상관 x

- **Convolutional Neural Network** : ***Region proposal module*** 이 생성한 region에서 ***fixed-length feature vector*** 를 추출하는 모듈
    - CNN (eg. AlexNet, VggNet 등)은 일정한 크기의 input을 받으므로, ***Region proposal algorithm*** 이 image에 대해서 주는 다양한 rectangle의 region을 크기나 종횡비에 상관없이 warping하여 일정한 크기에 맞춤
    - warping된 region을 CNN (eg. AlexNet, VggNet 등)에 forwarding하여 classification layer 이전의 layer의 결과를 활용

	    - AlexNet : fc<sub>2</sub> (4096 dimensional vector)를 ***fixed-length feature vector*** 로 활용
	    - VggNet (16) : pool<sub>5</sub> 에 해당하는 featuremap을 flattening하여 ***fixed-length feature vector*** 로 활용
	    - Training 시 CNN이 extract한 ***fixed-length feature vector*** 를 저장해놓고, 이를 이용하여 ***Linear svm*** 을 학습함

- **Linear svm module** : ***fixed-length feature vector*** 를 input으로 받아 classification을 하는 모듈
- **Bouding-box regression module** : ***fixed-length feature vector*** 를 input으로 받아 box를 표현하는 네 가지의 숫자 (x, y, w, h)를 계산하는 모듈

### 2.2 Test-time detection
Inference 단계에서 R-CNN은 다음의 과정으로 동작합니다.

1. ***Region proposal algorithm (eg. selective search)*** 로 2,000개 가량의 region 생성
2. 각각의 region을 input으로 CNN이 ***fixed-length feature vector*** 들을 추출
3. 각각의 ***fixed-length feature vector*** 를 input으로 각 class 별 ***Linear svm*** 이 score를 계산 
4. non-maximum suppresion을 이용, class score가 높은 region과 IoU (Intersection over Union)가 정해준 threshold 보다 큰 region들을 제거
- ***Boundig-box regression*** 을 하는 경우, 3과 4사이에서 이루어지며, 이 때 ***fixed-length feature vector*** 를 input으로 bouding box의 (x, y, w, h)를 계산


위의 Inference process를 보면 ***Region proposal algorithm*** 이 생성한 모든 region에 동일한 CNN이 적용되므로, 각 class에 해당하는 object를 detect하는 문제에서 모든 CNN parameter들을 공유합니다. 또한 이전의 다른 방법에 비해서 low-dimensional feature vector에 대해서 object detection이 이루어지므로, 속도가 빠릅니다.

### 2.3 Training
#### Supervised pre-training
위 절에서 언급했던대로 large auxiliary dataset (ILSVRC 2012 classificaion)에 대해서 pre-training된 CNN을 활용합니다.

#### Domain-specific fine-tuning
large auxiliary dataset (ILSVRC 2012 classificaion)에 pre-training된 CNN을 Object detection task에 맞게 변형하기위해서 기존에 pre-traning된 CNN에서 classification layer (eg. output layer)를 새롭게 Object detection을 위한 PASCAL VOC 데이터에 맞게 ***"object class의 개수 + background"*** 로 바꾸고, 해당 부분만 weight initialization을 합니다. SGD(Stochastic Gradient Descent)를 이용하여 Training 시 mini-batch를 구성하는 방식은 아래와 같습니다.

- **positive sample** : ***Region proposal algorithm*** 이 생성한 region들 중, 어떠한 class의 object이든지 간에 그 object의 ground-truth box와 IoU 값이 0.5 이상인 region
- **negative sample** : positive sample을 제외한 나머지 region
- **mini-batch** : 매 iteration 마다 전 class의 postive sample에서 32개, negative sample에서 96개를 sampling하여 mini-batch를 구성

#### Object category classifier
class 별 ***Linear svm*** 을 Training하기 위해서, class 별로 postive sample과 negative sample 선정합니다. positive sample과 negative sample을 선정하는 방식은 아래와 같습니다.

- **positive sample** : class 별 object의 ground-truth box
- **negative sample** : ***Region proposal algorithm*** 이 생성한 region들 중, class 별 object의 ground-truth box와 IoU 값이 0.3 미만인 region

위와 같이 positive sample과 negative sample이 학습하고자하는 class 별 ***Linear svm*** 에 대해서 구성되면, fine-tuned CNN을 이용하여 ***fixed-length feature vector*** 를 추출하고, 이를 input으로 class 별 ***Linear svm*** 을 학습합니다.

### 2.4 Results on PASCAL VOC 2010-2012

<p align="center"><img src = 'https://i.imgur.com/IbLaL4H.png'></p>

Table 1을 통해서 확인해보면 본 논문에서 제안하고 있는 R-CNN이 같은 ***Region proposal algorithm*** (eg. selective search)이 주는 region을 활용하는 방법론인 UVA, Regionlets 등의 방법론보다 훨씬 좋은 성능을 보이는 것을 확인할 수 있으며, ***Bounding-box regression*** 을 활용할 때, 성능이 더 올라간다는 것을 확인할 수 있습니다.
### 2.5 Results on ILSVRC2013 detection

<p align="center"><img src = 'https://i.imgur.com/imo9fwy.png'></p>

Figure 3을 보면 ILSVRC의 경우 PASCAL VOC보다 분류해야할 class의 수가 많아 ***mAP*** 성능이 낮은 것을 확인할 수 있지만, 논문에서 제안하는 R-CNN이 기존의 다른 방법론 보다 성능이 좋음을 확인할 수가 있습니다.
## 3. Visualization, ablation, and modes of error
### 3.1 Visualizing learend features
본 논문에서는 R-CNN의 구조로 학습된 CNN이 어떤 것을 학습하였는 지 확인하는 방법으로 다음과 같은 방법을 제안합니다. 본문에서는 다음과 같이 표현되어 있습니다.

> *We propose a simple (and complementary) non-parametric method that directly shows what the network learned. The idea is to single out a particular unit (feature) in the network and use if as if it were an object detector in its own right.*

> *Our method lets the selected unit "speak for itself" by showing eactly which inputs it fires on.*

위의 내용을 CNN으로 VggNet (16)을 활용했을 때로 예를 들자면, pool<sub>5</sub> 의 featuremap을 flattening한 ***fixed-length feature vector*** 에 대하여 특정 element 기준으로 내림차순을 하고, 특정 element의 값이 높은 region을 기준으로 non-maximum suppresion를 이용하여 region들을 정리해보는 것입니다. 표시해보면 아래와 같습니다.

<p align="center"><img src = 'https://i.imgur.com/rhzzobn.png'></p>

### 3.2 Ablation studies

<p align="center"><img src = 'https://i.imgur.com/KoiPTWM.png'></p>

#### Performance layer-by-layer without fine-tuning
위의 Table 2에서 볼 수 있듯이 R-CNN pool<sub>5</sub> 의 경우가 dense layer인 R-CNN fc<sub>6</sub> , R-CNN fc<sub>7</sub> 의 output을 feature로 썼을 때에 비해서, ***mAP*** 가 크게 차이나지 않고 상당히 높은 것을 보아, CNN에서 representational power는 dense layer보다 convolutional layer에 기인함을 알 수가 있습니다.

#### Performance layer-by-layer with fine-tuning
위의 Table 2에서 볼 수 있듯이  ***fine-tuning*** 을 하면 R-CNN fc<sub>6</sub> , R-CNN fc<sub>7</sub> 의 경우가 R-CNN FT pool<sub>5</sub> 보다 ***mAP*** 가 높은 데, 이는 pool<sub>5</sub> 는 task-specific이 아닌 image 자체에 적용되는 General feature들을 학습한 결과이기 때문이라고 볼 수 있습니다.

### 3.3 Network architectures

<p align='center'><img src = 'https://i.imgur.com/wbjkNse.png'></p>

R-CNN의 성능에는 Network architecture도 중요한 요소이며, R-CNN의 구조를 VggNet (16)에 적용하여 Object detection을 수행한 결과가 더 좋은 이유는 Network 자체의 복잡도가 VggNet (16)이 AlexNet보다 커서 representational power는 좋은 것에 기인하며, Network 복잡도가 큰 만큼 계산이 오래걸립니다.

### 3.4 Robustness of R-CNN

<p align="center"><img src = 'https://i.imgur.com/p2MjqoI.png'></p>

Figure 6을 통해서 Object detection을 하고자하는 object에 occlusion (occ), truncation (trn), bounding-box area (size), aspect ratio (asp), viewpoint (view), part visibility (part) 등의 문제점이 존재할 때, R-CNN의 Object detection 성능을 보면 이전의 방법론인 Deformable Part Model (DPM)에 비해서 그 성능이 훨씬 좋음을 확인할 수 있습니다. 또한 ***fine-tuning*** 과 ***Bounding-box regression*** 을 활용했을 경우에 위와 같은 상황들에서 성능을 더 끌어올릴 수 있음을 확인할 수 있습니다.


## 4. Conclusion
본 논문에서는 R-CNN의 좋은 성능이 다음의 두 가지에 기인하여 나타난 것이라고 이야기합니다.

- ***Object detection*** 을 하기위해서 Bottom up인 ***Region proposal algorithm*** 의 결과에 representational power가 좋은 CNN을 적용
- ***"Supervised pre-training (for auxilary task) and domain-specific fine-tuning"*** paradigm