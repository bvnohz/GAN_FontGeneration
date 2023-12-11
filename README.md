# Hangeul Font Generator with GAN
GAN(Generative Adversarial Nets) 모델을 활용한 새로운 Font Generation 프로젝트
   - 팀원: 박주현, 윤지영, 이종호
   - 프로젝트 기간: 2023.11.13. ~ 2023.11.27.
   - 개발환경:  

     
  
## Project Introduction
   - 다양한 Font 스타일이 존재하지만, Font를 만드는 일은 많은 시간과 비용이 드는 전문적인 작업임
   - 이에 따라서, GAN model을 활용해서 여러가지 스타일의 폰트를 학습하고, 학습한 여러가지 폰트 스타일을 바탕으로 새로운 스타일의 Font를 생성하는 과정을 진행하기로 결정
   - 각각의 폰트 스타일의 특징을 잘 표현하는 벡터로 구성된 분포 공간인 Latent Space(Feature Space)를 파악하고, 해당 공간 사이에서 새로운 feature를 발견해서 새로운 스타일의 font를 만들어내기로 함

 

## Research and Analysis
### GAN(Generative Adversarial Nets)
생성자(Generator)와 판별자(Discriminator)가 서로 경쟁을 하면서 실제에 가까운 이미지나 사람이 쓴 것과 같은 글 등, 여러 가지 fake data들을 생성한다. 두 네트워크를 적대적으로 학습시키는 비지도 학습 기반의 생성모델(Unsupervised Generative model)이다.

Generative Adversarial Nets 이라는 이름은 실제 데이터의 분포와 유사한 분포를 추정하기 위해서 Generator, Discriminator 두 모델을 적대적(Adversarial) 방식을 통해 모델을 training시키기 때문에 붙여진 이름이다.

GAN Model의 최종적인 목적은 training data와 비교했을 때 구분할 수 없을 정도로 유사한 fake data를 생성할 수 있도록 training data의 분포를 추정하는 fake data의 분포를 찾는 것이다.

<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/291fb607-cee8-49c4-9f9e-4fa48d135526" width="600"></p>


### \# Generator
Generator의 역할은 Discriminator가 real과 fake를 구별할 수 없을 만큼 진짜같은 fake data를 만들어내는 것이다. Noise vector 'z'를 표준정규분포로부터 샘플링한 후에, 'z'를 input으로 넣어서 fake data를 만든다. 가짜이지만 진짜같은 데이터를 만들어 내는 것이 목표이기 때문에 discriminator에 만든 fake data를 넣었을 때 높은 확률을 반환하는 방향으로 weight를 업데이트시키면서 학습한다.


### \# Discriminator
Discriminator의 역할은 주어진 input data가 real data인지 fake data인지를 구별하는 것이다. input data가 주어졌을 때 discriminator의 output은 input data가 real data일 확률을 반환한다. 진짜데이터와 가짜데이터를 판별하는 것이 목적이기 때문에, Generator는 고정시켜두고 real data가 들어왔을 때는 높은 확률을 반환하고, fake data가 들어왔을 때는 낮은 확률을 반환하는 방향으로 weight를 업데이트하는 방향으로 discriminator를 학습시킨다.


GAN은 위와 같이 Generator & Discriminator를 번갈아 학습시키면서 Generator는 Discriminator가 판별할 수 없을 만큼 가짜 데이터를 잘 만들어 낼 수 있도록 만들고, Discriminator는 Generator가 진짜같은 가짜데이터를 만들어내더라도 잘 판별할 수 있도록 만들면서 균형점을 찾아간다.


## Network Structure
### Model Structure
<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/0930a018-4ffb-44ef-b580-e4a94d602286" width="600"></p>


### \# Generator
이미지를 low dimensional vector로 mapping시키는 encoder와 다시 이미지로 복원시키는 decoder, 두 부분으로 구성된다.
Generator에서는 source font인 고딕체글 새로운 스타일의 font로 바꾸는 style transfer가 이루어진다. 따라서 모델에 변환하기를 원하는 스타일의 폰트의 카테고리(category vector; c vector)에 대해서 입력해 주어야 한다.
c-vector는 encoder를 통과해서 low dimensional vector로써 mapping된 z-vector에 더해져서 decoder에 입력으로 들어간다. 따라서 어떤 font style을 입력하느냐에 따라서 원하는 style의 font로 변환할 수 있다.


### \# Discriminator
학습에서 이용된 이미지인지(real data, source image) Generator에서 생성된 이미지인지(fake data) 판단하는 역할을 한다.
해당 data가 특정한 종류의 category에 해당하는 font가 맞는지도 함께 판단한다.
이렇게 두 가지 판단을 하는 discriminator를 만들기 위해서 FC Layer를 2개로 독립적으로 만들어서 2개의 output이 나올 수 있도록 model architecture를 구성한다.


## Losses for training
GAN Model은 generator, discriminator가 서로 겨루면서 학습하는 generative model이기 때문에 학습에 있어서 여러가지의 불안정성 문제가 발생한다. 따라서 잘 설계된 loss function을 필요로 한다.


### \# Loss for Generator
**Target similarity Loss(L1 Loss)** : generator의 가장 기본 목표를 위한 것으로, real data와 fake data를 pixel 단위로 비교하기 위해서 MAE(Mean Absolute Error)를 사용한다. MAE가 크다면 두 이미지가 다르다는 의미이기 때문에 generator는 이 loss를 줄이는 방향으로 학습된다.


**Z-vector similarity Loss(Constant Loss)** : encoder를 통과한 뒤 만들어지는 source image의 latent space에서의 위치(z-vector)와 generator에서 생성된 fake image의 위치(z-vector)를 비교해서 loss를 계산한다. Constant loss는 두 z-vector가 비슷하게 유지될 수 있도록 제어하는 역할을 수행한다.


**Category Loss**: discriminator가 올바른 font category로 판단할 수 있을 만큼 font style(feature)이 잘 반영된 image data를 생성할 수 있도록 학습되기 위한 loss이다.


### \# Loss for Discriminator
**Binary Loss** : 입력받은 image data가 fake인지 real인지 구분하는 loss이며 예측 정도를 T/F 예측을 0~1 사이의 값으로 출력한다. true or false 두가지 category를 비교하게 되므로 이진분류에 쓰이는 Binary Cross Entropy(BCE)를 사용한다.


**Category Loss** : font category를 올바르게 예측할 수 있도록 하는 loss, 이는 generator가 font style의 특징을 제대로 담아서 생성할 수 있도록 돕는 역할을 하며, cross entropy를 이용한다.



## Training Codes
```
training
├── execution.py  # load dataset, visualizing
├── models.py     # Generator(Encoder, Decoder), Discriminator, deep learning functions(conv2d, relu etc.)
└── train.py      # model Trainer

get_data
├── dataload.py   # font.ttf -> image.png -> .pkl
└── utils.py      # data pre-processing etc.
```


## Requirement
```
  python / 3.10.12 (colab)
  torch / 2.1.0
  cv2
  matplotlib
  numpy
  PIL / 9.4.0
```


## Result
학습시킨 모델을 이용해서 새로운 두 가지의 font style을 입력으로 넣었을 때 그 결과로 두 폰트 사이에서 style transfer가 매끄럽게 발생하고 있는 것을 확인할 수 있다.
<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/43aa2aa0-590f-4a8d-b82e-995f6a50ae7f" width="450"></p>

위 결과를 이용해서 사용자는 웹페이지에서 주어진 font style 이미지 중에서 원하는 폰트 스타일을 2가지 선택하면, 두 가지 폰트 스타일의 latent space에서 scrolling을 통해서 새로운 font style을 생성하여 보여준다.
<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/df83c83e-5a47-43a4-b8b8-0eeecc0de2fe" width="700"></p>
