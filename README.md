# Hangeul Font Generator with GAN
GAN(Generative Adversarial Nets) 모델을 활용한 새로운 Font Generation 프로젝트
   - 팀원: 박주현, 윤지영, 이종호
   - 프로젝트 기간: 2023.11.13. ~ 2023.11.27.
   - 개발환경: 
  
## Project Introduction
   - 다양한 Font 스타일이 존재하지만, Font를 만드는 일은 많은 시간과 비용이 드는 전문적인 작업임
   - 이에 따라서, GAN model을 활용해서 여러가지 스타일의 폰트를 학습하고, 학습한 여러가지 폰트 스타일을 바탕으로 새로운 스타일의 Font를 생성하는 과정을 진행하기로 결정


## Research and Analysis

### GAN(Generative Adversarial Nets)
생성자(Generator)와 판별자(Discriminator)가 서로 경쟁을 하면서 실제에 가까운 이미지나 사람이 쓴 것과 같은 글 등, 여러 가지 fake data들을 생성한다. 두 네트워크를 적대적으로 학습시키는 비지도 학습 기반의 생성모델(Unsupervised Generative model)이다.

Generative Adversarial Nets 이라는 이름은 실제 데이터의 분포와 유사한 분포를 추정하기 위해서 Generator, Discriminator 두 모델을 적대적(Adversarial) 방식을 통해 모델을 training시키기 때문에 붙여진 이름이다.

GAN Model의 최종적인 목적은 training data와 비교했을 때 구분할 수 없을 정도로 유사한 fake data를 생성할 수 있도록 training data의 분포를 추정하는 fake data의 분포를 찾는 것이다.

<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/291fb607-cee8-49c4-9f9e-4fa48d135526" width="600"></p>


**[Generator]**
Generator의 역할은 Discriminator가 real과 fake를 구별할 수 없을 만큼 진짜같은 fake data를 만들어내는 것이다.

Noise vector 'z'를 표준정규분포로부터 샘플링한 후에, 'z'를 input으로 넣어서 fake data를 만든다. 가짜이지만 진짜같은 데이터를 만들어 내는 것이 목표이기 때문에 discriminator에 만든 fake data를 넣었을 때 높은 확률을 반환하는 방향으로 weight를 업데이트시키면서 학습한다.


**[Discriminator]**
Discriminator의 역할은 주어진 input data가 real data인지 fake data인지를 구별하는 것이다. input data가 주어졌을 때 discriminator의 output은 input data가 real data일 확률을 반환한다.

진짜데이터와 가짜데이터를 판별하는 것이 목적이기 때문에, Generator는 고정시켜두고 real data가 들어왔을 때는 높은 확률을 반환하고, fake data가 들어왔을 때는 낮은 확률을 반환하는 방향으로 weight를 업데이트하는 방향으로 discriminator를 학습시킨다.


GAN은 위와 같이 Generator & Discriminator를 번갈아 학습시키면서 Generator는 Discriminator가 판별할 수 없을 만큼 가짜 데이터를 잘 만들어 낼 수 있도록 만들고, Discriminator는 Generator가 진짜같은 가짜데이터를 만들어내더라도 잘 판별할 수 있도록 만들면서 균형점을 찾아간다.


## Network Structure

### Model Structure
모델구조사진






## Training Codes
```
common
├── dataset.py    # load dataset
├── function.py   # deep learning functions : conv2d, relu etc.
├── models.py     # Generator(Encoder, Decoder), Discriminator
├── train.py      # model Trainer
└── utils.py      # data pre-processing etc.

get_data
├── font2img.py   # font.ttf -> image
└── package.py    # .png -> .pkl
```



## Result
