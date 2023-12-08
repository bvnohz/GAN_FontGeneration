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
실제에 가까운 이미지나 사람이 쓴 것과 같은 글 등, 여러 가지 fake data들을 생성하는 model
생성자(Generator)와 판별자(Discriminator)가 서로 경쟁을 하면서 새로운 데이터를 생성한다.

<p align="center"><img src="https://github.com/juooo1117/GAN_Hangeul/assets/95035134/291fb607-cee8-49c4-9f9e-4fa48d135526" width="600"></p>


**[Generator]**


**[Discrinimator]**






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
