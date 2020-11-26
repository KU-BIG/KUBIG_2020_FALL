1. Covid-19 Face Mask Detection (단기, 조장: 구형석)
: 사진 혹은 영상에서 해당 사람이 마스크를 썼는지 아닌지 탐지하는 모델을 만드는 프로젝트로, 데이터셋을 만드는 과정부터 딥러닝 모델 제작까지 하나의 사이클을 작게나마 경험할 수 있는 프로젝트입니다!
* 딥러닝을 아직 배우지 않으신 분들도, 이번 프로젝트를 통해 딥러닝을 미리 체험해보시면서 앞으로 관련 분야를 더 공부해볼지말지 결정할 수 있는 좋은 기회로 생각해주셨으면 좋겠습니다
* 해당 프로젝트는 11월 완성을 목표로 하고, 마음이 잘 맞으시는 분들과는 겨울방학 때 또 다른 새로운 프로젝트들을 해볼 생각을 가지고 있습니다
* 해당 프로젝트들에서는 다음과 같은 개념을 학습 혹은 체험할 수 있습니다
1) 딥러닝 기초
2) 이미지 인식 모델의 기초: cnn
3) Object Detection의 기초, 적어도 원리라도
4) Face Detection, Face Landmark Detection
5) Transfer Learning
6) 사진과 동영상 모두에 적용 가능한 모델 만들기
7) 이미지 관련 딥러닝 프로젝트의 대략적인 흐름
</br>

<파일 소개>
(1) mask_classifier.py
* tensorflow 2.3.0
* mask_classifier model을 만들어주는 코드입니다  
* Dataset 정보는 코드 내에 주석을 참고해주세요  
* Dataset 경로만 잘 설정해 주시면 model 파일을 생성할 수 있습니다  
(models 폴더 내 사전 제작된 model 파일이 있습니다)  

(2) models 폴더
* face detector: opencv face detector(SSD와 Resnet-10 기반)
* model: mask classifier model

(3) video.py</br>
* tensorflow 2.3.0
* face detector, mask classifier 모델을 불러와서 비디오 혹은 실시간 영상에 대해 mask detection을 수행합니다.
