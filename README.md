# 실행환경
python3.6에서 진행하였습니다.
```python3
virtualenv venv # 가상환경 설치
source venv/bin/activate # 윈도우 가상환경 실행
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
pip install -r requirements.txt # 패키지 설치

# 필요 패키지 설치 중 torch설치 에러 날 경우(윈도우)
pip install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl #직접 whl로 접근하여 설치
pip install -r requirements.txt #이후 다시 진행
```

# 한국어 감정 분석
## NSMC.py 

> LSTM 기반 기초 모델의 소스코드 입니다. (참고 : https://wikidocs.net/44249)

### 실행방법
1. NSMC/data 폴더에  ratings_train.txt, ratings_test.txt, ko_data.csv, sample.csv 파일을 다운로드 합니다.
2. python3 EmotionLines/NSMC/NSMC.py 명령어로 모델 학습 및 예측을 합니다.
```python3
python3 EmotionLines/NSMC/NSMC.py
```
3. 예측 값이 저장된 NSMC/data/sample.csv 파일을 열어 첫 행에 Id, Predicted를 입력합니다.

## KoBERT.py

> Pre-trained된 KoBERT를 NSMC로 fine-tuning한 모델의 소스코드 입니다. (참고 : https://github.com/SKTBrain/KoBERT)
#### *NVIDIA GPU를 사용하고 CUDA 드라이버가 설치된 환경에서 작동하는 코드입니다.

### 실행방법
1. NSMC/data 폴더에  ratings_train.txt, ratings_test.txt, ko_data_no_index.csv 파일을 다운로드 합니다.
2. python3 EmotionLines/NSMC/KoBERT.py 명령어로 모델 학습 및 예측을 합니다.
```python3
python3 EmotionLines/NSMC/KoBERT.py
```
3. 예측 값이 저장된 NSMC/data/kobert_out.csv 파일을 열어 첫 행에 Id, Predicted를 입력합니다.

# 영어 감정 분석

> Stanford NLP의 pretrained GloVe를 사용합니다 [링크](https://nlp.stanford.edu/projects/glove/)에서 glove.42B.300d.txt를 다운받아 루트 디렉토리에 넣어주세요 

## 실행 방법
### 기본 CNN모델
```python3
python3 EmotionLines/cnn.py
```
### CNN + LSTM 복합 모델
```python3
python3 EmotionLines/cnn_plus_lstm.py
```
