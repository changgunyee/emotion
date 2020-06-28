# 한국어 감정 분석
## NSMC.py 

> LSTM 기반 기초 모델의 소스코드 입니다. (참고 : https://wikidocs.net/44249)

### 실행방법
1. NSMC/data 폴더에  ratings_train.txt, ratings_test.txt, ko_data.csv, ko_sample.csv 파일을 다운로드 합니다.
2. python3 NSMC.py 명령어로 모델 학습 및 예측을 합니다.
3. NSMC/data 폴더에 예측 값이 저장된 ko_sample.csv 파일을 열어 첫 행에 Id, Predicted를 입력합니다.

## KoBERT.py

> Pre-trained된 KoBERT를 NSMC로 fine-tuning한 모델의 소스코드 입니다. (참고 : https://github.com/SKTBrain/KoBERT)
#### *NVIDIA GPU를 사용하고 CUDA 드라이버가 설치된 환경에서 작동하는 코드입니다.

### 실행방법
1. NSMC/data 폴더에  ratings_train.txt, ratings_test.txt, ko_data_no_index.csv 파일을 다운로드 합니다.
2. python3 KoBERT.py 명령어로 모델 학습 및 예측을 합니다.
3. 예측 값이 저장된 kobert_out.csv 파일을 열어 첫 행에 Id, Predicted를 입력합니다.
