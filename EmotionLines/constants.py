import numpy as np

LABELS=np.array(['anger','disgust','fear','joy','neutral','non-neutral','sadness','surprise'])

DATA_MAP={
    'submission_input':'./EmotionLines/Friends/en_data.csv',
    'submission_output':'./EmotionLines/Friends/en_sample.csv',
    'train':'./EmotionLines/Friends/friends_train.json',
    'dev':'./EmotionLines/Friends/friends_dev.json',
    'test':'./EmotionLines/Friends/friends_test.json'
}
