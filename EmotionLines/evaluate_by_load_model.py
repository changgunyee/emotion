from EmotionLines.constants import *
from EmotionLines.utils import *
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.preprocessing.text import Tokenizer
from EmotionLines.preprocessor import preprocess,model_tokenizer,cal_max_length


model = keras.models.load_model('checkpoint-1.379.h5')

utters_train,labels_train=load_dialogues(DATA_MAP['train'])
utters_dev,labels_dev=load_dialogues(DATA_MAP['dev'])
utters_test,labels_test=load_dialogues(DATA_MAP['test'])
utters_train+=utters_dev
labels_train+=labels_dev

utters=utters_train + utters_test
max_seq_word_length = cal_max_length(utters)
tokenizer = model_tokenizer(utters)
vocab_size = len(tokenizer.word_index) + 1
x_train, y_train, x_test, y_test,max_seq_len = preprocess(utters_train, labels_train, utters_test, labels_test, tokenizer)

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


x_submission=load_csv_dialogues(DATA_MAP['submission_input'])
x_submission=tokenizer.texts_to_sequences(x_submission)
x_submission= pad_sequences(x_submission, padding='post', maxlen=max_seq_len)

predictions=model.predict(x_submission)
classe_idxs=np.argmax(predictions,axis=1)
y_submission=map_idx_to_label(classe_idxs)
output_to_csv(DATA_MAP['submission_output'],y_submission)
