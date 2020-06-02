from constants import *
from utils import *
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.preprocessing.text import Tokenizer

model = keras.models.load_model('checkpoint-1.457.h5')

utters_train,labels_train=load_dialogues(DATA_MAP['train'])
utters_dev,labels_dev=load_dialogues(DATA_MAP['dev'])
utters_test,labels_test=load_dialogues(DATA_MAP['test'])
utters_train+=utters_dev
labels_train+=labels_dev

utters=utters_train+utters_test
max_word_length= max([len(utter.split()) for utter in utters])
tokenizer=Tokenizer(num_words=max_word_length)
tokenizer.fit_on_texts(utters)
vocab_size = len(tokenizer.word_index) + 1

x_train=tokenizer.texts_to_sequences(utters_train)
x_test=tokenizer.texts_to_sequences(utters_test)

x_train = pad_sequences(x_train, padding='post', maxlen=max_word_length)
x_test= pad_sequences(x_test, padding='post', maxlen=max_word_length)


y_train=map_label_to_idx(np.array(labels_train))
y_test=map_label_to_idx(np.array(labels_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


x_submission=load_csv_dialogues(DATA_MAP['submission_input'])
x_submission=tokenizer.texts_to_sequences(x_submission)
x_submission= pad_sequences(x_submission, padding='post', maxlen=max_word_length)

predictions=model.predict(x_submission)
classe_idxs=np.argmax(predictions,axis=1)
y_submission=map_idx_to_label(classe_idxs)
output_to_csv(DATA_MAP['submission_output'],y_submission)
