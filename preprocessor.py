from keras import backend
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import map_label_to_idx
import numpy as np

def set_core_to_use(num):
    config = tf.ConfigProto(device_count={"CPU": num})
    backend.tensorflow_backend.set_session(tf.Session(config=config))


def cal_max_length(utters):
    return max([len(utter.split()) for utter in utters])

def model_tokenizer(texts):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def preprocess(utters_train,labels_train,utters_test,labels_test,tokenizer):
    x_train=tokenizer.texts_to_sequences(utters_train)
    x_test=tokenizer.texts_to_sequences(utters_test)
    max_seq_len=max([len(x) for x in x_train])
    x_train = pad_sequences(x_train, padding='post', maxlen=max_seq_len)
    x_test= pad_sequences(x_test, padding='post', maxlen=max_seq_len)

    y_train=map_label_to_idx(np.array(labels_train))
    y_test=map_label_to_idx(np.array(labels_test))
    return x_train,y_train,x_test,y_test,max_seq_len