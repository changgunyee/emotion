from constants import *
from utils import *
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from preprocessor import *
from utils import *

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath,'r',encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


utters_train, labels_train = load_dialogues(DATA_MAP['train'])
utters_dev, labels_dev = load_dialogues(DATA_MAP['dev'])
utters_test, labels_test = load_dialogues(DATA_MAP['test'])
utters_train += utters_dev
labels_train += labels_dev

utters = utters_train + utters_test
max_seq_word_length = cal_max_length(utters)
tokenizer = model_tokenizer(utters)
vocab_size = len(tokenizer.word_index) + 1
x_train, y_train, x_test, y_test, max_seq_len = preprocess(utters_train, labels_train, utters_test, labels_test,
                                                           tokenizer)

input_dim=x_train.shape[1]
model=Sequential()
embedding_dim = 300
embedding_matrix = create_embedding_matrix(
    './glove.42B.300d.txt',
    tokenizer.word_index, embedding_dim)

seq_input = layers.Input(shape=(max_seq_len,), dtype='int32')
seq_embedded=layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_seq_len,
                           trainable=True)(seq_input)


filters=[2,3,4,5]
conv_models=[]
for filter in filters:
    conv_feat = layers.Conv1D(filters=128,
                            kernel_size=filter,
                            activation='relu',
                            padding='valid')(seq_embedded) #Convolution Layer
    pooled_feat = layers.GlobalMaxPooling1D()(conv_feat) #MaxPooling
    conv_models.append(pooled_feat)

conv_merged = layers.concatenate(conv_models, axis=1) #filter size가 2,3,4,5인 결과들 Concatenation
model_output = layers.Dropout(0.2)(conv_merged)
# model_output = layers.Dense(10, activation='relu')(model_output)
# logits = layers.Dense(1, activation='sigmoid')(model_output)
model_output=layers.Dense(10,input_dim=input_dim,activation='relu')(model_output)
logits=layers.Dense(8,activation='softmax')(model_output)
model = Model(seq_input, logits) #(입력,출력)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,
                  epochs=40,
                  validation_data=(x_test,y_test),
                  batch_size=128)
plot_history(history)
model.save('my_model.h5')

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


x_submission=load_csv_dialogues(DATA_MAP['submission_input'])
x_submission=tokenizer.texts_to_sequences(x_submission)
x_submission= pad_sequences(x_submission, padding='post', maxlen=max_seq_len)

predictions=model.predict(x_submission)
classe_idxs=np.argmax(predictions,axis=1)
y_submission=map_idx_to_label(classe_idxs)
output_to_csv(DATA_MAP['submission_output'],y_submission)