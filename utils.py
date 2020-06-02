import numpy as np
import json
import re
import csv
import matplotlib.pyplot as plt
from constants import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# nltk.download('stopwords')

def preprocessing(text):
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]

    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatization
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]

    tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

    # stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def load_dialogues(filename):
    utterances= []
    labels = []
    with open(filename,'r',encoding='UTF-8') as f:
        dialogues = json.loads(f.read())
        for dialogue in dialogues:
            for line in dialogue:
                utterances.append(clean_str(line['utterance']))
                labels.append(line['emotion'])

    return utterances, labels


def load_csv_dialogues(filename):
    utterances=[]
    with open(filename,encoding='CP949') as f:
        reader=csv.reader(f)
        for row in reader:
            utterances.append(clean_str(row[4]))

    return utterances[1:]

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def clean_str(string):
    string = re.sub(r'[^\x00-\x7F]+', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def map_label_to_idx(y:np.array)->np.array:
    return np.array([list(map(int, LABELS == y_)) for y_ in y])


def map_idx_to_label(idxs:np.array)->np.array:
    return np.array([ LABELS[idx]  for idx in idxs])

def output_to_csv(filename,output):
    with open(filename,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Id','Predicted'])
        for i in range(len(output)):
            writer.writerow([i,output[i]])

if __name__ == '__main__':
    output_to_csv(DATA_MAP['submission_output'],["a","b","c"])