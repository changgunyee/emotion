import json
import re
import csv
import matplotlib.pyplot as plt
from EmotionLines.constants import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_dialogues(filename):
    utterances= []
    labels = []
    with open(filename,'r',encoding='UTF-8') as f:
        dialogues = json.loads(f.read())
        for dialogue in dialogues:
            for line in dialogue:
                utterances.append(preprocessing(line['utterance']))
                labels.append(line['emotion'])

    return utterances, labels


def load_csv_dialogues(filename):
    utterances=[]
    with open(filename,encoding='CP949') as f:
        reader=csv.reader(f)
        for row in reader:
            utterances.append(preprocessing(row[4]))

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

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


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
