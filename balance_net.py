from utils import *
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras import optimizers,callbacks
from preprocessor import preprocess,model_tokenizer,cal_max_length

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 bsecause of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath,'r',encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

def create_embedding_layer(seq_input,input_dim,embedding_dim,embedding_mat,max_seq_len,trainable):
    # static channel
    embedding_layer= Embedding(input_dim,
                                embedding_dim,
                                weights=[embedding_mat],
                                input_length=max_seq_len,
                                trainable=trainable)

    embedded_sequences= embedding_layer(seq_input)
    return embedded_sequences

def create_lstm_to_cnn(embedded_sequences_frozen,embedded_sequences_train,conv_props):
    l_lstm1f = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_frozen)
    l_lstm1t = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_train)
    l_lstm1 = Concatenate(axis=1)([l_lstm1f, l_lstm1t])

    convs=create_multiple_kernel_size_conv([l_lstm1],conv_props)
    l_lstm_c = Concatenate(axis=1)(convs)
    return l_lstm_c

def create_cnn_to_lstm(embedded_sequences_frozen,embedded_sequences_train,conv_props):
    convs=create_multiple_kernel_size_conv([embedded_sequences_frozen,embedded_sequences_train],conv_props)
    l_merge_2 = Concatenate(axis=1)(convs)
    l_c_lstm = Bidirectional(LSTM(12,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(l_merge_2)
    return l_c_lstm

def create_multiple_kernel_size_conv(embedded_sequences,conv_props):
    convs = []
    for kernel_size in conv_props['kernel_sizes']:
        for embedded_sequence in embedded_sequences:
            l_conv= Conv1D(conv_props['filters'], kernel_size, activation='relu')(embedded_sequence)
            l_conv = Dropout(0.3)(l_conv)
            convs.append(l_conv)
    return convs


if __name__=='__main__':
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

    model=Sequential()
    embedding_dim = 300
    embedding_matrix = create_embedding_matrix('./glove.42B.300d.txt',tokenizer.word_index, embedding_dim)

    sequence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedded_sequences_frozen = create_embedding_layer(sequence_input, vocab_size, embedding_dim, embedding_matrix, max_seq_len, False)
    embedded_sequences_train = create_embedding_layer(sequence_input, vocab_size, embedding_dim, embedding_matrix, max_seq_len, True)

    conv_props = {'kernel_sizes': [2,3,5,6,8], 'filters': 24}
    l_c_lstm = create_lstm_to_cnn(embedded_sequences_frozen, embedded_sequences_train, conv_props)

    conv_props = {'kernel_sizes': [4, 3, 2], 'filters': 12}
    l_lstm_c=create_cnn_to_lstm(embedded_sequences_frozen,embedded_sequences_train,conv_props)

    l_merge = Concatenate(axis=1)([l_lstm_c, l_c_lstm])
    l_pool = MaxPooling1D(4)(l_merge)
    l_drop = Dropout(0.5)(l_pool)
    l_flat = Flatten()(l_drop)
    l_dense = Dense(26, activation='relu')(l_flat)
    preds = Dense(8, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.002)
    model.compile(loss='categorical_crossentropy',optimizer=adadelta,metrics=['acc'])

    model_checkpoints = callbacks.ModelCheckpoint("checkpoint-{val_loss:.3f}.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=0)

    model.summary()
    model.save('BalanceNet.h5')

    print("Training Progress:")
    model_log = model.fit(x_train, y_train,
                          validation_data=(x_test, y_test),
                          epochs=50,
                          batch_size=128,
                          verbose=True,
                          callbacks=[model_checkpoints])

