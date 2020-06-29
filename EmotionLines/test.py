import pytest
from EmotionLines.preprocessor import *

utters_train=["Oh my God, he\u0092s lost it. He\u0092s totally lost it.",
              "What?",
              "Or! Or, we could go to the bank, close our accounts and cut them off at the source."]
labels_train=["non-neutral","surprise","neutral"]
utters_test=["Why do all you\u0092re coffee mugs have numbers on the bottom?",
            "Oh. That\u0092s so Monica can keep track. That way if one on them is missing, she can be like, \u0091Where\u0092s number 27?!\u0092"]
labels_test=[ "surprise","non-neutral"]
utters = utters_train + utters_test


def test_cal_max_length():
    max_seq_word_length = cal_max_length(utters)
    assert max_seq_word_length==22

def test_preorcess():
    tokenizer = model_tokenizer(utters)
    vocab_size = len(tokenizer.word_index) + 1
    x_train, y_train, x_test, y_test, max_seq_len = preprocess(utters_train, labels_train, utters_test, labels_test,
                                                               tokenizer)
    print(x_train,y_train,x_test,y_test,max_seq_len)
