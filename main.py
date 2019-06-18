import prerpocess_corpus
import numpy as np
import keras.backend as K
from tensorflow import reshape

def perplexity(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    # print("--------------------")
    # y_true = reshape(y_true, [-1])
    # y_pred = reshape(y_pred, [-1])
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perp = K.exp(cross_entropy)
    return perp


# Pre-process corpus(removing extra files, removing out of range words, ...
CLEAN_CORPUS = False
source_corpus_path = "MirasText_sample.txt"
clean_corpus_path = "clean_corpus.txt"
split_factor = 0.85

if CLEAN_CORPUS:
    pr_corp = prerpocess_corpus.ProcessCorpus(source_corpus_path)
    pr_corp.content_processing()

sent_prepare = prerpocess_corpus.SentencePreparation(clean_corpus_path)
sequences = sent_prepare.tokenize()
train_x, train_y, test_x, test_y = sent_prepare.train_test_split(split_factor, sequences)
############
# # Training and Model definition

seq_length = sent_prepare.max_len

