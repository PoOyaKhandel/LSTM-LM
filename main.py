import prerpocess_corpus
from numpy import array
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from pickle import dump


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
# define model
model = Sequential()
model.add(Embedding(sent_prepare.vocab_size+1, 300, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(sent_prepare.vocab_size+1, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(train_x, train_y, validation_split=0.1, batch_size=128, epochs=100)
# test
print('\n# Evaluate on test data')
results = model.evaluate(test_x, test_y)
print('test loss, test acc:', results)
#
# # save the model to file
# model.save('model.h5')
# # save the tokenizer
# dump(tokenizer, open('tokenizer.pkl', 'wb'))

