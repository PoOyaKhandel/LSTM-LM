import prerpocess_corpus
from numpy import array
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from pickle import dump


# Pre-process corpus(removing extra files, removing out of range words, ...
source_corpus_path = "MirasText_sample.txt"
clean_corpus_path = "clean_corpus.txt"
source_text = prerpocess_corpus.load_doc(source_corpus_path)
clean_sequences = prerpocess_corpus.clean_doc(source_text)
prerpocess_corpus.save_doc(clean_sequences, clean_corpus_path)

doc = prerpocess_corpus.load_doc(clean_corpus_path)
lines = doc.split("\n")



# Training and Model definition
tokenizer = Tokenizer()
print(len(lines[0]), len(lines[1]), len(lines[2]))

tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(len(sequences[0]), len(sequences[1]), len(sequences[50]))

# separate into input and output
sequences = array(sequences)
# x = pad_sequences(sequences, padding='post')
# print(x)
X, y = sequences[:-1], sequences[-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

