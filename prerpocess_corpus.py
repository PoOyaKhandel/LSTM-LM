"""
This module will pre-process Miras corpus, removes extra information, keeps only content of each sample and also
removes non-persian characters
"""
import stanfordnlp
import re
from keras.preprocessing.text import Tokenizer
import pandas as pd
from random import shuffle
import numpy as np
from multiprocessing import Process, Lock
from time import time


def write_file(text, fname):
    # print(text)
    file = open(fname, 'a')
    file.write(text)
    file.close()


class ProcessCorpus:
    """
    """
    def __init__(self, fname):
        self.nlp_pipeline = stanfordnlp.Pipeline(processors='tokenize', lang='fa')
        all_text = self.__load_doc(fname)
        self.content = self.__remove_extra_info(all_text)
        self.n_total_content = len(self.content)
        self.iteration = 0
        self.list_of_all_clean_contents = []

    @staticmethod
    def __load_doc(filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text

    @staticmethod
    def __remove_extra_info(text):
        lines = text.split("\n")
        final_tokens = []
        for line in lines:
            tokens = line.split('***')
            tokens = tokens[0]
            # removing extra spaces
            tokens = tokens.strip()
            final_tokens.append(tokens)
        # final_tokens = '\n'.join(final_tokens)
        return final_tokens

    @staticmethod
    def __remove_chars(sentence, mode=1):
        if mode == 0:
            return re.sub(
                '[^ \u0622\u0627\u0628\u067E\u062A-\u062C\u0686\u062D-\u0632\u0698\u0633'
                '-\u063A\u0641\u0642\u06A9\u06AF\u0644-\u0648\u06CC\u200c\u060C\u061B\u061F\u002E\u0021]',
                "", sentence)
        else:
            return re.sub(
                '[^ \u0622\u0627\u0628\u067E\u062A-\u062C\u0686\u062D-\u0632\u0698\u0633'
                '-\u063A\u0641\u0642\u06A9\u06AF\u0644-\u0648\u06CC]',
                "", sentence)

    def __make_clean_content_list(self, a_content, lck):
        lck.acquire()
        # print("acontent:", a_content)
        self.iteration += 1
        print(self.iteration, " Contents processed!")
        write_file(a_content, "clean_corpus.txt")
        lck.release()

    def __sentence_processing(self, content, lck):
        content = self.__remove_chars(content, mode=0)
        # print("content:", content)
        doc = self.nlp_pipeline(content)
        list_of_sent_text = []
        for sent in doc.sentences:
            sent_text = [a_word.text for a_word in sent.words]
            sent_text = ' '.join(sent_text)
            sent_text = self.__remove_chars(sent_text)
            if not sent_text.isspace():
                sent_text = " ".join(sent_text.split())
                list_of_sent_text.append(sent_text)
        if len(list_of_sent_text) > 0:
            self.__make_clean_content_list('\n'.join(list_of_sent_text) + '\n', lck)

    def content_processing(self):
        all_process = []
        lock = Lock()
        t1 = time()
        for content in self.content:
            pr = Process(target=self.__sentence_processing, args=(content, lock))
            all_process.append(pr)
            pr.daemon = True
            pr.start()
            if len(all_process) == 4:
                all_process[0].join()
                all_process.remove(all_process[0])

        t2 = time()
        print("Sentence Tokenization Finished After", t2 - t1, "!")
        print("Average time for a content", (t2-t1)/self.n_total_content)
        print("Number of NonPersian Content:", self.n_total_content - self.iteration)


class SentencePreparation:

    def __init__(self, path):
        self.text = self.load_text(path)
        self.vocab_size = None
        self.max_len = None

    @staticmethod
    def load_text(path):
        with open(path, 'r') as ftext:
            text = ftext.read()
            ftext.close()
        return text.split("\n")

    @staticmethod
    def __analyze_sent(sent):
        reviews_len = [len(x) for x in sent]
        pd.Series(reviews_len).hist()
        describe = pd.Series(reviews_len).describe()
        print("Sentences length distribution:\n", describe)
        print("------------------------------")
        return int(describe.values[1] + describe.values[2])

    def tokenize(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.text)
        sequences = tokenizer.texts_to_sequences(self.text)
        self.vocab_size = len(tokenizer.word_index) + 1
        self.max_len = self.__analyze_sent(sequences)
        return sequences

    def __create_input_samples(self, base):
        max_len = self.max_len
        samples = []
        samples_label = []
        for el in base:
            el_len = len(el)
            if el_len > 2:
                for i in range(el_len-1):
                    if i < max_len:
                        new_el = el[:i+1]
                        new_el.extend([0]*(max_len-i-1))
                        samples.append(new_el)
                        samples_label.append(el[i+1])
                    if i >= max_len:
                        new_el = el[i-max_len+1:i+1]
                        samples.append(new_el)
                        samples_label.append(el[i+1])
        return samples, samples_label

    def train_test_split(self, factor, sequences):
        """
        creates train test sets after shuffling reviews(not in place)
        :param factor: split factor for train test sets
        :return: train and test data with their labels
        """
        shuffle(sequences)
        train_x = sequences[:int(factor*len(sequences))]
        test_x = sequences[int(factor*len(sequences))+1:]

        train_x, train_y = self.__create_input_samples(train_x)
        test_x, test_y = self.__create_input_samples(test_x)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y = to_categorical(train_y, num_classes=self.vocab_size+1)

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_y = to_categorical(test_y, num_classes=self.vocab_size+1)

        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    fname = "MirasText_sample.txt"
    write_file("", "clean_corpus.txt")
    cleanning_unit = ProcessCorpus(fname)
    # write_file(cleanning_unit.content, 'useful_content.txt')
    print("useful_content")
    cleanning_unit.content_processing()
    # write_file("Hello\n", "clean_corpus.txt")
    # write_file("Pooya", "clean_corpus.txt")



