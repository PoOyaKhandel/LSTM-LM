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


def write_file(text, fname):
    # print(text)
    file = open(fname, 'w')
    file.write(text)
    file.close()


class ProcessCorpus:
    """
    """
    def __init__(self, fname):
        self.nlp_pipeline = stanfordnlp.Pipeline(processors='tokenize', lang='fa')
        all_text = self.__load_doc(fname)
        self.content = self.__remove_extra_info(all_text)

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
        return final_tokens[0:5]

    @staticmethod
    def __remove_chars(sentence):
        return re.sub(
            '[^ \u0622\u0627\u0628\u067E\u062A-\u062C\u0686\u062D-\u0632\u0698\u0633'
            '-\u063A\u0641\u0642\u06A9\u06AF\u0644-\u0648\u06CC\u200c]',
            "", sentence)

    def __sentence_processing(self, content):
        doc = self.nlp_pipeline(content)
        list_of_sent_after_removing_chars = []
        for sent in doc.sentences:
            sent_text = [a_word.text for a_word in sent.words]
            sent_after_removing_chars = self.__remove_chars(sent_text)
            if not sent_after_removing_chars.isspace():
                sent_after_removing_chars = " ".join(sent_after_removing_chars.split())
                list_of_sent_after_removing_chars.append(sent_after_removing_chars)
        if len(list_of_sent_after_removing_chars) > 0:
            return '\n'.join(list_of_sent_after_removing_chars)
        else:
            return None

    def content_processing(self):
        list_of_all_clen_contents = []
        for content in self.content:
            # content = self.__remove_chars(content)
            clean_content = self.__sentence_processing(content)
            if clean_content is not None:
                list_of_all_clen_contents.append(clean_content)
        write_file('\n'.join(list_of_all_clen_contents), "clean_corpus.txt")


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
    cleanning_unit = ProcessCorpus(fname)
    # write_file(cleanning_unit.content, 'useful_content.txt')
    cleanning_unit.content_processing()




