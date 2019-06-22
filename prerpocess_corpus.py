"""
This module will pre-process Miras corpus, removes extra information, keeps only content of each sample and also
removes non-persian characters
"""
import stanfordnlp
import re
from keras.preprocessing.text import Tokenizer
import pandas as pd
from random import shuffle
from itertools import islice
import numpy as np
from multiprocessing import Process, Lock
from time import time
import os


def write_file(text, fname):
    # print(text)
    file = open(fname, 'a')
    file.write(text)
    file.close()


class ProcessCorpus:
    """
    """
    def __init__(self, path, n_files, n_process):
        self.base_file_path = path
        self.n_files = n_files
        self.n_process = n_process

    def process_first_step(self):
        cwd = os.getcwd()
        step1_output_dir = cwd + '/step1_output/'
        if os.path.exists(step1_output_dir):
            raise NotImplementedError("first step processing exists!")
        else:
            os.mkdir(step1_output_dir)

        output_files = [step1_output_dir + "step" + str(i+1) + "process.txt" for i in range(self.n_files)]
        number_of_all_contents_after_first_step = 0
        dirty_sample = 0
        punctuation_set = set('\u200c\u060C\u061B\u061F\u002E\u0021 ')
        with open(self.base_file_path, mode='r', encoding='utf8') as base_file:
            for line in base_file:
                tokens = line.split('***')
                tokens = tokens[0].strip()
                tokens = self.__remove_chars(tokens, mode='first_step')
                if set(tokens).issubset(punctuation_set):
                    dirty_sample += 1
                else:
                    with open(output_files[number_of_all_contents_after_first_step % self.n_files], 'a') as wr_file:
                        wr_file.write(tokens + '\n')

                number_of_all_contents_after_first_step += 1

                if number_of_all_contents_after_first_step % 100 == 0:
                    print(number_of_all_contents_after_first_step, " sample processed")

        print("Total samples:", number_of_all_contents_after_first_step,
              "Clean samples:", number_of_all_contents_after_first_step - dirty_sample,
              "Dirty samples:", dirty_sample)

    def process_second_step(self):
        cwd = os.getcwd()
        step1_output_dir = cwd + '/step1_output/'
        step2_output_dir = cwd + '/step2_output/'
        if os.path.exists(step1_output_dir):
            if os.path.exists(step2_output_dir):
                raise NotImplementedError("second step processing exists!")
            else:
                os.mkdir(step2_output_dir)
        else:
            raise NotImplementedError("No first step processing exists!")

        input_files = [step1_output_dir + "step" + str(i+1) + "process.txt" for i in range(self.n_files)]
        output_files = [step2_output_dir + "step" + str(i + 1) + "process.txt" for i in range(self.n_files)]

        all_process = []
        print("Second Step Processing is Starting . . .")
        t0 = time()
        for i in range(self.n_process):
            pr = Process(target=self.__sentence_processing, args=(input_files[i], output_files[i]))
            all_process.append(pr)
            pr.daemon = True
            pr.start()

        for a_pr in all_process:
            a_pr.join()

        print("########################################################################################################"
              "HALF of INPUT FILES are NOW TOKENIZAED."
              "#######################################################################################################")

        for i in range(self.n_process):
            pr = Process(target=self.__sentence_processing, args=(input_files[i+self.n_process], output_files[i]))
            all_process.append(pr)
            pr.daemon = True
            pr.start()

        for a_pr in all_process:
            a_pr.join()

        t1 = time()
        print("Second Step Processing is Finised After: ", (t1-t0)/3600, " hour."
              "Output files of this step are stored in: ", step1_output_dir, ".")

    @staticmethod
    def __remove_chars(sentence, mode='first_step'):
        if mode == 'first_step':
            sentence = re.sub('[\u0041-\u005A\u0061-\u007A]+', 'انگلیش', sentence)
            sentence = re.sub('[\u0030-\u0039\u0660-\u0669\u06F0-\u06F9]+', 'نامبر', sentence)
            return re.sub(
                '[^ \u0622\u0624\u0626\u0627\u0628\u062A-\u063A\u0641-\u064A\u06CC\u067E\u0686\u0698'
                '\u06A9\u06AA\u06AF\u06BE\u06CC\u06D0\u200c\u060C\u061B\u061F\u002E\u0021]',
                "", sentence)
        elif mode == 'second_step':
            return re.sub(
                '[^ \u0622\u0624\u0626\u0627\u0628\u062A-\u063A\u0641-\u064A\u06CC\u067E\u0686\u0698'
                '\u06A9\u06AA\u06AF\u06BE\u06CC\u06D0]',
                "", sentence)
        else:
            raise NotImplementedError("Invalid mode of removing chars")

    def __sentence_processing(self, input_file, output_file):
        nlp_pipline = stanfordnlp.Pipeline(processors='tokenize', lang='fa')
        with open(input_file, 'r') as inp_file:
            print("Sentence tokenization procedure is Started for File: ", input_file,  ".")

            batch_size = 100
            data_processed = 0
            finished = False
            t_start = time()
            a_text = list(islice(inp_file, data_processed, data_processed + batch_size - 1))
            while not finished:
                t_s_batch = time()

                data_processed += len(a_text)
                doc = nlp_pipline('\n'.join(a_text))
                list_of_sentence = []

                for sent in doc.sentences:
                    sent_text = [a_word.text for a_word in sent.words]
                    sent_text = ' '.join(sent_text)
                    sent_text = self.__remove_chars(sent_text, mode='second_step')
                    if not sent_text.isspace():
                        sent_text = " ".join(sent_text.split())
                        list_of_sentence.append(sent_text)

                with open(output_file, 'a') as out_file:
                    out_file.write('\n'.join(list_of_sentence) + '\n')
                t_f_batch = time()

                # if data_processed % 2000 == 0:
                print(data_processed, "docs of ", output_file, " are processed. Average time for a batch "
                                                               "per minute: ", (t_f_batch - t_s_batch) / 200)

                a_text = list(islice(inp_file, data_processed, data_processed + batch_size - 1))
                if len(a_text) == 0:
                    finished = True

            t_final = time()
            print("Sentence tokenization for ", output_file, " is finished after", (t_final - t_start)/3600, "hour"
                  "Average time per a tokenization: ", (t_final - t_start)/data_processed, " seconds.")


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
    BASE_FILE_PATH = "MirasText_sample.txt"
    n_files = 8
    n_process = 4

    pr_corpus = ProcessCorpus(BASE_FILE_PATH, n_files, n_process)
    # pr_corpus.process_first_step()
    pr_corpus.process_second_step()
