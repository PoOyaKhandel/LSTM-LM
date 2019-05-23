"""
This module will pre-process Miras corpus, removes extra information, keeps only content of each sample and also
removes non-persian characters
"""
import stanfordnlp
import re

class ProcessCorpus:
    """

    """
    def __init__(self, fname):
        self.nlp_pipeline = stanfordnlp.Pipeline(processors='tokenize', lang='fa')
        all_text = self.__load_doc(fname)
        self.useful_content = self.__remove_extra_info(all_text)
        # self.write_file(relavant_text, 'check.txt')

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
            # doc = self.nlp_pipeline(tokens)
            # for sent in doc.sentences:
            #     final_tokens.append(' '.join(sent.words))

        # final_tokens = '\n'.join(final_tokens[:4])
        return final_tokens

    @staticmethod
    def __write_file(text, fname):
        # print(text)
        file = open(fname, 'w')
        file.write(text)

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
            sent_text = ' '.join(sent_text)
            sent_after_removing_chars = self.__remove_chars(sent_text)
            if not sent_after_removing_chars.isspace():
                sent_after_removing_chars = sent_after_removing_chars.strip()
                list_of_sent_after_removing_chars.append(sent_after_removing_chars)
        if len(list_of_sent_after_removing_chars) > 0:
            return '\n'.join(list_of_sent_after_removing_chars)
        else:
            return None

    def content_processing(self):
        list_of_all_clen_contents = []
        for content in self.useful_content:
            clean_content = self.__sentence_processing(content)
            if clean_content is not None:
                list_of_all_clen_contents.append(clean_content)
        self.__write_file('\n'.join(list_of_all_clen_contents), "clean_corpus.txt")
        #
        #
        #
        #
        # doc = self.nlp_pipeline(self.useful_content)
        # list_of_all_clen_sentences = []
        # for sent in doc.sentences:
        #     sent_text = [a_word.text for a_word in sent.words]
        #     sent_text = ' '.join(sent_text)
        #     sent_after_removing_chars = self.__remove_chars(sent_text)
        #     if not sent_after_removing_chars.isspace():
        #         sent_after_removing_chars = sent_after_removing_chars.strip()
        #         list_of_all_clen_sentences.append(sent_after_removing_chars)
        # print(list_of_all_clen_sentences[:2])
        # self.__write_file('\n'.join(list_of_all_clen_sentences), "clean_corpus.txt")



# def remove_non_persian(mixed_str):
#     mixed_str_words = mixed_str.split(' ')
#     empty = lambda x: x != ''
#
#     min_range = int("0x600", 0)
#     max_range = int("0x6FF", 0)
#     clean_str_words = []
#
#     for aWord in filter(empty, mixed_str_words):
#         if min_range < ord(aWord[0]) < max_range:
#             clean_str_words.append(aWord)
#     # clean_str = " ".join(clean_str_words)
#     return clean_str_words
#
#
#
#
#
# def create_seq(tokens):
#     length = 50 + 1
#     sequences = []
#     for i in range(length, len(tokens)):
#         # select sequence of tokens
#         m = i - length
#         n = i
#         seq = tokens[m:n]
#         # convert into a line
#         c = len(seq)
#         line = ' '.join(seq)
#         # store
#         a = len(line)
#         b = tokens[n]
#         sequences.append(line)
#     return sequences
#
#
# # turn a doc into clean tokens
# def clean_doc(doc):
#     lines = doc.split("\n")
#     final_tokens = []
#     for line in lines:
#         tokens = line.split('***')
#         dirty_content = tokens[0]
#         clean_content = remove_non_persian(dirty_content)
#         if len(clean_content) != 0:
#             final_tokens.extend(clean_content)
#
#     sequences = create_seq(final_tokens)
#     return sequences
#
#
# # save tokens to file, one dialog per line
# def save_doc(lines, filename):
#     data = '\n'.join(lines)
#     file = open(filename, 'w')
#     file.write(data)
#     file.close()


if __name__ == '__main__':
    fname = "MirasText_sample.txt"
    test = ProcessCorpus(fname)
    test.content_processing()



