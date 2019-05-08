"""
This module will pre-process Miras corpus, removes extra information, keeps only content of each sample and also
removes non-persian characters
"""


def remove_non_persian(mixed_str):
    mixed_str_words = mixed_str.split(' ')
    empty = lambda x: x != ''

    min_range = int("0x600", 0)
    max_range = int("0x6FF", 0)
    clean_str_words = []

    for aWord in filter(empty, mixed_str_words):
        if min_range < ord(aWord[0]) < max_range:
            clean_str_words.append(aWord)
    # clean_str = " ".join(clean_str_words)
    return clean_str_words


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def create_seq(tokens):
    length = 49 + 1
    sequences = []
    for i in range(int(len(tokens)/length - 1)):
        # select sequence of tokens
        m = length * i
        n = length * (i + 1) - 1
        seq = tokens[m: n]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    return sequences


# turn a doc into clean tokens
def clean_doc(doc):
    lines = doc.split("\n")
    final_tokens = []
    for line in lines:
        tokens = line.split('***')
        dirty_content = tokens[0]
        clean_content = remove_non_persian(dirty_content)
        if len(clean_content) != 0:
            final_tokens.extend(clean_content)

    sequences = create_seq(final_tokens)
    return sequences


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


if __name__ == "__main__":
    f = '/home/po_oya/NLP Thesis/MirasText.txt'
    f2 = "MirasText_sample.txt"
    preprocess(f2, "processed_miras.txt")

