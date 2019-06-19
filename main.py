import prerpocess_corpus

CLEAN_CORPUS = True
DIRTY_CORPUS_PATH = "/home/po_oya/Thesis/LSTM-LM/MirasText_sample.txt"
CLEAN_CORPUS_PATH = ""


if CLEAN_CORPUS:
    clean_unit = prerpocess_corpus.ProcessCorpus(DIRTY_CORPUS_PATH)
