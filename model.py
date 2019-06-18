import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    """
    LSTM network description for training language model
    """
    def __init__(self, emb_size, vocab_size, n_hidden_l, train_on_gpu=False):
        super(LSTMModel).__init__(self)

        self.embedding_dim = emb_size
        self.vocab_size = vocab_size
        self.n_hidden_l = n_hidden_l
        self.train_on_gpu = train_on_gpu

        self.__build_model()

    def __build_model(self):
        padding_indx = 0
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_indx
        )

        self.LSTM = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.n_hidden_l,
            num_layers=2
        )

        self.hidden_to_tag = nn.Linear(self.n_hidden_l, self.vocab_size)

        self.softmax = nn.Softmax(self.hidden_to_tag)

    def init_hidden(self):
        hidden_a = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return hidden_a, hidden_b

    def forward(self, *input):
