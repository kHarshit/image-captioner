import torch
import torch.nn as nn
from torchvision.models import resnet101
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    """CNN acting as encoder"""

    def __init__(self, embedding_dim):
        """Replace fc layer of pretrained model and initialize layers"""

        super().__init__()
        model = resnet101(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)  # unpacking
        self.fc = nn.Linear(model.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)

    def forward(self, x):
        """Extract feature vectors from input images
        :param x: input images of size [B, C, H, W]
        :return: encoded images (feature vectors)"""

        with torch.no_grad():
            x = self.model(x)
        x = self.reshape(x.size(0), -1)  # features
        x = self.bn(self.fc(x))  # apply batch norm
        return x


class DecoderRNN(nn.Module):
    """A normal decoder RNN"""

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers, max_seq_len=20, drop_prob=0.5):
        """
        :param embedding_dim:
        :param hidden_dim: the number of features in the RNN output and in the hidden state (no of units in hidden layer)
        :param n_layers: the number of layers that make up the RNN, typically 1-3; greater than 1 means that you'll create a stacked RNN
        :param max_seq_len: max length of sentence to generate
        :param batch_first: whether or not the input/output of the RNN will have the batch_size as the first dimension (batch_size, seq_length, hidden_dim)
        """
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoded_captions, x, caption_lengths):
        """Decodes image feature vectors and generate captions"""

        # (batch_size, max_caption_length, embed_dim)
        embeds = self.embeds(encoded_captions)
        embeds = torch.cat((x.unsqueeze(1), embeds), 1)
        packed = pack_padded_sequence(embeds, caption_lengths, batch_first=True)
        lstm_out, hidden = self.lstm(packed)
        out = self.dropout(lstm_out)
        out = self.fc(out[0])
        return out


def init_hidden(self, features, states=None):
    """Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images"""

    sampled_ids = []
    inputs = features.unsqueeze(1)
    for i in range(self.max_seg_length):
        hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1)  # predicted: (batch_size)
        sampled_ids.append(predicted)
        inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
    sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
    return sampled_ids
