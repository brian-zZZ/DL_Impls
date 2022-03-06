import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    '''Untrainable sine-cosine positional embedding'''
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=1000):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        :param max_len: 
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_size))
        # self.pos_embed = PositionalEmbedding(embed_size, max_len)

        # Input vocabulary size of seg embedding is 2 (either 1 or 2),
        # 0 for padding addtionally
        self.seg_embed = nn.Embedding(3, embed_size, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        # Weight initial
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, sequence, segment_label):
        x = self.token_embed(sequence) + self.seg_embed(segment_label)
        x = x + self.pos_embed.data[:, :x.shape[1], :]

        # For sine-cosine pos embedding
        # x = self.token_embed(sequence) + self.seg_embed(segment_label) + self.pos_embed(sequence)
        
        return self.dropout(x)
