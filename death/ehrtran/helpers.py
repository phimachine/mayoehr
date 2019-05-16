import torch
import numpy as np
from torch.autograd import Variable

def get_non_pad_mask(seq, key_length=None):
    if key_length is None:
        assert seq.dim() == 2
        return seq.ne(0).float().unsqueeze(-1)
    else:
        batch = seq.shape[0]
        len_seq = seq.shape[1]
        mask=torch.zeros(batch,len_seq)
        for i in range(batch):
            mask[i,:key_length[i]]=1
        return mask.float().unsqueeze(-1)

def get_src_pos(seq, key_length):
    len_seq=seq.shape[1]
    src_pos=torch.arange(1, len_seq+1)
    src_pos=src_pos*get_non_pad_mask(seq,key_length).squeeze(-1).long()
    return src_pos


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q, key_length=None):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    if key_length is None:
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        return padding_mask
    else:
        len_q = seq_q.size(1)
        batch = seq_k.size(0)
        len_k = seq_k.size(1)
        padding_mask = torch.ones(batch,len_k).byte()
        for i in range(batch):
            padding_mask[i,:key_length[i]]=0
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s)).byte().cuda(), diagonal=1)
    subsequent_mask = Variable(subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)) # b x ls x ls

    return subsequent_mask


def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, \
         'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding


def norm2reci(tensor):
    norm = tensor.pow(2).sum().rsqrt()
    # if the residual is inf
    norm = norm.clamp(0,1e10)
    return norm
