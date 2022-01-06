import torch
import numpy as np

def subsequent_mask(size):
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8') # masking with upper triangle matrix
    return torch.from_numpy(mask)==0 # reverse (masking=False, non-masking=True)

def make_std_mask(tgt):
    tgt_mask = (tgt != np.zeros_like(tgt)) # pad masking
    tgt_mask = tgt.unsqueeze(-2) # reshape (n_batch, seq_len) -> (n_batch, 1, seq_len)
    tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # pad_masking & subsequent_masking
    return tgt_mask