#Methods to load training data

from __future__ import print_function
import sys
from collections import OrderedDict

import numpy as np
import numpy.random as npr
from sklearn import preprocessing

def dna_one_hot(seq, seq_len=None, flatten=True):
    """
    Input:
      seq
    Output:
       seq_vec: Flattened column vector
    """
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len) // 2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[int(seq[i-seq_start]),i] = 1
            except:
                seq_code[:,i] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None,:]

    return seq_vec

#test
seq = "ATAGCT"
dna_one_hot(seq)