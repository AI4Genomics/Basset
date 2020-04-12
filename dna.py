"""Helpers for working with DNA/RNA data"""

import os
import numpy as np

"""Helpers for working with DNA data"""

# fix vocabulary
dna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "*":4} # catch-all auxiliary token

dna_nt_only_vocab = {k:v for k,v in dna_vocab.items() if k in "ACGT"} # dna nucleotides only
rev_dna_vocab = {v:k for k,v in dna_nt_only_vocab.items()}

def get_vocab(vocab_name, vocab_order=None):
    if vocab_name=="dna":
        charmap = dna_vocab
    elif vocab_name=="dna_nt_only":
        charmap = dna_nt_only_vocab
    else:
        raise Exception("Unknown vocabulary name.")

    if vocab_order:
        if set(vocab_order) != set(charmap):
            raise ValueError("Provided `vocab` and `vocab_order` arguments are not compatible")
        else:
            charmap = {c: idx for idx, c in enumerate(vocab_order)}

    rev_charmap = {v: k for k, v in charmap.items()}
    return charmap, rev_charmap


test_seq = "ATATGCGTCCCCA"
get_vocab(test_seq)