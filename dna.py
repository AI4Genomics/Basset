#Methods to load training data

import numpy as np

#TODO: make sure mapping is the same order used in the preprocessing steps
def dna_to_onehot(seq_nts):
    """function that takes a dna sequence of nucleotide and returns onehot-encoded representation of it"""
    nt_dict = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]
              }
    return np.asarray([nt_dict[nt.upper()] for nt in seq_nts]).transpose() # transpose to have nb_cols=nb_nts

# test
seq = "ATGCAATTAGCT"
seq_vec = dna_to_onehot(seq)
print(seq_vec)

#TODO: transform one-hot array to dna characters (not vector)
def vecs2dna(seq_vecs):
    '''
    Input:
        seq_vecs:
    Output:
        seqs
    '''

    # possibly reshape
    if len(seq_vecs.shape) == 2:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))
    elif len(seq_vecs.shape) == 4:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))

    seqs = []
    for i in range(seq_vecs.shape[0]):
        seq_list = ['']*seq_vecs.shape[2]
        for j in range(seq_vecs.shape[2]):
            if seq_vecs[i,0,j] == 1:
                seq_list[j] = 'A'
            elif seq_vecs[i,1,j] == 1:
                seq_list[j] = 'C'
            elif seq_vecs[i,2,j] == 1:
                seq_list[j] = 'G'
            elif seq_vecs[i,3,j] == 1:
                seq_list[j] = 'T'
            elif seq_vecs[i,:,j].sum() == 1:
                seq_list[j] = 'N'
            else:
                print('Malformed position vector: ', seq_vecs[i,:,j], 'for sequence %d position %d' % (i,j), file=sys.stderr)
        seqs.append(''.join(seq_list))
    return seqs

#test
vecs2dna(seq_vec)
