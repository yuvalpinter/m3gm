'''
Script not aligned with any particular task, solely for exploration.
May contain redundancies.
'''

import pickle as pickle
import argparse
from collections import namedtuple
from scipy.sparse import csr_matrix

from nltk.corpus import wordnet as wn

__author__ = "Yuval Pinter, 2018"

SparseMatrixDataset = namedtuple('SparseMatrixDataset', ['matrices', 'index'])
DEFAULT_OUTPUT = 'data/sparse_wordnet_matrices.pkl'

def to_csr_matrix(ind_lists: list) -> csr_matrix:
    '''
    :param ind_lists: a list of pointers from each synsets to its relations
    :returns: CSR matrix representation
    '''
    data = []
    indices = []
    indptr = []
    for i,l in enumerate(ind_lists):
        indptr.append(len(data))
        indices.extend(l)
        data.extend([1] * len(l))
    i += 1
    indptr.append(len(data))
    return csr_matrix((data,indices,indptr), shape=(i, i))

### MAIN ###

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=".pkl file with CSR graphs")
    opts = parser.parse_args()
    
    sets = list(wn.all_synsets())

    rel_names = ['hypernyms', 'hyponyms', 'entailments', 'attributes', 'causes', 'member meronyms',\
                    'substance meronyms', 'part meronyms', 'member holonyms', 'substance holonyms', 'part holonyms', 'similar tos']
    rel_funcs = [lambda s: s.hypernyms(), lambda s: s.hyponyms(), lambda s: s.entailments(), lambda s: s.attributes(),\
                    lambda s: s.causes(), lambda s: s.member_meronyms(), lambda s: s.substance_meronyms(),\
                    lambda s: s.part_meronyms(), lambda s: s.member_holonyms(), lambda s: s.substance_holonyms(),\
                    lambda s: s.part_holonyms(), lambda s: s.similar_tos()]
                    
    # a function for all holonym types, a function for all meronym types
    all_holonyms = lambda sn: sn.part_holonyms() + sn.member_holonyms() + sn.substance_holonyms()
    all_meronyms = lambda sn: sn.part_meronyms() + sn.member_meronyms() + sn.substance_meronyms()
    
    rel_names.extend(['all holonyms', 'all meronyms'])
    rel_funcs.extend([all_holonyms, all_meronyms])

    # create sparse matrices
    inv_synset_dict = {sn:i for i,sn in enumerate(sets)}
    sparse_matrices = {}
    for rel, func in zip(rel_names, rel_funcs):
        sparse_graph = [[inv_synset_dict[j] for j in func(sn)] for sn in sets]
        if rel == 'hyponyms':
            # cycle that needs fixing - see `wordnet_analysis.ipynb`
            sparse_graph[inv_synset_dict[wn.synset('restrain.v.01')]].remove(inv_synset_dict[wn.synset('inhibit.v.04')])
        sparse_matrices[rel] = to_csr_matrix(sparse_graph)

    # save matrices
    to_pickle = SparseMatrixDataset(sparse_matrices, [s.name() for s in sets])
    pickle.dump(to_pickle, open(opts.output, 'wb'))

