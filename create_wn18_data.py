"""
this script creates a pickled dataset for training a relation prediction model.
usage: `python create_wn18_data.py --base-dir <location> [--no-symmetrics] --output <location>`
"""
import numpy as np
import pickle as pickle
import argparse
from os import path
from collections import defaultdict
from itertools import count
from io_utils import SparseMatrixDataset, WordnetPredictionDataset
from multigraph_utils import to_csr_matrix
from consts import SYMMETRIC_RELATIONS

DEFAULT_OUTPUT = 'data/sparse_wn18_matrices.pkl'

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="directory with train/valid/test files in text form")
    parser.add_argument("--no-symmetrics", action="store_true", help="create WN-18RSR format")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=".pkl file with CSR graphs")
    opts = parser.parse_args()
    train_file = path.join(opts.base_dir, 'train.txt')
    dev_file = path.join(opts.base_dir, 'valid.txt')
    test_file = path.join(opts.base_dir, 'test.txt')

    # create dictionary from synset name to index
    synset_dict = defaultdict(count(0).__next__)

    # train
    rels = {}
    with open(train_file, 'r') as train_insts:
        rels['train'] = defaultdict(lambda: defaultdict(list))
        for i, inst in enumerate(train_insts):
            from_s, rel, to_s = inst.strip().split('\t')
            rel = rel[1:] # no leading underscore
            if opts.no_symmetrics and rel in SYMMETRIC_RELATIONS: continue
            from_i = synset_dict[from_s]
            to_i = synset_dict[to_s]
            rels['train'][rel][from_i].append(to_i)
        print('read {} instances with {} relations from {}, now have {} synsets'\
                .format(i, len(rels['train']), train_file, len(synset_dict)))

    # dev
    with open(dev_file, 'r') as dev_insts:
        rels['dev'] = defaultdict(lambda: defaultdict(list))
        for i, inst in enumerate(dev_insts):
            from_s, rel, to_s = inst.strip().split('\t')
            rel = rel[1:] # no leading underscore
            if opts.no_symmetrics and rel in SYMMETRIC_RELATIONS: continue
            from_i = synset_dict[from_s]
            to_i = synset_dict[to_s]
            rels['dev'][rel][from_i].append(to_i)
        print('read {} instances with {} relations from {}, now have {} synsets'\
                .format(i, len(rels['dev']), dev_file, len(synset_dict)))

    # test
    with open(test_file, 'r') as test_insts:
        rels['test'] = defaultdict(lambda: defaultdict(list))
        for i, inst in enumerate(test_insts):
            from_s, rel, to_s = inst.strip().split('\t')
            rel = rel[1:] # no leading underscore
            if opts.no_symmetrics and rel in SYMMETRIC_RELATIONS: continue
            from_i = synset_dict[from_s]
            to_i = synset_dict[to_s]
            rels['test'][rel][from_i].append(to_i)
        print('read {} instances with {} relations from {}, now have {} synsets'\
                .format(i, len(rels['test']), test_file, len(synset_dict)))
    
    rel_names = set(list(rels['train'].keys()) + list(rels['dev'].keys()) + list(rels['test'].keys()))
    print('all relations: ', rel_names)
    node_list = [k for k,v in sorted(list(synset_dict.items()), key = lambda x: x[1])]
    print('node count: ', len(node_list))
    # sanity check
    print('first sorted synsets: ', node_list[:10])
    print('last sorted synsets: ', node_list[-10:])
    
    # create sparse matrices
    matrix_set = {}
    for div in ['train', 'dev', 'test']:
        sparse_matrices = {}
        print('creating matrices for relations in {} set'.format(div))
        for rel in rel_names:
            sparse_graph = [sorted(rels[div][rel][synset_dict[s]]) for s in node_list]
            sparse_matrices[rel] = to_csr_matrix(sparse_graph)
            # only reporting for rest of loop step
            print('created matrix for relation {}'.format(rel))
            non_nulls = [(ii, ts) for ii, ts in enumerate(sparse_graph) if len(ts) > 0]
            i, trgs = non_nulls[np.random.choice(range(len(non_nulls)))]
            print('sample: {}->{}'.format(node_list[i], [node_list[j] for j in trgs]))
        matrix_set[div] = SparseMatrixDataset(sparse_matrices, node_list)

    # save matrices
    print('saving dataset')
    to_pickle = WordnetPredictionDataset(matrix_set['train'], matrix_set['dev'], matrix_set['test'])
    pickle.dump(to_pickle, open(opts.output, 'wb'))
    print('done!')
