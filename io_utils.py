import numpy as np
import pickle as pickle
import codecs
from collections import namedtuple
from datetime import datetime
from consts import DEFAULT_EMBEDDING_DIM

__author__ = "Yuval Pinter, 2018"

WordnetPredictionDataset = namedtuple('WordnetPredictionDataset', ['train', 'dev', 'test'])
SparseMatrixDataset = namedtuple('SparseMatrixDataset', ['matrices', 'index'])


def timeprint(str):
    """
    Unclever method for logging just the time of the printed line.
    :param str:
    :return:
    """
    print('{}\t{}'.format(datetime.now(), str))


def load_prediction_dataset(filename):
    """
    :param filename: file containing WordnetPredictionDataset with WordNet graphs in train, dev and test
    """
    ds = pickle.load(open(filename, 'rb'))
    return ds, ds.train.index


def load_graphs(filename):
    """
    loads WordNet graphs from pre-pickled resource
    :param filename: .pkl file with graph in sparse matrices format
    """
    ds = pickle.load(open(filename, 'rb'))
    return ds.matrices, ds.index


def load_embeddings(filename, a2i, emb_size=DEFAULT_EMBEDDING_DIM):
    """
    loads embeddings for synsets ("atoms") from existing file,
    or initializes them to uniform random
    """
    atom_to_embed = {}
    if filename is not None:
        if filename.endswith('npy'):
            return np.load(filename)
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                split = line.split()
                if len(split) > 2:
                    atom = split[0]
                    vec = split[1:]
                    atom_to_embed[atom] = np.asfarray(vec)
        embedding_dim = len(atom_to_embed[list(atom_to_embed.keys())[0]])
    else:
        embedding_dim = emb_size
    out = np.random.uniform(-0.8, 0.8, (len(a2i), embedding_dim))
    if filename is not None:
        for atom, embed in list(atom_to_embed.items()):
            if atom in a2i:
                out[a2i[atom]] = np.array(embed)
    return out
