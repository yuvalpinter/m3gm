import dynet as dy
import numpy as np

__author__ = "Yuval Pinter, 2018"

def softmaxify(neg_assocs):
    """
    stable softmax function built before dynet's utility was efficient
    :param neg_assocs: association scores for negative samples
    :return: numpy array of softmaxed scores
    """
    #return dy.softmax(dy.concatenate(list(neg_assocs.values()))).value() # can replace 4 lines below
    neg_assoc_scores = np.array(list(neg_assocs.values()))
    exp_neg_assocs = np.exp(neg_assoc_scores - np.max(neg_assoc_scores))
    assoc_scores_sumexp = np.sum(exp_neg_assocs)
    return exp_neg_assocs / assoc_scores_sumexp


def dyagonalize(col):
    """
    A convoluted way to make a dynet vector into a dynet matrix where it's the diagonal
    God I hope there's a better way.
    :param col: column vector in dynet format
    """
    col_dim = col.dim()[0][0]
    nump_eye = np.eye(col_dim)
    return dy.cmult(col, dy.inputTensor(nump_eye))