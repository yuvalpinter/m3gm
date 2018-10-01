import numpy as np

__author__ = "Yuval Pinter, 2018"

def h_at_n(ranks, n=10):
    """
    hits at n
    :param ranks: list of predicted ranks for targets
    :param n: metric parameter, targets appearing in top n get counted
    :return: hits at n score
    """
    return float(len([r for r in ranks if r <= n])) / len(ranks)


def mq(ranks, max_rank):
    """
    mean quantile
    :param ranks: list of predicted ranks for targets
    :param max_rank: total size of ranked list
    :return: mean quantile score
    """
    return 1.0 - (np.average(ranks)/(max_rank-1))


def mrr(ranks):
    """
    mean reciprocal rank
    :param ranks: list of predicted ranks for targets
    :return: mean reciprocal rank score
    """
    return np.average([1.0/r for r in ranks])
