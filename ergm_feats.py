"""
Feature extractor for ERGM-based models.
Inputs named `g` are graphs, `gs` are lists of graphs, all in CSR format, unless
explicitly stated otherwise.
"""
import numpy as np

__author__ = "Yuval Pinter, 2018"

def self_loops(g):
    return g.diagonal().sum()


def edge_count(g):
    return g.getnnz()


def mutual_edges(g1, g2):
    return g1.multiply(g2.transpose()).sum() # this is pointwise


def stars_count(g, k, direction):
    """
    :param g: graphs
    :param n: k of star
    :param direction: in or out star
    """
    dir = 0 if direction == 'in' else 1
    return sum(g.sum(dir) == k).sum()    


def one_rel_star_counts(in_degs, out_degs):
    osc = {}
    
    osc['i1sc'] = sum(in_degs == 1).sum()
    osc['o1sc'] = sum(out_degs == 1).sum()
    osc['i2sc'] = sum(in_degs == 2).sum()
    osc['o2sc'] = sum(out_degs == 2).sum()
    osc['i3sc'] = sum(in_degs == 3).sum()
    osc['o3sc'] = sum(out_degs == 3).sum()

    osc['i1psc'] = sum(in_degs >= 1).sum()
    osc['o1psc'] = sum(out_degs >= 1).sum()
    osc['i2psc'] = sum(in_degs >= 2).sum()
    osc['o2psc'] = sum(out_degs >= 2).sum()
    osc['i3psc'] = sum(in_degs >= 3).sum()
    osc['o3psc'] = sum(out_degs >= 3).sum()
    
    return osc


def two_rel_star_counts(in_degs1, out_degs1, in_degs2, out_degs2):
    tsc = {}

    tsc['i2sc'] = (sum(np.multiply(in_degs1, in_degs2)) == 1).sum()
    tsc['o2sc'] = (sum(np.multiply(out_degs1, out_degs2)) == 1).sum()
    tsc['i3sc112'] = (sum(np.multiply(in_degs1-1, in_degs2)) == 1).sum()
    tsc['o3sc112'] = (sum(np.multiply(out_degs1-1, out_degs2)) == 1).sum()
    tsc['i3sc122'] = (sum(np.multiply(in_degs1, in_degs2-1)) == 1).sum()
    tsc['o3sc122'] = (sum(np.multiply(out_degs1, out_degs2-1)) == 1).sum()
    
    tsc['i2psc'] = (sum(np.multiply(in_degs1, in_degs2)) >= 1).sum()
    tsc['o2psc'] = (sum(np.multiply(out_degs1, out_degs2)) >= 1).sum()
    tsc['i3psc112'] = (sum(np.multiply(in_degs1-1, in_degs2)) >= 1).sum()
    tsc['o3psc112'] = (sum(np.multiply(out_degs1-1, out_degs2)) >= 1).sum()
    tsc['i3psc122'] = (sum(np.multiply(in_degs1, in_degs2-1)) >= 1).sum()
    tsc['o3psc122'] = (sum(np.multiply(out_degs1, out_degs2-1)) >= 1).sum()

    return tsc


def three_rel_star_counts(in_degs1, out_degs1, in_degs2, out_degs2, in_degs3, out_degs3):
    ttsc = {}
    
    ttsc['i3sc'] = (sum(np.multiply(in_degs1,\
                    np.multiply(in_degs2, in_degs3)))\
                    == 1).sum()
    ttsc['o3sc'] = (sum(np.multiply(out_degs1,\
                    np.multiply(out_degs2, out_degs3)))\
                    == 1).sum()
    ttsc['i3psc'] = (sum(np.multiply(in_degs1,\
                    np.multiply(in_degs2, in_degs3)))\
                    >= 1).sum()
    ttsc['o3psc'] = (sum(np.multiply(out_degs1,\
                    np.multiply(out_degs2, out_degs3)))\
                    >= 1).sum()
    
    return ttsc
