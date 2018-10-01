"""
Read results from prediction run and compute alphas post-mortem to decide
best values for each relation
"""
import sys
import pandas as pd
import numpy as np
from metrics import mrr, h_at_n

__author__ = "Yuval Pinter, 2018"

ALPHAS_RANGE = np.linspace(0,1,101)

if __name__ == '__main__':
    in_file = sys.argv[1]
    print('reading from {}'.format(in_file))
    N = 6068 if 'dev' in in_file else 6168
    with open(in_file) as in_read:
        df = pd.read_csv(in_read, sep='\t')
    
    rel_alphas = {}
    agg_best_ranks = []
    agg_zero_ranks = []
    agg_half_ranks = []
    agg_one_ranks = []
    
    for rl in set(df.rel):
        rel_df = pd.DataFrame(df[df.rel == rl])
        if len(rel_df) == 0:
            rel_alphas[rl] = 0.5
            continue
            
        best_mrr = 0.
        best_arg_mrr = -1.
        rel_best_ranks = []
        print('relation\talpha\tMRR\th@10\th@3\th@1')
        for alpha in ALPHAS_RANGE:
            rel_df['total'] = (rel_df.assoc_score * (1. - alpha)) + (rel_df.ergm_score * alpha)
            ranks = []
            for i in rel_df.idx.unique():
                ddf = rel_df[rel_df.idx == i]
                ranked = ddf.sort_values(by='total', ascending=False)
                r = 1
                for label in ranked.is_gold:
                    if label:
                        ranks.append(r)
                        r -= 1 # based on accepted eval method
                    r += 1
            
            if alpha == 0.0: agg_zero_ranks.extend(ranks)
            if alpha == 0.5: agg_half_ranks.extend(ranks)
            if alpha == 1.0: agg_one_ranks.extend(ranks)
            
            amrr = mrr(ranks)
            
            if amrr > best_mrr:
                rel_best_ranks = ranks
                best_mrr = amrr
                best_arg_mrr = alpha
            
            # just for reporting
            hat10 = h_at_n(ranks, n=10)
            hat3 = h_at_n(ranks, n=3)
            hat1 = h_at_n(ranks, n=1)
            
            if int(alpha * 100) % 20 == 0:
                print('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(rl, alpha, amrr, hat10, hat3, hat1))
        
        agg_best_ranks.extend(rel_best_ranks)
        rel_alphas[rl] = best_arg_mrr
        print('=====')
        print('{} best mrr: alpha = {} ({:.4f})'.format(rl, best_arg_mrr, best_mrr))
    
    overall_best_mrr = mrr(agg_best_ranks)
    zero_mrr = mrr(agg_zero_ranks)
    half_mrr = mrr(agg_half_ranks)
    one_mrr = mrr(agg_one_ranks)
    
    best_10 = h_at_n(agg_best_ranks, n=10)
    best_3 = h_at_n(agg_best_ranks, n=3)
    best_1 = h_at_n(agg_best_ranks, n=1)
    
    half_10 = h_at_n(agg_half_ranks, n=10)
    half_3 = h_at_n(agg_half_ranks, n=3)
    half_1 = h_at_n(agg_half_ranks, n=1)
    
    print('=====')
    print('best mrr = {:.4f}, from alphas =\n{}\n'.format(overall_best_mrr, rel_alphas))
    print('mrr at 0.0, 0.5, 1.0: {:.4f}, {:.4f}, {:.4f}'.format(zero_mrr, half_mrr, one_mrr))
    print('total {} lists, {} instances, k = {}'.format(len(df.idx.unique()), len(agg_best_ranks), len(ddf)))
    tot_ratio = float(len(agg_best_ranks)) / N
    print('total gains over 0.5: {:.5f} mrr, {:.5f} h@10, {:.5f} h@3, {:.5f} h@1'\
                .format((overall_best_mrr - half_mrr) * tot_ratio,
                        (best_10 - half_10) * tot_ratio,
                        (best_3 - half_3) * tot_ratio,
                        (best_1 - half_1) * tot_ratio))
                