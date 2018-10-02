import numpy as np
import dynet as dy
import argparse
import copy
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from contextlib import contextmanager

from model import MultiGraphErgm
from training import macro_loops
from io_utils import timeprint, load_prediction_dataset, load_embeddings
from multigraph_utils import targets, sources, join_sets, co_graph
from metrics import h_at_n, mrr, mq
from pretrain_assoc import TRANSLATIONAL_EMBED_MODE, MODES_STR
from consts import SYMMETRIC_RELATIONS, M3GM_DEFAULT_NEGS

__author__ = "Yuval Pinter, 2018"

none_context = contextmanager(lambda: iter([None]))()


def node_loop(change_idx, ergm, rel, node, assoc_cache, model_caches, tr_gr, te_gr, override_rel, rerank_k, is_source,
              log_file, rerank_file):
    # collect gold data for source node
    gold_trues = targets(te_gr, node) if is_source else sources(te_gr, node)
    if len(gold_trues) == 0:
        return 0, [], [], [], [], [], change_idx

    # engineer the evaluation to ignore trivialities like trues affecting trues
    known_trues = targets(tr_gr, node) if is_source else sources(tr_gr, node)
    sym_cands = sources(tr_gr, node) if is_source else targets(tr_gr, node)
    score_ignores = known_trues + sym_cands + [node]
    rank_ignores = score_ignores + gold_trues

    # rank based on association score
    if is_source:
        assoc_scores = ergm.score_from_source_cache(assoc_cache, node)
    else:
        assoc_scores = ergm.score_from_target_cache(assoc_cache, node)

    local_gold_ranks = {}

    # compute ranks before re-ranking
    for g in gold_trues:
        # rule override
        if override_rel:
            if g in sym_cands:
                local_gold_ranks[g] = 1
                continue
        g_score = assoc_scores[g]
        rank = 1 + len([v for i, v in enumerate(assoc_scores) if v > g_score \
                        and i not in rank_ignores])
        local_gold_ranks[g] = rank

    if log_file is not None:
        if is_source:
            log_file.write('{} targets for source {}:{} found in ranked places {}\n' \
                           .format(rel, node, synsets[node], list(local_gold_ranks.values())))
        else:
            log_file.write('{} sources for target {}:{} found in ranked places {}\n' \
                           .format(rel, node, synsets[node], list(local_gold_ranks.values())))

    # find gold not to be re-ranked
    unchanged_local_ranks = [i for i in list(local_gold_ranks.values()) \
                             if i + len([j for j in list(local_gold_ranks.values()) if j < i]) > rerank_k]

    if override_rel or len(unchanged_local_ranks) == len(gold_trues):
        return len(gold_trues), unchanged_local_ranks, list(local_gold_ranks.values()), [], [], [], change_idx

    # rerank
    full_ranking = [i for i, t in sorted(enumerate(assoc_scores), key=lambda x: -x[1]) if i not in score_ignores]
    to_rerank = full_ranking[:rerank_k]  # synset indices

    gold_idcs_to_be_reranked = [g for g in list(local_gold_ranks.keys()) if g in to_rerank]
    gold_ranks_to_be_reranked = [local_gold_ranks[k] for k in gold_idcs_to_be_reranked]

    # re-rank top n based on ergm score
    reranked_scores = {t: assoc_scores[t] for t in to_rerank}
    erg_scores = {}
    for n in to_rerank:  # TODO this part is parallelizable
        src = node if is_source else n
        trg = n if is_source else node
        perm = False

        erg_gold_score = ergm.add_edge(src, trg, rel, perm,
                                       caches=model_caches).scalar_value()
        reranked_scores[n] += erg_gold_score
        erg_scores[n] = erg_gold_score

        if rerank_file is not None:
            rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\n' \
                              .format(rel, change_idx, is_source, node, n, n in gold_trues, assoc_scores[n],
                                      erg_gold_score))

    gold_reranked = []
    gold_ergs = []
    gold_idcs = []
    for gn in gold_idcs_to_be_reranked:
        gold_reranked.append(1 + len([t for k, t in list(reranked_scores.items()) \
                                      if t > reranked_scores[gn] and k not in rank_ignores]))
        gold_ergs.append(1 + len([t for k, t in list(erg_scores.items()) \
                                  if t > erg_scores[gn] and k not in rank_ignores]))
        gold_idcs.append(synsets[g])

    # report how places changed after re-ranking
    pls = list(zip(gold_idcs_to_be_reranked, gold_idcs,
                   gold_ranks_to_be_reranked, gold_reranked, gold_ergs))
    if is_source:
        desc = 'target for source'
    else:
        desc = 'source for target'
    if log_file is not None:
        for pl in pls:
            log_file.write('{}\t{} {} {}:{}, {}:{} re-ranked from [{}] to [{}], by ergm score only [{}]\n' \
                           .format(change_idx, rel, desc, node, synsets[node], pl[0], pl[1], pl[2], pl[3], pl[4]))
            print('{} {} {}:{}, {}:{}, re-ranked from {} to {}, by ergm score only {}' \
                  .format(rel, desc, node, synsets[node], pl[0], pl[1], pl[2], pl[3], pl[4]))

    return len(gold_trues), unchanged_local_ranks, list(
        local_gold_ranks.values()), gold_reranked, gold_ergs, pls, change_idx + 1


def eval(prev_graphs, graphs, ergm, opts, N, log_file, rerank_file):
    writing = log_file is not None

    caches = (copy.deepcopy(ergm.cache),
              copy.deepcopy(ergm.feature_vals))

    rel_all_ranks = {}  # for final results
    rel_pre_ranks = {}  # for improvement analysis
    rel_erg_ranks = {}  # for ergm-alone analysis
    all_pre_ranks = []
    all_all_ranks = []
    all_erg_ranks = []
    insts = Counter()
    total_misses = Counter()
    overrides = Counter()
    rerank_ups = Counter()
    rerank_downs = Counter()
    erg_ups = Counter()
    erg_downs = Counter()
    rerank_diff = Counter()
    erg_diff = Counter()

    change_idx = 1

    rels_order = list(graphs.items())
    for rel, te_gr in rels_order:
        if rel == 'co_hypernym':
            continue

        # set up
        if writing:
            timeprint('testing relation {}'.format(rel))
            log_file.write('relation: {}\n'.format(rel))
        # add incrementally, eval each edge, revert
        tr_gr = prev_graphs[rel]  # to filter known connections
        s_assoc_cache = ergm.source_ranker_cache(rel)
        t_assoc_cache = ergm.target_ranker_cache(rel)
        override_rel = opts.rule_override and rel in SYMMETRIC_RELATIONS
        all_ranks = []
        pre_ranks = []
        erg_ranks = []
        if override_rel and writing:
            log_file.write('RELATION OVERRIDE\n')
        node_order = list(range(N))  # DO NOT RANDOMIZE THIS - NEED TO PREDICT BOTH SIDES
        for node in tqdm(node_order):
            s_trues, s_unch_loc_ranks, s_loc_gold_ranks, s_gold_reranked, s_gold_ergs, s_pls, change_idx = \
                node_loop(change_idx, ergm, rel, node, s_assoc_cache,
                          caches, tr_gr, te_gr, override_rel, opts.rerank, True, log_file, rerank_file)
            t_trues, t_unch_loc_ranks, t_loc_gold_ranks, t_gold_reranked, t_gold_ergs, t_pls, change_idx = \
                node_loop(change_idx, ergm, rel, node, t_assoc_cache,
                          caches, tr_gr, te_gr, override_rel, opts.rerank, False, log_file, rerank_file)

            total_trues = s_trues + t_trues
            insts[rel] += (total_trues)
            if override_rel:
                overrides[rel] += total_trues

            ulr = s_unch_loc_ranks + t_unch_loc_ranks
            lgr = s_loc_gold_ranks + t_loc_gold_ranks
            grr = s_gold_reranked + t_gold_reranked
            ger = s_gold_ergs + t_gold_ergs
            total_misses[rel] += (len(ulr))

            pre_ranks.extend(lgr)
            if override_rel:
                erg_ranks.extend(lgr)
                all_ranks.extend(lgr)
            else:
                all_ranks.extend(ulr + grr)
                erg_ranks.extend(ulr + ger)

            for pl in s_pls + t_pls:
                if pl[3] < pl[2]:
                    rerank_ups[rel] += 1
                if pl[3] > pl[2]:
                    rerank_downs[rel] += 1
                if pl[4] < pl[2]:
                    erg_ups[rel] += 1
                if pl[4] > pl[2]:
                    erg_downs[rel] += 1
                rerank_diff[rel] += (pl[2] - pl[3])
                erg_diff[rel] += (pl[2] - pl[4])

        rel_all_ranks[rel] = all_ranks
        rel_pre_ranks[rel] = pre_ranks
        rel_erg_ranks[rel] = erg_ranks

        all_all_ranks.extend(all_ranks)
        all_pre_ranks.extend(pre_ranks)
        all_erg_ranks.extend(erg_ranks)

    if writing:
        log_file.write('\nper relation:\n')
        for rel in list(graphs.keys()):
            if insts[rel] > 0 and insts[rel] - total_misses[rel] > 0:
                log_file.write('\n{}:\n'.format(rel))
                log_file.write('{} instances, {} misses\n'.format(insts[rel], total_misses[rel]))
                log_file.write('reranks: {} up, {} down\n'.format(rerank_ups[rel], rerank_downs[rel]))
                log_file.write('ERGM only: {} up, {} down\n'.format(erg_ups[rel], erg_downs[rel]))
                log_file.write('rank diff: {}, ERGM only: {}\n'.format(rerank_diff[rel], erg_diff[rel]))
                log_file.write('metrics: pre-rank\trerank\tERGM only\n')
                log_file.write('average rank: {:.5f}\t{:.5f}\t{:.5f}\n'.format(np.average(rel_pre_ranks[rel]),
                                                                               np.average(rel_all_ranks[rel]),
                                                                               np.average(rel_erg_ranks[rel])))
                log_file.write('mrr: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mrr(rel_pre_ranks[rel]), mrr(rel_all_ranks[rel]),
                                                                      mrr(rel_erg_ranks[rel])))
                log_file.write(
                    'mq: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mq(rel_pre_ranks[rel], N), mq(rel_all_ranks[rel], N),
                                                          mq(rel_erg_ranks[rel], N)))
                log_file.write('h@100: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(rel_pre_ranks[rel], n=100),
                                                                        h_at_n(rel_all_ranks[rel], n=100),
                                                                        h_at_n(rel_erg_ranks[rel], n=100)))
                log_file.write(
                    'h@10: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(rel_pre_ranks[rel]), h_at_n(rel_all_ranks[rel]),
                                                            h_at_n(rel_erg_ranks[rel])))
                log_file.write('h@1: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(rel_pre_ranks[rel], n=1),
                                                                      h_at_n(rel_all_ranks[rel], n=1),
                                                                      h_at_n(rel_erg_ranks[rel], n=1)))

        log_file.write('\ntotals:\n')
        log_file.write('total number of instances: {}\n'.format(sum(insts.values())))
        log_file.write('total misses: {}\n'.format(sum(total_misses.values())))
        log_file.write('overrides: {}\n'.format(sum(overrides.values())))
        log_file.write(
            'rerank improvements: {}; regressions: {}\n'.format(sum(rerank_ups.values()), sum(rerank_downs.values())))
        log_file.write(
            'only ERGM improvements: {}; regressions: {}\n'.format(sum(erg_ups.values()), sum(erg_downs.values())))
        log_file.write(
            'total rank diffs: rerank {}, only ERGM {}\n'.format(sum(rerank_diff.values()), sum(erg_diff.values())))

        log_file.write('metrics: pre-rank\trerank\tERGM only\n')
        log_file.write(
            'average rank: {:.5f}\t{:.5f}\t{:.5f}\n'.format(np.average(all_pre_ranks), np.average(all_all_ranks),
                                                            np.average(all_erg_ranks)))
        log_file.write(
            'mrr: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mrr(all_pre_ranks), mrr(all_all_ranks), mrr(all_erg_ranks)))
        log_file.write(
            'mq: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mq(all_pre_ranks, N), mq(all_all_ranks, N), mq(all_erg_ranks, N)))
        log_file.write(
            'h@100: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(all_pre_ranks, n=100), h_at_n(all_all_ranks, n=100),
                                                     h_at_n(all_erg_ranks, n=100)))
        log_file.write('h@10: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(all_pre_ranks), h_at_n(all_all_ranks),
                                                               h_at_n(all_erg_ranks)))
        log_file.write('h@1: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(all_pre_ranks, n=1), h_at_n(all_all_ranks, n=1),
                                                              h_at_n(all_erg_ranks, n=1)))

    print('number of instances:', sum(insts.values()))
    print('total misses:', sum(total_misses.values()))
    print('overrides:', sum(overrides.values()))
    print('average rank:', np.average(all_all_ranks))
    print('mrr: {:.4f}'.format(mrr(all_all_ranks)))
    print('mq:', mq(all_all_ranks, N))
    print('h@100: {:.5f}'.format(h_at_n(all_all_ranks, n=100)))
    print('h@10: {:.5f}'.format(h_at_n(all_all_ranks)))
    print('h@1: {:.5f}'.format(h_at_n(all_all_ranks, n=1)))

    return mrr(all_all_ranks), h_at_n(all_all_ranks, n=10), h_at_n(all_all_ranks, n=3), h_at_n(all_all_ranks, n=1)


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument("--input", required=True, help=".pkl file with WordNet eval dataset")
    parser.add_argument("--embeddings", help="pretrained synset embeddings")
    parser.add_argument("--model", help="pretrained model file (optional)")
    parser.add_argument("--model-only-init", action='store_true',
                        help="pretrained model file has only association features")
    parser.add_argument("--ergm-model", help="pretrained ERGM model file (optional)")
    parser.add_argument("--model-out", help="destination for model file (optional; only if none is loaded)")
    parser.add_argument("--rerank-out", help="output file for reranker training")
    parser.add_argument("--v", type=int, default=0)  # verbosity
    parser.add_argument("--debug", action='store_true')

    # general setup
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--emb-size", type=int, default=-1)
    parser.add_argument("--eval-dev", action='store_true', help="evaluate on dev set, not test")
    parser.add_argument("--assoc-mode", default=TRANSLATIONAL_EMBED_MODE,
                        help="Association mode. Options: {}, default: {}".format(MODES_STR, TRANSLATIONAL_EMBED_MODE))

    # training set engineering
    parser.add_argument("--co-hypernyms", action='store_true', help="include co-hypernym graph for scores")
    parser.add_argument("--skip-symmetrics", action='store_true', help="skip symmetric relations in ERGM training")
    parser.add_argument("--rand-nodes", type=bool, default=True, help="randomize each relation's nodes in training")
    parser.add_argument("--rand-all", action='store_true', help="randomize all nodes in training, across relations")

    # training params
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--regularize", type=float, default=0.1)
    parser.add_argument("--neg-samp", type=int, default=M3GM_DEFAULT_NEGS, help="number of negative samples")
    parser.add_argument("--no-assoc-bp", action='store_true', help="do not backprop into association model")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout for association model only, set to 0.0 in no-assoc-bp mode")

    # testing
    parser.add_argument("--rule-override", type=bool, default=True, help="rule-based override for symmetric relations")
    parser.add_argument("--rerank", type=int, default=100, help="number of top results to rerank")
    opts = parser.parse_args()

    # init
    start_time = datetime.now()

    # reporting
    timeprint('graphs file = {}'.format(opts.input))
    timeprint('embeddings file = {}'.format(opts.embeddings if opts.embeddings is not None \
                                                else 'of size {}'.format(opts.emb_size)))
    timeprint('association mode = {}'.format(opts.assoc_mode))
    timeprint('reranker output file = {}'.format(opts.rerank_out))
    if opts.model is None:
        timeprint('model output file = {}, only init = {}'.format(opts.model_out, opts.model_only_init))
        timeprint('epochs = {}'.format(opts.epochs))
        timeprint('Adagrad learning rate = {}'.format(opts.learning_rate))
        timeprint('neg-samp = {}'.format(opts.neg_samp))
        timeprint('rand-nodes = {}'.format(opts.rand_nodes))
        timeprint('dropout = {}'.format(opts.dropout))
        timeprint('regularizer labmda = {}'.format(opts.regularize))
    else:
        timeprint('model file = {}, ergm model file = {}'.format(opts.model, opts.ergm_model))
        if opts.ergm_model is not None:
            timeprint('test mode only')
        else:
            if opts.no_assoc_bp:
                timeprint('association model not backpropped into')
            timeprint('epochs = {}'.format(opts.epochs))
            timeprint('neg-samp = {}'.format(opts.neg_samp))
            timeprint('regularizer labmda = {}'.format(opts.regularize))
            timeprint('model output file = {}'.format(opts.model_out))
    if opts.skip_symmetrics:
        timeprint('training ERGM without symmetric relation iterations')
    timeprint('rerank list length = {}'.format(opts.rerank))
    if opts.co_hypernyms:
        timeprint('adding co-hypernym graph')
    if opts.rule_override:
        timeprint('using symmetricity rule override for testing')
    if opts.eval_dev:
        timeprint('evaluating dev set')
    else:
        timeprint('evaluating test set')

    # load dataset
    ds, synsets = load_prediction_dataset(opts.input)
    N = len(synsets)  # graph size
    idx_diffs = [(t, d, te) for t, d, te in zip(synsets, ds.dev.index, ds.test.index) if t != d or t != te]
    assert len(idx_diffs) == 0
    s2i = {s: i for i, s in enumerate(synsets)}

    # get synset embeddings
    timeprint('loading embeddings...')
    embs = load_embeddings(opts.embeddings, s2i, opts.emb_size)

    # training phase
    # collect graphs (+ dev if we're in final test mode)
    if opts.eval_dev:
        tr_graphs = ds.train.matrices
    else:
        tr_graphs = join_sets([ds.train.matrices, ds.dev.matrices])

    te_graphs = ds.dev.matrices if opts.eval_dev \
        else ds.test.matrices

    if opts.co_hypernyms:
        tr_graphs['co_hypernym'] = co_graph(tr_graphs['hypernym'])

    drop = opts.dropout if not opts.no_assoc_bp else 0.0
    dev_results = None
    if opts.model is not None and opts.ergm_model is None and not opts.model_only_init:
        exit('not clear about this run mode. did you mean {} as an ergm-model arg?'.format(opts.model))
    if opts.ergm_model is not None and not opts.model_only_init:
        # load and skip training
        timeprint('loading ERGM from file: {}'.format(opts.ergm_model))
        ergm = MultiGraphErgm(tr_graphs, embs, opts.assoc_mode, ergm_path=opts.ergm_model)
    elif opts.model is not None and opts.ergm_model is not None:
        # load and skip training
        timeprint('loading association from file: {}'.format(opts.model))
        timeprint('loading ERGM from file: {}'.format(opts.ergm_model))
        ergm = MultiGraphErgm(tr_graphs, embs, opts.assoc_mode, reg=opts.regularize, dropout=drop,
                              model_path=opts.model,
                              path_only_init=True,
                              ergm_path=opts.ergm_model)
    else:
        dev_results = []
        # training phase
        if opts.model is not None:  # there's a pretrained association model
            ergm = MultiGraphErgm(tr_graphs, embs, opts.assoc_mode, reg=opts.regularize,
                                  dropout=drop, model_path=opts.model,
                                  path_only_init=True)
        else:
            ergm = MultiGraphErgm(tr_graphs, embs, opts.assoc_mode, reg=opts.regularize,
                                  dropout=drop)
        initial_weights = ergm.ergm_weights.as_array()
        trainer = dy.AdagradTrainer(ergm.model, opts.learning_rate)
        iteration_scores = []
        log_file_name = 'pred-train-log-{}_{}.txt'.format(start_time.date(), start_time.time())
        timeprint('starting training phase, writing to {}'.format(log_file_name))
        with open(log_file_name, 'a') as log_file:
            log_file.write('====\n')
            for ep in range(opts.epochs):
                iteration_scores.extend(macro_loops(opts, ep + 1, ergm, trainer, log_file, synsets))
                if opts.eval_dev and ep < opts.epochs - 1:
                    dev_results.append(eval(tr_graphs, te_graphs, ergm, opts, N, log_file=None, rerank_file=None))
        if opts.model_out is not None:
            # save model
            timeprint('saving trained model to {}'.format(opts.model_out))
            ergm.save(opts.model_out, initial_weights)
        print('scores:', '\t'.join([str(sc) for sc in iteration_scores[::100]]))

    # dev/test
    test_file_name = 'pred-{}-log-{}_{}.txt'.format('dev' if opts.eval_dev else 'test',
                                                    start_time.date(), start_time.time())
    with (open(opts.rerank_out, 'a') if opts.rerank_out is not None else none_context) as rerank_file:
        timeprint('starting inference phase, writing to {}'.format(test_file_name))
        if rerank_file is not None:
            rerank_file.write('rel\tidx\tis_source\tnode\tprediction\tis_gold\tassoc_score\tergm_score\n')
        with open(test_file_name, 'a') as log_file:
            if dev_results is not None and len(dev_results) > 0:
                log_file.write('mrr,h@10,h@3,h@1\n' + '\n'.join([str(rrs) for rrs in dev_results]) + '\n')
            ress = eval(tr_graphs, te_graphs, ergm, opts, N, log_file, rerank_file)
            if dev_results is not None:
                dev_results.append(ress)

    if dev_results is not None:
        print('epoch dev results:')
        print('\n'.join(['\t'.join([str(rs) for rs in rrs]) for rrs in dev_results]))
