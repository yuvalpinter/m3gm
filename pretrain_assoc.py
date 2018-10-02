import dynet as dy
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime

from consts import ASSOC_DEFAULT_NEGS, SYMMETRIC_RELATIONS
from io_utils import timeprint, load_prediction_dataset, load_embeddings
from math_utils import dyagonalize
from multigraph_utils import canonicalize_name, targets, sources, join_sets
from metrics import h_at_n, mrr, mq

__author__ = "Yuval Pinter, 2018"

BILINEAR_MODE = 'bilin'
DISTMULT = 'distmult'
DIAG_RANK1_MODE = 'diag_r1'
TRANSLATIONAL_EMBED_MODE = 'transE'

MODES = [BILINEAR_MODE, DISTMULT, DIAG_RANK1_MODE, TRANSLATIONAL_EMBED_MODE]
MODES_STR = ', '.join(MODES)


class AssociationModel:
    """
    Structure for training a relation prediction model for inferring semantic graph relations
    In the context of this project, it will be used to pre-train the association component
    for later use in a MultiGraphErgm.
    """
    def __init__(self, graphs, embeddings, mode=TRANSLATIONAL_EMBED_MODE, dropout=0.0, model_path=None):
        """
        :param graphs: dictionary of <relation:CSR-format graph>s, node-aligned
        :param embeddings: list of numpy array embeddings, indices aligned to nodes
        :param mode: mode of calculating association score, options: {}
        """.format(MODES_STR)
        # input validation
        graph_sizes = list(set([g.shape[0] for g in list(graphs.values())]))
        assert len(graph_sizes) == 1
        assert len(embeddings) == graph_sizes[0], '{} != {}'.format(len(embeddings), graph_sizes[0])
        
        # raw members
        self.graphs = {canonicalize_name(k):g for k,g in list(graphs.items())}
        self.mode = mode
        
        # documenationy members
        self.relation_names = sorted(self.graphs.keys())
        if 'co_hypernym' in self.relation_names:
            self.relation_names.remove('co_hypernym')
        self.vocab_size = graph_sizes[0]
        self.R = len(self.relation_names)
        self.emb_dim = len(embeddings[0])
        self.dropout = dropout

        # model members
        self.model = dy.Model()
        # TODO consider using no_update param for embeddings
        self.embeddings = self.model.add_lookup_parameters((self.vocab_size, self.emb_dim))
        self.embeddings.init_from_array(embeddings)
        
        # init association parameter
        self.no_assoc = False # so can be overriden in inheritors
        
        # first determine 
        if self.mode == BILINEAR_MODE:              # full-rank bilinear matrix
            assoc_dim = (self.emb_dim, self.emb_dim)
        elif self.mode == DIAG_RANK1_MODE:          # diagonal bilinear matrix + rank 1 matrix
            # first row = diagonal
            # second row = 'source factor'
            # third row = 'target factor'
            assoc_dim = (3, self.emb_dim)
        elif self.mode == TRANSLATIONAL_EMBED_MODE: # additive relational vector
            assoc_dim = self.emb_dim
        elif self.mode == DISTMULT:                 # diagonal bilinear matrix
            assoc_dim = self.emb_dim
        else:
            raise ValueError('unsupported mode: {}. allowed are {}'\
                             .format(self.mode, ', '.join(MODES_STR)))
            
        # init actual parameter
        self.word_assoc_weights = {r:self.model.add_parameters(assoc_dim) for r in self.relation_names}
        if model_path is not None:
            self.model.populate(model_path + '.dyn')
        
        timeprint('finished initialization for association model.')
        
    def word_assoc_score(self, source_idx, target_idx, relation):
        """
        NOTE THAT DROPOUT IS BEING APPLIED HERE
        :param source_idx: embedding index of source atom
        :param target_idx: embedding index of target atom
        :param relation: relation type
        :return: score
        """
        # prepare
        s = self.embeddings[source_idx]
        if self.no_assoc:
            A = dy.const_parameter(self.word_assoc_weights[relation])
        else:
            A = dy.parameter(self.word_assoc_weights[relation])
        dy.dropout(A, self.dropout)
        t = self.embeddings[target_idx]
        
        # compute
        if self.mode == BILINEAR_MODE:
            return dy.transpose(s) * A * t
        elif self.mode == DIAG_RANK1_MODE:
            diag_A = dyagonalize(A[0])
            rank1_BC = A[1] * dy.transpose(A[2])
            ABC = diag_A + rank1_BC
            return dy.transpose(s) * ABC * t
        elif self.mode == TRANSLATIONAL_EMBED_MODE:
            return -dy.l2_norm(s - t + A)
        elif self.mode == DISTMULT:
            return dy.sum_elems(dy.cmult(dy.cmult(s, A), t))
    
    def source_ranker_cache(self, rel):
        """
        test mode only (no updates, no dropout)
        :param rel: relation to create cache for quick score calculation once source is given
        :return: mode-appropriate pre-computation for association scores
        """
        T = self.embeddings.as_array()
        A = self.word_assoc_weights[rel].as_array()
        if self.mode == BILINEAR_MODE:
            return A.dot(T.transpose())
        elif self.mode == DIAG_RANK1_MODE:
            diag_A = np.diag(A[0])
            rank1_BC = np.outer(A[1],A[2])
            ABC = diag_A + rank1_BC
            return ABC.dot(T.transpose())
        elif self.mode == TRANSLATIONAL_EMBED_MODE:
            return A - T
        elif self.mode == DISTMULT:
            return A * T # elementwise, broadcast

    def target_ranker_cache(self, rel):
        """
        test mode only (no updates, no dropout)
        :param rel: relation to create cache for quick score calculation once target is given
        :returns: mode-appropriate pre-computation for association scores
        """
        S = self.embeddings.as_array()
        A = self.word_assoc_weights[rel].as_array()
        if self.mode == BILINEAR_MODE:
            return S.dot(A)
        elif self.mode == DIAG_RANK1_MODE:
            diag_A = np.diag(A[0])
            rank1_BC = np.outer(A[1],A[2])
            ABC = diag_A + rank1_BC
            return S.dot(ABC)
        elif self.mode == TRANSLATIONAL_EMBED_MODE:
            return S + A
        elif self.mode == DISTMULT:
            return S * A # elementwise, broadcast
    
    def score_from_source_cache(self, cache, src):
        """
        test mode only (no updates, no dropout)
        :param cache: cache computed earlier using source_ranker_cache
        :param src: index of source node to create ranking of all targets for
        :return: array of scores for all possible targets
        """
        s = self.embeddings[src].npvalue()
        if self.mode == BILINEAR_MODE:
            return (s.dot(cache)).transpose()
        elif self.mode == DIAG_RANK1_MODE:
            return (s.dot(cache)).transpose()
        elif self.mode == TRANSLATIONAL_EMBED_MODE:
            diff_vecs = s + cache
            return -np.sqrt((diff_vecs * diff_vecs).sum(1))
        elif self.mode == DISTMULT:
            return cache.dot(s)
    
    def score_from_target_cache(self, cache, trg):
        """
        test mode only (no updates, no dropout)
        :param cache: cache computed earlier using target_ranker_cache
        :param trg: index of target node to create ranking of all sources for
        :return: array of scores for all possible sources
        """
        t = self.embeddings[trg].npvalue()
        if self.mode == BILINEAR_MODE:
            return cache.dot(t)
        elif self.mode == DIAG_RANK1_MODE:
            return cache.dot(t)
        elif self.mode == TRANSLATIONAL_EMBED_MODE:
            diff_vecs = cache - t
            return -np.sqrt((diff_vecs * diff_vecs).sum(1)) #[-np.linalg.norm(s) for s in t + cache]
        elif self.mode == DISTMULT:
            return cache.dot(t)
    
    def save(self, filename):
        self.model.save(filename)


def train_iteration(opts, assoc_model, trainer, do_sym, log_file):
    """
    Setup where association scores are learned, relation by relation.
    based on `model.macro_loops()`
    :return: full-graph iteration scores
    """
    ep_loss = 0.0
    
    # iterate over relations:
    graphs_order = list(assoc_model.graphs.items())
    # TODO maybe even randomize edge order across relations
    if opts.rand_nodes:
        dy.np.random.shuffle(graphs_order)
    for rel, g in graphs_order:
        # report
        if opts.v > 0:
            timeprint('starting loop over {}'.format(rel))

        if opts.rule_override and rel in SYMMETRIC_RELATIONS and not do_sym:
            timeprint('RELATION OVERRIDEN')
            continue
        
        # iterate over nodes (each as source + as target):
        node_order = list(range(N))
        if opts.rand_nodes:
            dy.np.random.shuffle(node_order)
        for node in tqdm(node_order):
            if opts.debug and node % 100 != 0:
                continue
            ep_loss += node_iteration(rel, g, node, opts, assoc_model, trainer, log_file, is_source=True)
            ep_loss += node_iteration(rel, g, node, opts, assoc_model, trainer, log_file, is_source=False)

    return ep_loss


def node_iteration(rel, g, node, opts, assoc_model, trainer, log_file, is_source):
    """
    Perform one iteration of trying to score a node's neighbors above negative samples.
    """
    
    # true instances likelihood
    trues = targets(g, node) if is_source else sources(g, node)
    side = '->' if is_source else '<-'
    if len(trues) == 0: return 0.0
    
    if opts.debug:
        dy.renew_cg(immediate_compute = True, check_validity = True)
    else:
        dy.renew_cg()
    
    # compute association score as dynet expression (can't do this above due to staleness)
    true_scores = []
    for tr in trues:
        if is_source:
            j_assoc_score = assoc_model.word_assoc_score(node, tr, rel)
        else:
            j_assoc_score = assoc_model.word_assoc_score(tr, node, rel)
        if log_file is not None:
            log_file.write('{} {}\tTRUE_{}\t{:.3e}\n'\
                         .format(node, side, tr, j_assoc_score.scalar_value()))
        true_scores.append(j_assoc_score)


    # false targets likelihood - negative sampling (uniform)
    # collect negative samples
    if opts.nll:
        sample_scores = [[ts] for ts in true_scores]
    else:
        margins = []
    neg_samples = [np.random.choice(range(N)) for _ in range(opts.neg_samp * len(trues))]
    # remove source and true targets if applicable
    for t in [node] + trues:
        if t in neg_samples:
            neg_samples.remove(t)
            neg_samples.append(np.random.choice(range(N)))
    for (i,ns) in enumerate(neg_samples):
        # compute association score as dynet expression
        if is_source:
            ns_assoc_score = assoc_model.word_assoc_score(node, ns, rel)
        else:
            ns_assoc_score = assoc_model.word_assoc_score(ns, node, rel)
        if log_file is not None:
            log_file.write('{} {}\tNEG_{}\t{:.3e}\n'\
                         .format(node, side, ns, ns_assoc_score.scalar_value()))
        corresponding_true = i // opts.neg_samp
        if opts.nll:
            sample_scores[corresponding_true].append(ns_assoc_score)
        else:
            # TODO maybe use dy.hinge()
            ctt_score = true_scores[corresponding_true]
            margin = ctt_score - ns_assoc_score
            margins.append(dy.rectify(dy.scalarInput(1.0) - margin))


    # compute overall loss
    if opts.nll:
        if len(sample_scores) == 0:
            dy_loss = dy.scalarInput(0.0)
        else:
            dy_loss = dy.esum([dy.pickneglogsoftmax(dy.concatenate(scrs), 0) for scrs in sample_scores])
    else:
        if len(margins) == 0:
            dy_loss = dy.scalarInput(0.0)
        else:
            dy_loss = dy.esum(margins)
    sc_loss = dy_loss.scalar_value()
    if log_file is not None:
        log_file.write('{}\tLOSS\t{:.3e}\n'\
                         .format(node, sc_loss))
                         
    # backprop and recompute score
    if opts.v > 1:
        timeprint('overall loss for relation {}, node {} as {} = {:.6f}'\
                  .format(rel, node, 'source' if is_source else 'target', sc_loss))

    dy_loss.backward()
    trainer.update()

    return sc_loss


def eval(assoc_model, tr_graphs, te_graphs, opts, N, log_file=None):

    all_t_ranks = []
    all_s_ranks = []
    insts = 0
    for rel,te_gr in list(te_graphs.items()):
        if log_file is not None:
            log_file.write('relation: {}\n'.format(rel))
        # add incrementally, eval each edge, revert
        g = assoc_model.graphs[rel]
        s_assoc_cache = assoc_model.source_ranker_cache(rel)
        for n in range(N):
            # node as source
            # collect gold data for source node
            gold_targs = targets(te_gr, n)
            if len(gold_targs) == 0: continue
            insts += len(gold_targs)
            
            # rank based on assocation score
            known_targs = targets(tr_graphs[rel], n)
            sym_targs = sources(tr_graphs[rel], n)
            all_ignored = known_targs + sym_targs + gold_targs + [n]
            s_assoc_scores = assoc_model.score_from_source_cache(s_assoc_cache, n)
            n_ranks = []
            for g in gold_targs:
                if opts.rule_override and rel in SYMMETRIC_RELATIONS:
                    if g in sym_targs:
                        n_ranks.append(1)
                        continue
                g_score = s_assoc_scores[g]
                rank = 1 + len([v for i,v in enumerate(s_assoc_scores) if v > g_score\
                                                                     and i not in all_ignored])
                n_ranks.append(rank)
                
            if log_file is not None:
                log_file.write('targets for source {}:{} found in ranked places {}\n'\
                            .format(n,synsets[n],n_ranks))
                            
            all_t_ranks.extend(n_ranks)

        t_assoc_cache = assoc_model.target_ranker_cache(rel)
        for n in range(N):
            # node as target - same
            # collect gold data for target node
            gold_srcs = sources(te_gr, n)
            if len(gold_srcs) == 0: continue
            insts += len(gold_srcs)
            
            # rank based on assocation score
            known_srcs = sources(tr_graphs[rel], n)
            sym_srcs = targets(tr_graphs[rel], n)
            all_ignored = known_srcs + sym_srcs + gold_srcs + [n]
            t_assoc_scores = assoc_model.score_from_target_cache(t_assoc_cache, n)
            n_ranks = []
            for g in gold_srcs:
                if opts.rule_override and rel in SYMMETRIC_RELATIONS:
                    if g in sym_srcs:
                        n_ranks.append(1)
                        continue
                g_score = t_assoc_scores[g]
                rank = 1 + len([v for i,v in enumerate(t_assoc_scores) if v > g_score\
                                                                     and i not in all_ignored])
                n_ranks.append(rank)
            
            if log_file is not None:
                log_file.write('sources for target {}:{} found in ranked places {}\n'\
                            .format(n,synsets[n],n_ranks))
            all_s_ranks.extend(n_ranks)
    
    return insts, all_s_ranks, all_t_ranks


if __name__ == '__main__':
    # parse params
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".pkl file with WordNet eval dataset")
    parser.add_argument("--embeddings", help="pretrained synset embeddings")
    parser.add_argument("--assoc-mode", default=BILINEAR_MODE,
                        help="Association mode. Options: {}, default: {}".format(MODES_STR, BILINEAR_MODE))
    parser.add_argument("--model", help="pretrained model file (optional), no training will happen")
    parser.add_argument("--model-out", help="destination for model file (optional; only if none is loaded)")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--emb-size", type=int, default=-1, help="dimension of embeddings (-1 - from input)")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--neg-samp", type=int, default=ASSOC_DEFAULT_NEGS, help="nodes for negative sample")
    parser.add_argument("--rand-nodes", action="store_false", help="randomize node order in training")
    parser.add_argument("--rule-override", action="store_false", help="rule-based override for symmetric relations")
    parser.add_argument("--eval-dev", action='store_true', help="evaluate on dev set (otherwise - test)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nll", action='store_true', help="use negative log likelihood loss")
    parser.add_argument("--no-log", action='store_true')
    parser.add_argument("--early-stopping", action='store_true', help="stop if model hasn't improved in 3 epochs")
    parser.add_argument("--v", type=int, default=1, help="verbosity")
    parser.add_argument("--debug", action='store_true')
    opts = parser.parse_args()
    
    start_time = datetime.now()
    
    # reporting
    timeprint('graphs file = {}'.format(opts.input))
    if opts.embeddings is not None:
        timeprint('embeddings file = {}'.format(opts.embeddings))
    else:
        timeprint('embeddings size = {}'.format(opts.emb_size))
    timeprint('association mode = {}'.format(opts.assoc_mode))
    timeprint('negative samples = {}'.format(opts.neg_samp))
    if opts.model is None:
        timeprint('model file = {}'.format(opts.model_out))
        if opts.nll:
            timeprint('using negative log likelihood loss')
        else:
            timeprint('using margin loss')
        if opts.no_log:
            timeprint('no log file. timestamp for test: {}_{}' \
                      .format(start_time.date(), start_time.time()))
        if opts.early_stopping:
            timeprint('employing early stopping (over last 2 epochs)')
        timeprint('dropout = {}'.format(opts.dropout))
        timeprint('max epochs = {}'.format(opts.epochs))
        timeprint('Adagrad learning rate = {}'.format(opts.learning_rate))
        timeprint('rand-nodes = {}'.format(opts.rand_nodes))
    else:
        timeprint('test mode only')
        timeprint('model file = {}'.format(opts.model))
    if opts.rule_override:
        timeprint('overriding symmetric relations with rule-based system')
    if opts.eval_dev:
        timeprint('evaluating dev set')
    else:
        timeprint('evaluating test set')
    
    # load dataset
    ds, synsets = load_prediction_dataset(opts.input)
    N = len(synsets) # graph size
    idx_diffs = [(t,d,te) for t,d,te in zip(synsets,ds.dev.index,ds.test.index) if t != d or t != te ]
    assert len(idx_diffs) == 0
    s2i = {s:i for i,s in enumerate(synsets)}

    # get synset embeddings
    timeprint('loading embeddings...')
    embs = load_embeddings(opts.embeddings, s2i, opts.emb_size)
    
    # train
    # models evaluated on test add dev edges to their training set
    tr_graphs = ds.train.matrices if opts.eval_dev\
                else join_sets([ds.train.matrices,
                                ds.dev.matrices])
    te_graphs = ds.dev.matrices if opts.eval_dev \
                                else ds.test.matrices
    
    if opts.model is not None:
        # load and skip training (eval mode)
        timeprint('loading association model from file: {}'.format(opts.model))
        assoc_model = AssociationModel(tr_graphs, embs, opts.assoc_mode, model_path=opts.model)
    else:
        # training phase
        assoc_model = AssociationModel(tr_graphs, embs, opts.assoc_mode, opts.dropout)
        trainer = dy.AdagradTrainer(assoc_model.model, opts.learning_rate)
        with open('assoc-pred-train-log-{}_{}.txt'.format(start_time.date(), start_time.time()), 'a') as log_file:
            if opts.no_log:
                log_file = None
            else:
                log_file.write('====\n')
            iteration_losses = [] # will hold loss averages
            dev_mrrs = []
            saved_name = None
            
            N = assoc_model.vocab_size
            for ep in range(opts.epochs):
                # report
                if opts.v > 0:
                    timeprint('starting epoch {}'.format(ep + 1)) 
                iteration_losses.append(train_iteration(opts, assoc_model, trainer, ep % 5 == 4, log_file))
                if opts.early_stopping:
                    timeprint('evaluating after epoch {}'.format(ep+1))
                    insts, all_s_ranks, all_t_ranks = eval(assoc_model, tr_graphs, te_graphs, opts, N)
                    # save model with epoch count and remove previous if exists
                    ep_mrr = mrr(all_s_ranks + all_t_ranks)
                    ep_h10 = h_at_n(all_s_ranks + all_t_ranks)
                    ep_h1 = h_at_n(all_s_ranks + all_t_ranks, n=1)
                    timeprint('mrr: {:.4f}, h@10: {:.4f}, h@1: {:.4f}'.format(ep_mrr, ep_h10, ep_h1))
                    if len(dev_mrrs) < 1 or ep_mrr > min(dev_mrrs[-2:]):
                        if len(dev_mrrs) < 1 or ep_mrr > max(dev_mrrs):
                            best_insts = insts
                            best_all_s_ranks = all_s_ranks
                            best_all_t_ranks = all_t_ranks
                            last_saved_name = saved_name
                            saved_name = '{}-ep-{:02d}.dyn'.format(opts.model_out, ep + 1)
                            timeprint('saving trained model to {}'.format(saved_name))
                            assoc_model.save(saved_name)
                            # remove previous model(s)
                            if last_saved_name is not None:
                                os.remove(last_saved_name)
                    else: break
                    dev_mrrs.append(ep_mrr)
                
        print('training losses:', '\t'.join([str(sc) for sc in iteration_losses]))
    
    # report
    with open('assoc-pred-{}-log-{}_{}.txt'.format('dev' if opts.eval_dev else 'test',
                                                   start_time.date(), start_time.time()), 'a') as log_file:
                                             
        if opts.model_out is None:
            # eval on dev using pre-loaded model
            best_insts, best_all_s_ranks, best_all_t_ranks = eval(assoc_model, tr_graphs, te_graphs, opts, N, log_file)
        elif not opts.early_stopping:
            # save model, eval on dev
            timeprint('saving trained model to {}'.format(opts.model_out))
            assoc_model.save(opts.model_out + '.dyn')
            best_insts, best_all_s_ranks, best_all_t_ranks = eval(assoc_model, tr_graphs, te_graphs, opts, N, log_file)
        
        best_all_ranks = best_all_s_ranks + best_all_t_ranks
        log_file.write('total number of instances: {}\n'.format(best_insts))
        log_file.write('average rank: source {:.2f}\ttarget {:.2f}\toverall {:.2f}\n'\
                       .format(np.average(best_all_s_ranks), np.average(best_all_t_ranks), np.average(best_all_ranks)))
        log_file.write('mrr: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mrr(best_all_s_ranks), mrr(best_all_t_ranks), mrr(best_all_ranks)))
        log_file.write('mq: {:.4f}\t{:.4f}\t{:.4f}\n'.format(mq(best_all_s_ranks, N), mq(best_all_t_ranks, N), mq(best_all_ranks, N)))
        log_file.write('h@100: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(best_all_s_ranks, n=100), h_at_n(best_all_t_ranks, n=100), h_at_n(best_all_ranks, n=100)))
        log_file.write('h@10: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(best_all_s_ranks), h_at_n(best_all_t_ranks), h_at_n(best_all_ranks)))
        log_file.write('h@1: {:.5f}\t{:.5f}\t{:.5f}\n'.format(h_at_n(best_all_s_ranks, n=1), h_at_n(best_all_t_ranks, n=1), h_at_n(best_all_ranks, n=1)))
    
    print('number of instances:', best_insts)
    print('average rank:', np.average(best_all_ranks))
    print('mrr: {:.4f}'.format(mrr(best_all_ranks)))
    print('mq:', mq(best_all_ranks, N))
    print('h@100: {:.5f}'.format(h_at_n(best_all_ranks, n=100)))
    print('h@10: {:.5f}'.format(h_at_n(best_all_ranks)))
    print('h@1: {:.5f}'.format(h_at_n(best_all_ranks, n=1)))
