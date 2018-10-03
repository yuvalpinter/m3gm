import sys
import numpy as np
import pickle as pickle
from itertools import combinations
import dynet as dy
import copy

from multigraph_utils import tupstr, csr_eye, cyclic_triads, find_key, \
                             find_cyclic_key
from ergm_feats import edge_count, mutual_edges, \
                       one_rel_star_counts, two_rel_star_counts, \
                       three_rel_star_counts
from io_utils import timeprint
from pretrain_assoc import AssociationModel, BILINEAR_MODE

__author__ = "Yuval Pinter, 2018"

class MultiGraphErgm(AssociationModel):
    """
    Structure maintaining large graphs (10^6+ nodes) with multiple relation types
    where ERGM features are computed regularly in single-node-addition scenarios.
    A cache is maintained to hasten feature updates across all relation combinations.
    """
    def __init__(self, graphs, embeddings, assoc_mode=BILINEAR_MODE, reg=0.0, dropout=0.0,
                 no_assoc=False, model_path=None, ergm_path=None,
                 path_only_init=False):
        """
        :param graphs: dictionary of {relation:CSR-format graph}s, node-aligned
        :param embeddings: list of numpy array embeddings, indices aligned to nodes
        :param model_path: optional path for files with pre-trained association model (read by super)
        :param ergm_path: optional path for files with pre-trained model
        :param path_only_init: model_path only used for initialization
        """
        # input validation
        AssociationModel.__init__(self, graphs, embeddings, assoc_mode, dropout, model_path=model_path)

        # raw members
        self.no_assoc = no_assoc
        self.regularize = reg

        # cache members
        self.cache = {}
        self.edge_counts = self.add_cache_dict('ec')                   # keys are single relations
        self.mutual_edge_counts = self.add_cache_dict('mec')           # keys are unordered relation pairs
        self.two_path_counts = self.add_cache_dict('tpc')              # keys are ordered relation pairs
        self.transitive_closure_counts = self.add_cache_dict('tcc')    # keys are ordered relation triplets
        self.directed_triangle_counts = self.add_cache_dict('dtc')     # keys are ordered relation triplets
        self.in_degs = self.add_cache_dict('ins')                      # keys are single relations, values are big lists
        self.out_degs = self.add_cache_dict('outs')                    # keys are single relations, values are big lists
        self.in_one_star_counts = self.add_cache_dict('i1sc')          # keys are single relations
        self.out_one_star_counts = self.add_cache_dict('o1sc')         # keys are single relations
        self.in_two_star_counts = self.add_cache_dict('i2sc')          # keys are unordered relation pairs
        self.out_two_star_counts = self.add_cache_dict('o2sc')         # keys are unordered relation pairs
        self.in_three_star_counts = self.add_cache_dict('i3sc')        # keys are unordered relation triplets
        self.out_three_star_counts = self.add_cache_dict('o3sc')       # keys are unordered relation triplets
        # 'at least k' stars - 'one/two/three plus'
        self.in_one_p_star_counts = self.add_cache_dict('i1psc')       # keys are single relations
        self.out_one_p_star_counts = self.add_cache_dict('o1psc')      # keys are single relations
        self.in_two_p_star_counts = self.add_cache_dict('i2psc')       # keys are unordered relation pairs
        self.out_two_p_star_counts = self.add_cache_dict('o2psc')      # keys are unordered relation pairs
        self.in_three_p_star_counts = self.add_cache_dict('i3psc')     # keys are unordered relation triplets
        self.out_three_p_star_counts = self.add_cache_dict('o3psc')    # keys are unordered relation triplets

        self.missing_node_indices = []          # updates during training (NOT SURE IF NEEDED)

        timeprint('computing ERGM features...')
        self.init_ergm_features()               # populates self.feature_vals
        timeprint('finished! computed {} features'.format(len(self.feature_vals)))
        timeprint('{} non-zero features'.format(np.count_nonzero(list(self.feature_vals.values()))))

        # documentationy again, for efficient updates
        encountered_features = list(self.feature_vals.keys()) # canonical ordering from now on
        
        if ergm_path is not None:
            ergm_model_path = ergm_path
        elif (model_path is not None) and (not path_only_init):
            ergm_model_path = model_path
        else:
            ergm_model_path = None
                
        if ergm_model_path is None:
            self.feature_set = encountered_features
        else:
            self.feature_set = pickle.load(open(ergm_model_path + '.feats'))
            assert sorted(self.feature_set) == sorted(encountered_features)
        
        if ergm_model_path is None:
            self.ergm_weights = self.model.add_parameters(len(self.feature_set))
        
        if model_path is None and ergm_model_path is None:
            # 'model_path is not None' is initialized in super()
            # TODO support other association modes (affects downstream)
            if self.no_assoc:
                self.word_assoc_weights = {r:self.model.add_parameters((self.emb_dim, self.emb_dim), init=dy.ConstInitializer(0.0)) for r in self.relation_names}
            else:
                self.word_assoc_weights = {r:self.model.add_parameters((self.emb_dim, self.emb_dim)) for r in self.relation_names}
        elif ergm_model_path is not None:
            pc = dy.ParameterCollection()
            dy.load(ergm_model_path + '.dyn', pc)
            pc_list = pc.parameters_list()
            i = 0
            self.ergm_weights = pc_list[i]
            if not path_only_init:
                self.word_assoc_weights = {}
                rel_order = self.relation_names
                for r in rel_order:
                    i += 1
                    self.word_assoc_weights[r] = pc_list[i]
                i += 1
                assert i == len(pc_list),\
                       '{} relation params read but length is {}'.format(i, len(pc_list))
        
        self.dy_score = self.ergm_score()
        self.score = self.dy_score.scalar_value()

        self.score_is_stale = False

        timeprint('finished initialization. initial ERGM score = {}'.format(self.score))

    def add_cache_dict(self, key):
        """
        :return: new dictionary for desired key
        """
        new_dict = {}
        self.cache[key] = new_dict
        return new_dict

    def init_ergm_features(self, graphs=None):
        """
        Computes ERGM features from scratch, populates cache members and self.feature_vals
        :param graphs: if not None, changes underlying member and inits from it.
        """
        if graphs is not None:
            self.graphs = graphs
        self.feature_vals = {}
        self.init_ergm_cache()
        self.update_features()
        timeprint('initialized features from cache')

    def init_ergm_cache(self):
        """
        Computes ERGM features from scratch, populates cache members
        """

        # edges
        for r in self.relation_names:
            edges = edge_count(self.graphs[r])
            self.edge_counts[r] = edges

        timeprint('populated edge cache')

        # mutual edges
        for i, n1 in enumerate(self.relation_names):
            r1 = self.graphs[n1]
            for j in range(i+1): # unordered, including self
                n2 = self.relation_names[j]
                r2 = self.graphs[n2]
                mut_edges = mutual_edges(r1, r2)
                self.mutual_edge_counts[(n1, n2)] = mut_edges

        timeprint('populated mutual edge cache')

        # directed triangles - iterate over R^2 + choose(r,3)/3 ordered relation triplets
        eye = csr_eye(self.vocab_size)
        for i,j,k in cyclic_triads(self.R):
            n1, n2, n3 = self.relation_names[i], self.relation_names[j], self.relation_names[k]
            r1, r2, r3 = self.graphs[n1], self.graphs[n2], self.graphs[n3]
            dir_triangles = (r1 * r2 * r3).multiply(eye).sum()
            if i == j and j == k: # each triangle was counted thrice, except self loops
                self_loops = r1.diagonal().sum()
                dir_triangles += 2 * self_loops
                dir_triangles /=3
            self.directed_triangle_counts[(n1, n2, n3)] = dir_triangles

        timeprint('extracted directed triangle features')

        # transitive closures - iterate over ordered relation triplets
        # (also populate self.two_path_counts)
        for n1, r1 in list(self.graphs.items()):
            for n2, r2 in list(self.graphs.items()):
                two_paths = r1 * r2
                two_path_count = max([two_paths.sum(), sys.float_info.epsilon])
                self.two_path_counts[(n1, n2)] = two_path_count
                for n3, r3 in list(self.graphs.items()):
                    closures = two_paths.multiply(r3).sum() # pointwise
                    self.transitive_closure_counts[(n1, n2, n3)] = closures

        timeprint('populated transitivity cache')

        # 1-star cache for updates + self-2,3-stars
        for n, g in list(self.graphs.items()):
            self.in_degs[n] = g.sum(0) # numpy matrix
            self.out_degs[n] = g.sum(1).transpose() # numpy matrix

            osc = one_rel_star_counts(self.in_degs[n], self.out_degs[n])
            
            self.in_one_star_counts[n] = osc['i1sc']
            self.out_one_star_counts[n] = osc['o1sc']
            self.in_two_star_counts[(n, n)] = osc['i2sc']
            self.out_two_star_counts[(n, n)] = osc['o2sc']
            self.in_three_star_counts[(n, n, n)] = osc['i3sc']
            self.out_three_star_counts[(n, n, n)] = osc['o3sc']

            self.in_one_p_star_counts[n] = osc['i1psc']
            self.out_one_p_star_counts[n] = osc['o1psc']
            self.in_two_p_star_counts[(n, n)] = osc['i2psc']
            self.out_two_p_star_counts[(n, n)] = osc['o2psc']
            self.in_three_p_star_counts[(n, n, n)] = osc['i3psc']
            self.out_three_p_star_counts[(n, n, n)] = osc['o3psc']

        timeprint('populated 1r-star cache')

        # 2-stars
        for n1, n2 in combinations(self.relation_names, 2):
            
            tsc = two_rel_star_counts(self.in_degs[n1], self.out_degs[n1],\
                                      self.in_degs[n2], self.out_degs[n2])
            
            self.in_two_star_counts[(n1, n2)] = tsc['i2sc']
            self.out_two_star_counts[(n1, n2)] = tsc['o2sc']
            self.in_three_star_counts[(n1, n1, n2)] = tsc['i3sc112']
            self.out_three_star_counts[(n1, n1, n2)] = tsc['o3sc112']
            self.in_three_star_counts[(n1, n2, n2)] = tsc['i3sc122']
            self.out_three_star_counts[(n1, n2, n2)] = tsc['o3sc122']
            self.in_two_p_star_counts[(n1, n2)] = tsc['i2psc']
            self.out_two_p_star_counts[(n1, n2)] = tsc['o2psc']
            self.in_three_p_star_counts[(n1, n1, n2)] = tsc['i3psc112']
            self.out_three_p_star_counts[(n1, n1, n2)] = tsc['o3psc112']
            self.in_three_p_star_counts[(n1, n2, n2)] = tsc['i3psc122']
            self.out_three_p_star_counts[(n1, n2, n2)] = tsc['o3psc122']

        timeprint('populated 2r-star cache')

        # 3-stars
        for n1, n2, n3 in combinations(self.relation_names, 3):
            ttsc = three_rel_star_counts(self.in_degs[n1], self.out_degs[n1],\
                                         self.in_degs[n2], self.out_degs[n2],\
                                         self.in_degs[n3], self.out_degs[n3])
            
            self.in_three_star_counts[(n1, n2, n3)] = ttsc['i3sc']
            self.out_three_star_counts[(n1, n2, n3)] = ttsc['o3sc']
            self.in_three_p_star_counts[(n1, n2, n3)] = ttsc['i3psc']
            self.out_three_p_star_counts[(n1, n2, n3)] = ttsc['o3psc']

        timeprint('populated 3r-star cache')

    def reread_cache(self, new_cache):
        self.cache = new_cache
        self.edge_counts = self.cache['ec']
        self.mutual_edge_counts = self.cache['mec']
        self.two_path_counts = self.cache['tpc']
        self.transitive_closure_counts = self.cache['tcc']
        self.directed_triangle_counts = self.cache['dtc']
        self.in_degs = self.cache['ins']
        self.out_degs = self.cache['outs']
        self.in_one_star_counts = self.cache['i1sc']
        self.out_one_star_counts = self.cache['o1sc']
        self.in_two_star_counts = self.cache['i2sc']
        self.out_two_star_counts = self.cache['o2sc']
        self.in_three_star_counts = self.cache['i3sc']
        self.out_three_star_counts = self.cache['o3sc']
        self.in_one_p_star_counts = self.cache['i1psc']
        self.out_one_p_star_counts = self.cache['o1psc']
        self.in_two_p_star_counts = self.cache['i2psc']
        self.out_two_p_star_counts = self.cache['o2psc']
        self.in_three_p_star_counts = self.cache['i3psc']
        self.out_three_p_star_counts = self.cache['o3psc']

    def update_features(self):
        # edge counts
        for r, val in list(self.edge_counts.items()):
            self.feature_vals[('ec', r)] = val
        
        # mutual edge counts
        for (r1, r2), val in list(self.mutual_edge_counts.items()):
            self.feature_vals[('mec', r1, r2)] = val
            
        # two-path counts
        for (r1, r2), val in list(self.two_path_counts.items()):
            self.feature_vals[('tpc', r1, r2)] = val
        
        # directed triangles
        for (r1, r2, r3), val in list(self.directed_triangle_counts.items()):
            self.feature_vals[('dtc', r1, r2, r3)] = val
        
        # transitive closure
        for (r1, r2, r3), val in list(self.transitive_closure_counts.items()):
            denominator = self.two_path_counts[(r1, r2)]
            self.feature_vals[('trans', r1, r2, r3)] = \
                              0.0 if denominator == 0.0 \
                              else val / denominator
        
        # exact k-stars
        for r, val in list(self.in_one_star_counts.items()):
            self.feature_vals[('i1sc', r)] = val
        for r, val in list(self.out_one_star_counts.items()):
            self.feature_vals[('o1sc', r)] = val
        for (r1, r2), val in list(self.in_two_star_counts.items()):
            self.feature_vals[('i2sc', r1, r2)] = val
        for (r1, r2), val in list(self.out_two_star_counts.items()):
            self.feature_vals[('o2sc', r1, r2)] = val
        for (r1, r2, r3), val in list(self.in_three_star_counts.items()):
            self.feature_vals[('i3sc', r1, r2, r3)] = val
        for (r1, r2, r3), val in list(self.out_three_star_counts.items()):
            self.feature_vals[('o3sc', r1, r2, r3)] = val
        
        # at-least k-stars
        for r, val in list(self.in_one_p_star_counts.items()):
            self.feature_vals[('i1psc', r)] = val
        for r, val in list(self.out_one_p_star_counts.items()):
            self.feature_vals[('o1psc', r)] = val
        for (r1, r2), val in list(self.in_two_p_star_counts.items()):
            self.feature_vals[('i2psc', r1, r2)] = val
        for (r1, r2), val in list(self.out_two_p_star_counts.items()):
            self.feature_vals[('o2psc', r1, r2)] = val
        for (r1, r2, r3), val in list(self.in_three_p_star_counts.items()):
            self.feature_vals[('i3psc', r1, r2, r3)] = val
        for (r1, r2, r3), val in list(self.out_three_p_star_counts.items()):
            self.feature_vals[('o3psc', r1, r2, r3)] = val

    def zero_all_feats(self, r):
        for k in self.feature_vals:
            if r in k[1:]:
                self.feature_vals[k] = 0
    
    def ergm_score(self):
        """
        :return: ERGM score (dynet Expression) computed based on ERGM weights and features only
        Does not populate any field
        """
        W = dy.parameter(self.ergm_weights)
        f = dy.transpose(dy.inputVector([self.feature_vals[k] for k in self.feature_set]))
        return f * W

    def rescore(self):
        """
        Computes score based on current parameter and feature values, populates field
        """
        self.dy_score = self.ergm_score()
        self.score = self.dy_score.scalar_value()
        self.score_is_stale = False

    ### EDGE ABLATION MODE ###

    def remove_edge(self, src_idx, trg_idx, rel, update_feats=True, permanent=True,
                    caches=None, report_feat_diff=False):
        """
        Removes edge from graph, updates cache and feature values
        :param src_idx: index of source node from edge to remove
        :param trg_idx: index of target node from edge to remove
        :param rel: relation type
        :param update_feats: flag for not updating all cache and features, to be deferred
        :returns: if permanent=False, returns ergm score of removed-edge graph
        """
        if permanent:
            self.score_is_stale = True
            cached_feats = None
            cached_cache = None
        else:
            if caches is not None:
                cached_cache = copy.deepcopy(caches[0])
                cached_feats = caches[1]
            else:
                cached_cache = copy.deepcopy(self.cache)
                cached_feats = copy.deepcopy(self.feature_vals)
            update_feats=True # no other mode possible

        # update cache members
        # decrement edge count for rel
        self.edge_counts[rel] -= 1

        # pair cache members
        for r,g in list(self.graphs.items()):
            if rel == 'hypernym' and r == 'co_hypernym':
                continue

            # decrement mutual edge count for pairs with trg-src edges
            if g[trg_idx, src_idx] == 1:
                self.mutual_edge_counts[find_key(self.mutual_edge_counts, (rel, r))] -= 1

            # decrement two-paths for x-src-trg and src-trg-y
            self.two_path_counts[(r, rel)] -= self.in_degs[r][0,src_idx]
            self.two_path_counts[(rel, r)] -= self.out_degs[r][0,trg_idx]
            
            # triplet cache members
            for r2, g2 in list(self.graphs.items()):
                if rel == 'hypernym' and r2 == 'co_hypernym':
                    continue
                    
                # decrement transitive closures from two-paths src-x-trg
                if self.out_degs[r][0,src_idx] > 0 and self.in_degs[r2][0,trg_idx] > 0:
                    r_r2_betweens = (g[src_idx] * g2[:,trg_idx]).sum()
                    self.transitive_closure_counts[(r, r2, rel)] -= r_r2_betweens
                # decrement directed triangle count
                if self.out_degs[r2][0,trg_idx] > 0 and self.in_degs[r][0,src_idx] > 0:
                    r_r2_cycles = (g2[trg_idx] * g[:,src_idx]).sum()
                    rs_key = find_cyclic_key(self.directed_triangle_counts, (r, rel, r2))
                    self.directed_triangle_counts[rs_key] -= r_r2_cycles

        # decrement src's out_degree and trg's in_degree in rel and update all related caches
        self.out_degs[rel][0,src_idx] -= 1
        self.in_degs[rel][0,trg_idx] -= 1

        if update_feats:
            # recompute heavy cache updates from raw counts
            self.update_stars_cache_from_edge(rel, src_idx, trg_idx, added=False)

            # update features from caches
            self.update_features()

        if not permanent and report_feat_diff:
            timeprint('changed feature values:')
            diff_keys = [k for k in self.feature_vals if self.feature_vals[k] != cached_feats[k]]
            if len(diff_keys) > 0:
                print('\n'.join(['{}: from {} to {}'\
                      .format(k, cached_feats[k], self.feature_vals[k]) for k in diff_keys]))
        
        if permanent:
            # remove actual edge
            self.graphs[rel][src_idx,trg_idx] = 0
        else:
            if rel == 'hypernym':
                self.zero_all_feats('co_hypernym')
            
            # prepare return value
            ret = self.ergm_score()

            # revert everything
            self.reread_cache(cached_cache)
            self.feature_vals = cached_feats

            # return prepared score
            return ret


    def add_edge(self, src_idx, trg_idx, rel, permanent=False, caches=None, report_feat_diff=False):
        """
        Uses cache to update feature values and produce score
        :param src_idx: index of source node from edge to add
        :param trg_idx: index of target node from edge to add
        :param rel: relation type
        :param permanent: True if node assignment to remain as is (inference mode, or restitution)
        :param cache: optional - precomputed backup members (cache, features)
        :return: new ergm score
        """
        # back cache up
        if caches is not None:
            backup_cache = copy.deepcopy(caches[0])
            backup_feats = caches[1]
        elif not permanent:
            backup_cache = copy.deepcopy(self.cache)
            backup_feats = copy.deepcopy(self.feature_vals)
        else:
            backup_cache = None
            backup_feats = None

        # update cache members
        # increment edge count for r
        self.edge_counts[rel] += 1

        # pair cache members
        for r,g in list(self.graphs.items()):
            if rel == 'hypernym' and r == 'co_hypernym':
                continue

            # increment mutual edge count for pairs with trg-src edges
            if g[trg_idx, src_idx] == 1:
                self.mutual_edge_counts[find_key(self.mutual_edge_counts, (rel, r))] += 1

            # increment two-paths for x-src-trg and src-trg-y
            self.two_path_counts[(r, rel)] += self.in_degs[r][0,src_idx]
            self.two_path_counts[(rel, r)] += self.out_degs[r][0,trg_idx]

            # triplet cache members
            for r2, g2 in list(self.graphs.items()):
                if rel == 'hypernym' and r2 == 'co_hypernym':
                    continue
                    
                # increment transitive closures from two-paths src-x-trg
                if self.out_degs[r][0,src_idx] > 0 and self.in_degs[r2][0,trg_idx] > 0:
                    r_r2_betweens = (g[src_idx] * g2[:,trg_idx]).sum()
                    self.transitive_closure_counts[(r, r2, rel)] += r_r2_betweens
                # increment directed triangle count
                if self.out_degs[r2][0,trg_idx] > 0 and self.in_degs[r][0,src_idx] > 0:
                    r_r2_cycles = (g2[trg_idx] * g[:,src_idx]).sum()
                    rs_key = find_cyclic_key(self.directed_triangle_counts, (r, rel, r2))
                    self.directed_triangle_counts[rs_key] += r_r2_cycles

        # increment src's out_degree and trg's in_degree in rel and update all related caches
        self.out_degs[rel][0,src_idx] += 1
        self.in_degs[rel][0,trg_idx] += 1
        
        self.update_stars_cache_from_edge(rel, src_idx, trg_idx)
        
        # update features from caches
        self.update_features()
        if rel == 'hypernym':
            self.zero_all_feats('co_hypernym')
        
        if report_feat_diff:
            timeprint('changed feature values:')
            diff_keys = [k for k in self.feature_vals if self.feature_vals[k] != backup_feats[k]]
            print('\n'.join(['{}: from {} to {}'\
                  .format(k, backup_feats[k], self.feature_vals[k]) for k in diff_keys]))

        # compute score for loss
        ret = self.ergm_score()

        if permanent:
            # add actual edge
            self.graphs[rel][src_idx,trg_idx] = 1
            # update score
            self.dy_score = ret
            self.score = ret.scalar_value()
            self.score_is_stale = False
        else:
            self.reread_cache(backup_cache)
            self.feature_vals = backup_feats

        return ret

    ### END EDGE ABLATION MODE ###

    def update_stars_cache_from_edge(self, rel, src, trg, added=True):
        """
        An ugly, ugly function to try and do update_stars_cache() efficiently.
        """
        # TODO add global ifs for large-degree nodes to remove unnecessary combinatorial checks?
        curr_src_degs = {r: self.out_degs[r][0, src] for r in self.relation_names}
        curr_trg_degs = {r: self.in_degs[r][0, trg] for r in self.relation_names}
        
        # self-stars
        if added:
            # src side
            # self-stars
            if curr_src_degs[rel] == 1:
                self.out_one_star_counts[rel] += 1
                self.out_one_p_star_counts[rel] += 1
            if curr_src_degs[rel] == 2:
                self.out_one_star_counts[rel] -= 1
                self.out_two_star_counts[(rel, rel)] += 1
                self.out_two_p_star_counts[(rel, rel)] += 1
            if curr_src_degs[rel] == 3:
                self.out_two_star_counts[(rel, rel)] -= 1
                self.out_three_star_counts[(rel, rel, rel)] += 1
                self.out_three_p_star_counts[(rel, rel, rel)] += 1
            if curr_src_degs[rel] == 4:
                self.out_three_star_counts[(rel, rel, rel)] -= 1
                
            # 2-stars
            for r2 in self.relation_names:
                if r2 == rel: continue
                if curr_src_degs[rel] * curr_src_degs[r2] == 1:
                    self.out_two_star_counts[find_key(self.out_two_star_counts, (rel, r2))] += 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] >= 1:
                    self.out_two_p_star_counts[find_key(self.out_two_p_star_counts, (rel, r2))] += 1
                if curr_src_degs[rel] == 2 and curr_src_degs[r2] == 1:
                    self.out_two_star_counts[find_key(self.out_two_star_counts, (rel, r2))] -= 1
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, rel, r2))] += 1
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, rel, r2))] += 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] == 2:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r2))] += 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] >= 2:
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r2))] += 1
                if curr_src_degs[rel] == 3 and curr_src_degs[r2] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, rel, r2))] -= 1
                if curr_src_degs[rel] == 2 and curr_src_degs[r2] == 2:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r2))] -= 1
            
            # 3-stars
            for r2, r3 in combinations(self.relation_names, 2):
                if r2 == rel or r3 == rel: continue
                if curr_src_degs[rel] * curr_src_degs[r2] * curr_src_degs[r3] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r3))] += 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] * curr_src_degs[r3] >= 1:
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r3))] += 1
                if curr_src_degs[rel] == 2 and curr_src_degs[r2] * curr_src_degs[r3] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r3))] -= 1
            
            # trg side
            # self-stars
            if curr_trg_degs[rel] == 1:
                self.in_one_star_counts[rel] += 1
                self.in_one_p_star_counts[rel] += 1
            if curr_trg_degs[rel] == 2:
                self.in_one_star_counts[rel] -= 1
                self.in_two_star_counts[(rel, rel)] += 1
                self.in_two_p_star_counts[(rel, rel)] += 1
            if curr_trg_degs[rel] == 3:
                self.in_two_star_counts[(rel, rel)] -= 1
                self.in_three_star_counts[(rel, rel, rel)] += 1
                self.in_three_p_star_counts[(rel, rel, rel)] += 1
            if curr_trg_degs[rel] == 4:
                self.in_three_star_counts[(rel, rel, rel)] -= 1
                
            # 2-stars           
            for r2 in self.relation_names:
                if r2 == rel: continue
                if curr_trg_degs[rel] * curr_trg_degs[r2] == 1:
                    self.in_two_star_counts[find_key(self.in_two_star_counts, (rel, r2))] += 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] >= 1:
                    self.in_two_p_star_counts[find_key(self.in_two_p_star_counts, (rel, r2))] += 1
                if curr_trg_degs[rel] == 2 and curr_trg_degs[r2] == 1:
                    self.in_two_star_counts[find_key(self.in_two_star_counts, (rel, r2))] -= 1
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, rel, r2))] += 1
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, rel, r2))] += 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] == 2:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r2))] += 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] >= 2:
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r2))] += 1
                if curr_trg_degs[rel] == 3 and curr_trg_degs[r2] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, rel, r2))] -= 1
                if curr_trg_degs[rel] == 2 and curr_trg_degs[r2] == 2:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r2))] -= 1
            
            # 3-stars
            for r2, r3 in combinations(self.relation_names, 2):
                if r2 == rel or r3 == rel: continue
                if curr_trg_degs[rel] * curr_trg_degs[r2] * curr_trg_degs[r3] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r3))] += 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] * curr_trg_degs[r3] >= 1:
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r3))] += 1
                if curr_trg_degs[rel] == 2 and curr_trg_degs[r2] * curr_trg_degs[r3] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r3))] -= 1
            
        else: # edge removed
            # src side
            # self-stars
            if curr_src_degs[rel] == 0:
                self.out_one_star_counts[rel] -= 1
                self.out_one_p_star_counts[rel] -= 1
            if curr_src_degs[rel] == 1:
                self.out_one_star_counts[rel] += 1
                self.out_two_star_counts[(rel, rel)] -= 1
                self.out_two_p_star_counts[(rel, rel)] -= 1
            if curr_src_degs[rel] == 2:
                self.out_two_star_counts[(rel, rel)] += 1
                self.out_three_star_counts[(rel, rel, rel)] -= 1
                self.out_three_p_star_counts[(rel, rel, rel)] -= 1
            if curr_src_degs[rel] == 3:
                self.out_three_star_counts[(rel, rel, rel)] += 1
                
            # 2-stars
            for r2 in self.relation_names:
                if r2 == rel: continue
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] == 1:
                    self.out_two_star_counts[find_key(self.out_two_star_counts, (rel, r2))] -= 1
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] >= 1:
                    self.out_two_p_star_counts[find_key(self.out_two_p_star_counts, (rel, r2))] -= 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] == 1:
                    self.out_two_star_counts[find_key(self.out_two_star_counts, (rel, r2))] += 1
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, rel, r2))] -= 1
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, rel, r2))] -= 1
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] == 2:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r2))] -= 1
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] >= 2:
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r2))] -= 1
                if curr_src_degs[rel] == 2 and curr_src_degs[r2] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, rel, r2))] += 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] == 2:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r2))] += 1
            
            # 3-stars
            for r2, r3 in combinations(self.relation_names, 2):
                if r2 == rel or r3 == rel: continue
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] * curr_src_degs[r3] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r3))] -= 1
                if curr_src_degs[rel] == 0 and curr_src_degs[r2] * curr_src_degs[r3] >= 1:
                    self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r3))] -= 1
                if curr_src_degs[rel] == 1 and curr_src_degs[r2] * curr_src_degs[r3] == 1:
                    self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r3))] += 1
            
            # trg side
            # self-stars
            if curr_trg_degs[rel] == 0:
                self.in_one_star_counts[rel] -= 1
                self.in_one_p_star_counts[rel] -= 1
            if curr_trg_degs[rel] == 1:
                self.in_one_star_counts[rel] += 1
                self.in_two_star_counts[(rel, rel)] -= 1
                self.in_two_p_star_counts[(rel, rel)] -= 1
            if curr_trg_degs[rel] == 2:
                self.in_two_star_counts[(rel, rel)] += 1
                self.in_three_star_counts[(rel, rel, rel)] -= 1
                self.in_three_p_star_counts[(rel, rel, rel)] -= 1
            if curr_trg_degs[rel] == 3:
                self.in_three_star_counts[(rel, rel, rel)] += 1
                
            # 2-stars
            for r2 in self.relation_names:
                if r2 == rel: continue
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] == 1:
                    self.in_two_star_counts[find_key(self.in_two_star_counts, (rel, r2))] -= 1
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] >= 1:
                    self.in_two_p_star_counts[find_key(self.in_two_p_star_counts, (rel, r2))] -= 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] == 1:
                    self.in_two_star_counts[find_key(self.in_two_star_counts, (rel, r2))] += 1
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, rel, r2))] -= 1
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, rel, r2))] -= 1
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] == 2:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r2))] -= 1
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] >= 2:
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r2))] -= 1
                if curr_trg_degs[rel] == 2 and curr_trg_degs[r2] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, rel, r2))] += 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] == 2:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r2))] += 1
            
            # 3-stars
            for r2, r3 in combinations(self.relation_names, 2):
                if r2 == rel or r3 == rel: continue
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] * curr_trg_degs[r3] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r3))] -= 1
                if curr_trg_degs[rel] == 0 and curr_trg_degs[r2] * curr_trg_degs[r3] >= 1:
                    self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r3))] -= 1
                if curr_trg_degs[rel] == 1 and curr_trg_degs[r2] * curr_trg_degs[r3] == 1:
                    self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r3))] += 1
        
    
    def update_stars_cache(self, rel):
        """
        in bulk, for entire relation graph
        much prettier than the above, but much less efficient
        """
        out_ds = self.out_degs[rel]
        in_ds = self.in_degs[rel]

        # self-stars
        osc = one_rel_star_counts(in_ds, out_ds)
        self.in_one_star_counts[rel] = osc['i1sc']
        self.out_one_star_counts[rel] = osc['o1sc']
        self.in_two_star_counts[(rel, rel)] = osc['i2sc']
        self.out_two_star_counts[(rel, rel)] = osc['o2sc']
        self.in_three_star_counts[(rel, rel, rel)] = osc['i3sc']
        self.out_three_star_counts[(rel, rel, rel)] = osc['o3sc']
        self.in_one_p_star_counts[rel] = osc['i1psc']
        self.out_one_p_star_counts[rel] = osc['o1psc']
        self.in_two_p_star_counts[(rel, rel)] = osc['i2psc']
        self.out_two_p_star_counts[(rel, rel)] = osc['o2psc']
        self.in_three_p_star_counts[(rel, rel, rel)] = osc['i3psc']
        self.out_three_p_star_counts[(rel, rel, rel)] = osc['o3psc']

        # 2-stars
        for r2 in self.relation_names:
            if r2 == rel: continue
            tsc = two_rel_star_counts(in_ds, out_ds, self.in_degs[r2], self.out_degs[r2])
            self.in_two_star_counts[find_key(self.in_two_star_counts, (rel, r2))] = tsc['i2sc']
            self.out_two_star_counts[find_key(self.out_two_star_counts, (rel, r2))] = tsc['o2sc']
            self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, rel, r2))] = tsc['i3sc112']
            self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, rel, r2))] = tsc['o3sc112']
            self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r2))] = tsc['i3sc122']
            self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r2))] = tsc['o3sc122']
            self.in_two_p_star_counts[find_key(self.in_two_p_star_counts, (rel, r2))] = tsc['i2psc']
            self.out_two_p_star_counts[find_key(self.out_two_p_star_counts, (rel, r2))] = tsc['o2psc']
            self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, rel, r2))] = tsc['i3psc112']
            self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, rel, r2))] = tsc['o3psc112']
            self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r2))] = tsc['i3psc122']
            self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r2))] = tsc['o3psc122']

        # 3-stars
        for r2, r3 in combinations(self.relation_names, 2):
            if r2 == rel or r3 == rel: continue
            ttsc = three_rel_star_counts(in_ds, out_ds,\
                                         self.in_degs[r2], self.out_degs[r2],\
                                         self.in_degs[r3], self.out_degs[r3])
            
            self.in_three_star_counts[find_key(self.in_three_star_counts, (rel, r2, r3))] = ttsc['i3sc']
            self.out_three_star_counts[find_key(self.out_three_star_counts, (rel, r2, r3))] = ttsc['o3sc']
            self.in_three_p_star_counts[find_key(self.in_three_p_star_counts, (rel, r2, r3))] = ttsc['i3psc']
            self.out_three_p_star_counts[find_key(self.out_three_p_star_counts, (rel, r2, r3))] = ttsc['o3psc']

    def save(self, filename, initial_weights=None, save_with_embeddings=True):
        # model payload
        if save_with_embeddings:
            np.save(filename + '-embs.npy', self.embeddings.as_array())
            # self.model.save(filename + '.dyn') # saves all embeddings - move next row to else
        dy.save(filename + '.dyn', [self.ergm_weights] +\
                [self.word_assoc_weights[r] for r in self.relation_names]) # order matters for loading

        # feature ordering
        pickle.dump(self.feature_set, open(filename + '.feats', 'wb'))
        
        # nice-to-read score summary
        if initial_weights is not None:
            self.save_weights(filename, initial_weights)

    def save_weights(self, filename, initial_weights):
        """
        Save feature weights in readable form
        """
        with open(filename + '.scores', 'w') as file_out:
            file_out.write('feat cat\trelations\tfinal weight\tinitial weight\tdiff\n')
            w = self.ergm_weights.as_array()
            delta = w - initial_weights
            # order by feature scores
            s_del = sorted(enumerate(delta), key=lambda x: -np.abs(x[1]))
            for _,(k,d_i) in enumerate(s_del):
                file_out.write('{}\t{:.5f}\t{:.5f}\t{:.5f}\n'\
                                .format(tupstr(self.feature_set[k]), w[k],
                                        initial_weights[k], d_i))
