#!/usr/bin/env python
'''
Pre-processing join job for extracting synset embeddings for WordNet 3.0 IDs
'''

import numpy as np
from datetime import datetime
from collections import namedtuple, Counter
import pickle as pickle
from nltk.corpus import wordnet as wn # 3.0
import progressbar
import random

__author__ = "Yuval Pinter, 2018"

MATRICES_DATA_W_SETS = 'data/sparse_wordnet_matrices.pkl'
SYNSET_EMBEDDINGS_FILE = 'data/ae-w2v-synsets-21.txt'
NOUN_MAPPINGS = 'data/1.7.1to3.0-noun-all-mapping.tsv'
VERB_MAPPINGS = 'data/1.7.1to3.0-verb-all-mapping.tsv'
ADJ_LOOKUP_FILE = 'data/WordNet-1.7.1/dict/index.adj'
ADV_LOOKUP_FILE = 'data/WordNet-1.7.1/dict/index.adv'
OUTPUT_FILE = 'data/ae-w2v-synsets.vec'

EMBS_PREF = 'wn-2.1-'
PREF = len(EMBS_PREF)

SparseMatrixDataset = namedtuple('SparseMatrixDataset', ['matrices', 'index'])

def offset_pos_from_emb_key(str):
    return str[PREF:PREF + 8], str[PREF + 9:PREF + 10]
    
def offstr(off):
    return '{0:0>8}'.format(off)

def pos_of_set(set_name):
    return set_name.split('.')[-2]

if __name__ == '__main__':
    # map synset names to offsets
    sset_offs = {s.name():s.offset() for s in wn.all_synsets()}

    # load expected synset names in order (3.0)
    with open(MATRICES_DATA_W_SETS) as matrix_file:
        set_names = pickle.load(matrix_file)[1] # it's the same
        # as wn.all_synsets(), but let's be really careful and extensible.
    print('finished loading {} 3.0 synsets and offset map'.format(len(set_names)))
    
    # load embeddings
    syn_embs = {'n':{}, 'v':{}, 'r':{}, 'a':{}}
    with open(SYNSET_EMBEDDINGS_FILE) as embs_file:
        header_line = True
        for l in embs_file.readlines():
            if header_line: # skip
                header_line = False
                continue
            
            # not bothering with saving as numbers since we'll write it out the same
            key, vec = l.split(' ', 1)
            offset, pos = offset_pos_from_emb_key(key)
            syn_embs[pos][offset] = vec
    print('finished loading 2.1 embeddings: {} nouns, {} verbs, {} adjectives, {} adverbs'\
            .format(len(syn_embs['n']), len(syn_embs['v']),\
            len(syn_embs['a']), len(syn_embs['r'])))
    
    # read mappings from pre-composed tables using sensemap
    offset_maps = {'n':{}, 'v':{}, 'r':{}, 'a':{}}
    with open(NOUN_MAPPINGS) as noun_map_file:
        for l in noun_map_file:
            two, three = l.strip().split()
            key = int(three)
            if key in offset_maps['n']: continue # first is more reliable
            offset_maps['n'][key] = two.strip()
    print('read {} noun mappings'.format(len(offset_maps['n'])))
    
    with open(VERB_MAPPINGS) as verb_map_file:
        for l in verb_map_file:
            two, three = l.strip().split()
            key = int(three)
            if key in offset_maps['v']: continue # first is more reliable
            offset_maps['v'][key] = two.strip()
    print('read {} verb mappings'.format(len(offset_maps['v'])))
    
    # these have to be read from the original WN index format
    with open(ADJ_LOOKUP_FILE) as adj_map_file:
        for l in adj_map_file:
            if l[0] == ' ': continue # header
            parts = l.strip().split()
            num_of_sets = int(parts[2])
            offset_maps['a'][parts[0]] = parts[-num_of_sets:]
    print('read {} adjective mappings'.format(len(offset_maps['a'])))
    
    with open(ADV_LOOKUP_FILE) as adv_map_file:
        for l in adv_map_file:
            if l[0] == ' ': continue # header
            parts = l.strip().split()
            num_of_sets = int(parts[2])
            offset_maps['r'][parts[0]] = parts[-num_of_sets:]
    print('read {} adverb mappings'.format(len(offset_maps['r'])))
    
    # final join
    wrote = Counter()
    misses = Counter()
    emb_misses = Counter()
    questionable_mappings = Counter()
    bar = progressbar.ProgressBar()
    
    print('performing join... writing to', OUTPUT_FILE)
    with open(OUTPUT_FILE, 'w') as out_file:
        for sn in bar(set_names):
            p = pos_of_set(sn)
            if p == 's':
                p = 'a'
            if p == 'a' or p == 'r':
                nm = sn.split('.')[0]
                if nm not in offset_maps[p]:
                    misses[p] += 1
                    continue
                sense_ind = int(sn.split('.')[-1]) - 1
                l_offsets = offset_maps[p][nm]
                if len(l_offsets) <= sense_ind:
                    sense_ind = len(l_offsets) - 1
                    questionable_mappings[p] += 1
                ofs = l_offsets[sense_ind]
                if ofs not in syn_embs[p]:
                    sense_ind -= 1
                    while sense_ind >= 0:
                        if l_offsets[sense_ind] in syn_embs[p]:
                            ofs = l_offsets[sense_ind]
                            questionable_mappings[p] += 1
                            break
                        else:
                            sense_ind -= 1
                    if sense_ind < 0:
                        misses[p] += 1
                        continue
                if ofs not in syn_embs[p]:
                    emb_misses[p] += 1
                    continue
                emb = syn_embs[p][ofs]
                out_file.write(sn + ' ' + emb) # already stored with newline
                wrote[p] += 1
            elif p == 'n' or p == 'v':
                ofs = sset_offs[sn] # shouldn't ever fail
                if ofs not in offset_maps[p]:
                    misses[p] += 1
                    continue
                if offset_maps[p][ofs] not in syn_embs[p]:
                    emb_misses[p] += 1
                    continue
                emb = syn_embs[p][offset_maps[p][ofs]]
                out_file.write(sn + ' ' + emb) # already stored with newline
                wrote[p] += 1
            else:
                print('encountered unknown pos:', p)
    
    # report statistics
    print('wrote embeddings for:')
    print(wrote)
    print('===')
    print('missed in version switch:')
    print(misses)
    print('===')
    print('missed in embeddings table:')
    print(emb_misses)
    print('===')
    print('questionable lemma-based mappings:')
    print(questionable_mappings)