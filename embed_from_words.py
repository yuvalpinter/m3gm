import sys
import numpy as np
import codecs
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from io_utils import load_prediction_dataset, WordnetPredictionDataset, timeprint

LOWER = True
ALL_LEMMAS = True

def lemmas(s):
    if ALL_LEMMAS:
        name = '_'.join(s.lemma_names())
    else:
        name = s.lemma_names()[0]
    if LOWER:
        name = name.lower()
    return name.split('_')
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        timeprint('usage: embed_from_words.py input_embs output_embs [WN prediction dataset]')
        exit(1)
        
    in_file = sys.argv[1]
    
    # create target dataset
    if len(sys.argv) > 3:
        # third param is WN dataset
        wn_vocab = load_prediction_dataset(sys.argv[3])[-1]
        synsets = [wn.synset(w) for w in wn_vocab]
    else:
        synsets = list(wn.all_synsets())
    timeprint('read {} synsets'.format(len(synsets)))
    
    target_words = set()
    timeprint('preparing target word dataset')
    for s in tqdm(synsets):
        target_words.update(lemmas(s))
    
    # read input file (fasttext)
    embedding_dim = 0
    word_embs = {}
    errors = 0
    timeprint('reading input file')
    with codecs.open(in_file, "r", "utf-8") as f:
        for line in tqdm(f.readlines()):
            split = line.strip().split()
            if len(split) > 2:
                word = split[0]
                if word in target_words:
                    vec = np.array(split[1:])
                    try:
                        vec = vec.astype(np.float)
                    except ValueError:
                        errors += 1
                        continue
                    word_embs[word] = vec
                    if embedding_dim == 0:
                        embedding_dim = len(vec)
    timeprint('found {} words out of {} from {} synsets in file with {} errors.'\
           .format(len(word_embs), len(target_words), len(synsets), errors))
    
    # write sysnet vectors, averaged from lemmas
    out_file = sys.argv[2]
    timeprint('writing averaged vectors')
    seen = 0
    unseen = 0
    unseen_synsets = 0
    with open(out_file, "w") as fo:
        for s in tqdm(synsets):
            swords = lemmas(s)
            vecs = []
            any_word_has_emb = False
            for w in swords:
                if w in word_embs:
                    vecs.append(word_embs[w])
                    seen += 1
                    any_word_has_emb = True
                else:
                    vecs.append(np.random.uniform(-0.8, 0.8, embedding_dim))
                    unseen += 1
            if not any_word_has_emb:
                unseen_synsets += 1
            avg = np.average(vecs, 0)
            fo.write(s.name() + ' ' + ' '.join(['{:.6}'.format(d) for d in avg]) + '\n')
    timeprint('finished writing {} synset embeddings. Seen {} words, not seen {}. {} synsets have no embeddings at all.'\
           .format(len(synsets), seen, unseen, unseen_synsets))
