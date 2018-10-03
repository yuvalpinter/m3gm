# M3GM

**October 3: most of the code is here. All parts until association model training are validated.**

This repository is the future home for code for *Max-Margin Markov Graph Models (M3GMs)* as described in the [paper](http://arxiv.org/abs/1808.08644):

```
Yuval Pinter and Jacob Eisenstein. Predicting Semantic Relations using Global Graph Properties. EMNLP 2018.
```

# Code Requirements

The project was written and tested in Python 3.6. Some packages needed to run it include:
* `dynet 2.0`
* `scipy`
* `tqdm`
* `nltk` - with the `wordnet` corpus available
Write me if you find more blocking dependencies!

# Workflow

The eventual goal of training an M3GM model and replicating the results from the paper runs through a number of intermediate steps. Here is the hopefully full linearized flowchart, with some detailed descriptions in following sections:
1. Create a pickled WordNet prediction dataset in sparse matrix format, using [`create_wn18_data.py`](create_wn18_data.py). To use our exact dataset, obtain the distibution of WN18RR [here](https://github.com/villmow/datasets_knowledge_embedding/tree/master/WN18RR) and point the script at the text version.
1. Obtain synset embeddings. These can be AutoExtend-based ones, which map directly to synsets, or any downloadable word embeddings which can then be averaged across synset lexemes, such as those from [FastText](https://fasttext.cc/).
  1. If your embeddings are word-level, synsetify them using [`Embed_from_words.py`](embed_from_words.py). Run it without parameters to see usage.
1. Train an association model (for baseline results or for training an M3GM on top) using `pretrain_assoc.py`. Demo command (for the result from the paper) given below.
1. Train an M3GM using [`predict_wn18.py`](predict_wn18.py). Demo command (for results from the paper) given below.
  1. If so inclined, tune the `alpha_r` parameters using [`optimize_alpha_per_relation.py`](optimize_alpha_per_relation.py). You will need to do some math later to translate into results 
comparable to those in the paper.

Disclaimer: some of the code here takes a while to run. Any suggestions for improving any of the calculations, or for getting things to run on GPUs for that matter, will be most appreciated.

## Association Models

(TODO)

Demo command:
```
python pretrain_assoc.py --input data/wn18rr.pkl --embeddings data/ft-embs-all-lower.vec --model-out models/pret_transE --nll --assoc-mode transE --neg-samp 10 --early-stopping --eval-dev
```

## Max-Margin Markov Graph Models

(TODO)

Demo command:
```
python predict_wn18.py --input data/wn18rr.pkl --emb-size 300 --model models/pret_transE-ep-14 --model-only-init --assoc-mode transE --eval-dev --no-assoc-bp --epochs 3 --neg-samp 10 --regularize 0.01 --rand-all --skip-symmetrics --model-out models/from_pret_trE-3eps --rerank-out from_pret_trE-3eps.txt
```

# Model Development

A good entry point to try and play with the ERGM features underlying M3GM would be [`ergm_feats.py`](ergm_feats.py). Be sure to enter them into the cache and feature set in [`model.py`](model.py) so they can have weights trained for them.

Running the dataset creation code with the `--no-symmetrics` flag would result in a dataset we called WN18RSR when working on this research. It contains only the seven asymmetric, nonreciprocal relations. All model results on it are abysmal, but you're welcome to try :)

# Repo-level TODOs

[] Add exploration Notebook for WordNet (WN) structure
[] Add mapping of Synset codes from WN 1.6 all the way to 3.0.
[] Move non-script code into `lib` directory

# Contact
`uvp@gatech.edu`.
