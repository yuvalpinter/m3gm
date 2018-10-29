# M3GM

**October 29: all model code is here, documented, and validated. Bonus content is here. All done!**

This repository contains code for *Max-Margin Markov Graph Models (M3GMs)* as described in the [paper](http://www.aclweb.org/anthology/D18-1201): `Predicting Semantic Relations using Global Graph Properties`.

Full citation format below.

# Code Requirements

The project was written and tested in Python 3.6. Some packages needed to run it include:
* `dynet 2.0`
* `scipy`
* `tqdm`
* `nltk` - with the `wordnet` corpus available

Write me, or open an issue, if you find more blocking dependencies!

# Workflow

The eventual goal of training an M3GM model and replicating the results from the paper runs through a number of intermediate steps. Here is the hopefully full linearized flowchart, with some detailed descriptions in following sections:
1. Create a pickled WordNet prediction dataset in sparse matrix format, using [`create_wn18_data.py`](create_wn18_data.py). To use our exact dataset, obtain the distibution of WN18RR [here](https://github.com/villmow/datasets_knowledge_embedding/tree/master/WN18RR) and point the script at the text version.
1. Obtain synset embeddings. These can be AutoExtend-based ones, which map directly to synsets, or any downloadable word embeddings which can then be averaged across synset lexemes, such as those from [FastText](https://fasttext.cc/).
  1. If your embeddings are word-level, synsetify them using [`embed_from_words.py`](embed_from_words.py). Run it without parameters to see usage.
1. Train an association model (for baseline results or for training an M3GM on top) using [`pretrain_assoc.py`](pretrain_assoc.py). Demo command (for the result from the paper) given below.
1. Train an M3GM using [`predict_wn18.py`](predict_wn18.py). Demo command (for [results from the paper](https://nlpprogress.com/relation_prediction.html)) given below.
  1. If so inclined, tune the `alpha_r` parameters using [`optimize_alpha_per_relation.py`](optimize_alpha_per_relation.py). You will need to do some math later to translate into results comparable to those in the paper.

Disclaimer: some of the code here takes a while to run. Any suggestions for improving any of the calculations, or for getting things to run on GPUs for that matter, will be most appreciated.

## Association Models

This script trains a local association model using one of several models (see paper for details): Bilinear, TransE, Diag-R1 ("diagonal + rank-1 matrix"), DistMult.
Be sure to keep record of the embedding dimension used (no need to provide the dimension as an argument if initializing from an a pre-trained file) and of the association algorithm (`--assoc-mode`), as these will be necessary for downstream M3GM training.

One parameter you may want to add depending on your target setup is `--rule-override`, which trains modules for *all* relations, including the four symmetric ones (in WordNet).
It would also evaluate on trained modules in symmetric relations, rather than with a (high-accuracy) rule-based system.
The default behavior, without this parameter, is training said modules once every five epochs, as it helps with synset embeddings tuning.

The `--early-stopping` method used is: for each dev epoch, if its MRR score is lower than both of the last two epochs, halt and return the best model so far.

### Outputs
* the auto-generated log file (avoid using `--no-log`) will output many, many scores and their components for every single instance encountered.
* `--model-out` is readable both by this code for test mode, and by downstream M3GM trainer (`--model` param).

### Demo command
```
python pretrain_assoc.py --input data/wn18rr.pkl --embeddings data/ft-embs-all-lower.vec --model-out models/pret_transE --nll --assoc-mode transE --neg-samp 10 --early-stopping --eval-dev
```

## Max-Margin Markov Graph Models

The most powerful use case for M3GM is when we've trained a good association model, and augment it with weights for combinatorial graph features by way of M3GM training.
It is best if the association weights, as well as the word embeddings, are frozen from this point on, using the `--no-assoc-bp` parameter. If we believe some of them to be bad, they can later be weighted down using the [`optimize_alpha_per_relation.py`](optimize_alpha_per_relation.py) post-processor, which computes a best-performing association component weight for each relation.
`--model-only-init` is a related parameter, which ensures that the M3GM component is trained over the data (makes more sense when considering that there's also an `--ergm-model` input parameter which can be used for picking up training from a saved point).

A prerequesite for this code to run in the common mode is that both `--emb-size` and `assoc-mode` are set to the same values that the association model was trained with.

### Outputs
* the auto-generated log file (avoid using `--no-log`) will output ERGM scores for all instances and negative samples in training phase, and all cases of re-ranking in the development data traversals.
* `--model-out` will save the model in a four-file format that can later be read by both this script and the test-mode code (**TODO**).
* `--rerank-out` provides an input file for [`optimize_alpha_per_relation.py`](optimize_alpha_per_relation.py). It includes all to-be-reranked lists from the dev set and scores from both association and graph components, as well as flags for the true instances.

### Demo command
```
python predict_wn18.py --input data/wn18rr.pkl --emb-size 300 --model models/pret_transE-ep-14 --model-only-init --assoc-mode transE --eval-dev --no-assoc-bp --epochs 3 --neg-samp 10 --regularize 0.01 --rand-all --skip-symmetrics --model-out models/from_pret_trE-3eps --rerank-out from_pret_trE-3eps.txt
```

# Model Development

A good entry point to try and play with the ERGM features underlying M3GM would be [`ergm_feats.py`](ergm_feats.py). Be sure to enter them into the cache and feature set in [`model.py`](model.py) so they can have weights trained for them.

Running the dataset creation code with the `--no-symmetrics` flag would result in a dataset we called WN18RSR when working on this research. It contains only the seven asymmetric, nonreciprocal relations. All model results on it are abysmal, but you're welcome to try :)

# Repo-level TODOs

- [X] Add exploration Notebook for WordNet (WN) structure
- [X] Add mapping of Synset codes from WN 1.7.1 all the way to 3.0.
- [ ] Move non-script code into `lib` directory
- [ ] Remove `dy.parameter()` calls (deprecated in `dynet 2.0.4`)
- [ ] Turn any remaining TODOs from here into repo issues

# Citation

```
@InProceedings{pinter-eisenstein:2018:EMNLP,
  author    = {Pinter, Yuval  and  Eisenstein, Jacob},
  title     = {{Predicting Semantic Relations using Global Graph Properties}},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  month     = {October-November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {1741--1751},
  url       = {http://www.aclweb.org/anthology/D18-1201}
}
```

# Contact
`uvp@gatech.edu`.
