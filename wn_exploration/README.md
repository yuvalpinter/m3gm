# WordNet Exploration Bonus Content

This directory contains two general WordNet tools.

[`wordnet_analysis.ipynb`](wn_exploration/wordnet_analysis.ipynb) is a walk-through of some of the graph properties of WordNet, including some useful ways for extracting data from NLTK's WordNet corpus into `networkx` format.

[`create_wordnet_data.py`](wn_exploration/create_wordnet_data.py) is a script for creating a convenient dataset for the entire WordNet (not just WN18/RR like in the main directory). It outputs a `.pkl` file with a simple structure, where the relations are excoded as a `spicy` sparse matrix.

Enjoy!

--- Yuval Pinter, October 2018
