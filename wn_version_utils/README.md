This directory contains a few tools and data from my painful attempts to reconcile WordNet versions.

The tl;dr is:
* WN18RR and nltk's WordNet corpus both use version 3.0.
* The AutoExtend embeddings I intended to use for synsets claim to correspond to version 2.1.
* Synsets are uniquely identified via an `offset` attribute, which changes across versions.
* The AutoExtend offsets actually correspond to version 1.7.1.
* A resource called SenseMap helps match synsets across *adjacent* versions.
* I collected all of these from version 1.7.1 and produced a joined mapping.

`embeddings_21-30.py` helps specifically with the embeddings scenario. Let me know if you need to know exactly where to get all the auxiliary data for this operation.

`senses-cleanmap.sh` is the script I used to map sensemap files into tabular offset-to-offset form.

`join-sensemaps.sh` is the iterative (and lossy) transformation from 1.7.1 through 2.0 and 2.1 all the way to 3.0.

Here are the coverage stats from 1.7.1's AutoExtend offsets to 3.0:

| POS | Synsets | Found in 1.7.1 | Non-Zero | Loss (including zeros) |
|-----|---------|----------------|----------|------------------------|
| N   |  82,115 |         75,658 |   41,999 | 48.9% (7.9%)           |
| V   |  13,767 |         13,048 |   11,048 | 19.8% (5.2%)           |
| A*  |  18,156 |         17,162 |   13,720 | 24.4% (5.5%)           |
| R*  |   3,621 |          3,531 |    2,976 | 17.8% (2.5%)           |
| All | 117,659 |        109,399 | 69,743** | 40.7% (7.0%)           |

Notes:
* Lemma-based matching for adjectives (719) and adverbs (90) performed for cases where the 1.7.1 WordNet version did not contain as many senses for a lemma as 3.0.
* AutoExtend (Rothe and Schütze, 2015) report 73,844 Synsets with word2vec embeddings, and others the algorithm helped further, so the difference (>=4,101) may have to do with WordNet version ID alignment.

The zipped `.tsv` files are the outputs of this script.

Enjoy!

--- Yuval Pinter, October 2018
