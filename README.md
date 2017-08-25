# Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks

This repository consists of preprocessing and evaluation scripts used in the paper entitled Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks.
The preprocessing script cleaned corpora, tokenized and sentenced it.
Evaluation scripts can be used to measure the representativeness of a word embedding model.

---

## About the paper

Paper can be read:
https://arxiv.org/abs/1708.06025

Trained embeddings models:
http://nilc.icmc.usp.br/embeddings

### Abstract

Word embeddings have been found to provide meaningful representations for words in an efficient way; therefore, they have become common in Natural Language Processing systems. In this paper, we evaluated different word embedding models trained on a large Portuguese corpus, including both Brazilian and European variants. We trained 31 word embedding models using FastText, GloVe, Wang2Vec and Word2Vec. We evaluated them intrinsically on syntactic and semantic analogies and extrinsically on POS tagging and sentence semantic similarity tasks. The obtained results suggest that word analogies are not appropriate for word embedding evaluation; task-specific evaluations appear to be a better option. 

---

### Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Preprocessing text file](#preprocessing-text-file)
  * [Semantic evaluation](#semantic-evaluation)
  * [Syntactic and Semantic analogies](#syntactic-and-semantic-analogies)

---

## Installation
```
pip install -r requirements.txt 
```

## Usage

### Preprocessing text file

in order to train embedding models
```
python preprocessing.py <input_file.txt> <output_file.txt>
```

### Semantic evaluation

Sentence Similarity
```
python evaluate.py <embedding_model.txt> --lang
```
Parameter **--lang** can be set depending on portuguese variant chosen.

Brazilian Portuguese
```
br
```
European Portuguese
```
eu
```

### Syntactic and Semantic analogies

This method is similar to that one developed by [nlx-group](https://github.com/nlx-group/lx-dsemvectors)
```
python evaluate.py <embedding_model.txt> <testset.txt>
```
#### Brazilian Portuguese testsets

Only syntactic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr_syntactic.txt
```
Only semantic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr_semantic.txt
```
All analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogiesBr.txt
```
#### European Portuguese testsets

Only syntactic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies_syntactic.txt
```
Only semantic analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies_semantic.txt
```
All analogies
```
python evaluate.py <embedding_model.txt> analogies/testset/LX-4WAnalogies.txt
```
