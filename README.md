# universal_embeddings — thesis experimentation playground

This repository is an experimentation playground for my thesis on **universal multilingual embeddings**, focused on comparing:

- **Geometric approaches** (subspace construction / SVD-style methods)
vs.
- **Gradient-based approaches** (InfoNCE, pairwise contrastive, SupCon)

The goal is to test whether geometric universal-subspace methods can match (or be equivalent in effect to) gradient-based contrastive training, both empirically and theoretically.

## What this runner does

The main experiment script:

1. Trains / constructs a projection subspace `V` using multiple methods
   - geometric → run_geometric_training_example(...)
   - infonce → run_inforce_training_example(...)
   - pairwise → run_pairwise_training_example(...)
   - supcon → run_supcon_training_example(...)

2. Builds a multilingual retrieval evaluation set
   - Reads data/eval/sentences.csv and data/eval/links.csv
   - Creates evaluation retrieval groups

3. Embeds sentences using an Ollama embedding model with disk caching
   - Model: granite-embedding:278m
   - Cache directory: ./emb_cache_granite278m

4. Evaluates all learned subspaces
   - Uses subspace coordinate projection
   - Writes experiment results to disk

## Repository layout

Expected structure:

.
├── experiments/
│   └── exp_runner.py
├── models/
│   ├── geometric.py
│   ├── Infonce.py
│   ├── pair_wise.py
│   └── supcon.py
├── eval/
│   ├── process_tatoeba.py
│   ├── eval_runner.py
│   └── embedder.py
├── data/ (not on github)
│   ├── merged.tsv
│   └── eval/
│       ├── sentences.csv
│       └── links.csv
└── emb_cache_granite278m/

## Data format

### Training data: data/merged.tsv

Each row:

src_lang<TAB>tgt_lang<TAB>src_sentence<TAB>tgt_sentence

### Evaluation data

data/eval/sentences.csv

data/eval/links.csv

Used to construct multilingual retrieval groups.

## Requirements

Python 3.10+ recommended

Required packages typically include:

- torch
- numpy

Ollama must be installed and running.

Required embedding model:

granite-embedding:278m

Pull model using:

ollama pull granite-embedding:278m

## Installation

Create environment:

python -m venv venv

Activate:

Linux/Mac:
source venv/bin/activate

Windows:
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Running experiments

Example run:

python -m experiments.exp_runner \
  --exp 0 \
  --seed 12 \
  --data_size 500 \
  --epochs 5 \
  --n_min 1 \
  --n_max 30 \
  --K 50 \
  --r 128 \
  --n_sent_ret 20

## Arguments

--exp
Experiment id.

--seed
Random seed.

--data_size
Number of training pairs sampled.

--epochs
Training epochs.

--n_min
Geometric minimum dataset size.

--n_max
Geometric maximum dataset size.

--K
Geometric subspace dimension.

--r
Geometric method parameter.

--n_sent_ret
Number of sentence groups used for retrieval evaluation.

## Outputs

The experiment prints the output file path after completion.

Results include evaluation scores for:

- geometric
- infonce
- pairwise
- supcon

## Embedding cache

Embeddings are cached in:

./emb_cache_granite278m

Once cached, future runs reuse embeddings.

## Reproducibility

Reproducibility is controlled by:

- random seed
- deterministic subset sampling
- cached embeddings

## Thesis context

This repository supports experiments comparing:

Geometric universal subspace extraction

vs

Gradient-based contrastive learning projections

The thesis hypothesis tested here is that geometric methods can be equivalent in effect to gradient-based methods when trained on the exact same dataset, on multilingual retrieval tasks.
