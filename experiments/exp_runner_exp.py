import argparse
import os

import torch
import numpy as np

from models.geometric import run_geometric_training_example
from models.Infonce import run_inforce_training_example
from models.pair_wise import run_pairwise_training_example
from models.base import run_base_retrieval_example
from models.supcon import run_supcon_training_example
from eval.process_tatoeba import extract_parallel_maxcover
from eval.eval_runner import run_full_eval
from eval.embedder import OllamaEmbedder, DiskEmbeddingCache, CachedEmbedder
from .utils import remove_nones_parallel, torch_embedder_to_numpy
from models.gcca import run_gcca_training_example
# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(
    exp_number: int,
    seed: int,
    data_size: int,
    epochs: int,
    n_min: int,
    n_max: int,
    K_geo: int,
    r: int,
    num_sentences_for_retreival: int,
):
    print(f"----------------Running Experiment #{exp_number}----------------")
    DATA_DIR = "data/merged.tsv"
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

    EVAL_SENTENCES_DIR = "data/eval/sentences.csv"
    EVAL_LINKS_DIR = "data/eval/links.csv"

    # Other training hyperparams (keep fixed unless you want to expose them too)
    BATCH = int(data_size/epochs)
    BATCH = min(BATCH, 5120)
    lr = 1e-3
    tau = 0.07

    # ---- Training ----
    print(f"--------{exp_number}--------Training-------{exp_number}---------")

    print("--------vecmap--------")
    geometric = run_geometric_training_example(
        tsv_path=DATA_DIR,
        seed=seed,
        pair_subset_size=data_size,
        sentence_subset_size=data_size * 2,
        K=K_geo,
        r=r,
        n_min=n_min,
        n_max=n_max,
        NUM_RUNS=epochs,
    )

    # Determine D and K_grad from geometric output
    D, K_grad = geometric.shape
    supcon, languages = run_supcon_training_example(
        tsv_path=DATA_DIR,
        seed=seed,
        subset_size=data_size,
        d=D,
        k=K_grad,
        batch_size_pairs=BATCH,
        epochs=epochs,
        lr=lr,
        tau=tau,
        device=DEVICE_STR,
    )
   
    gcca = run_gcca_training_example(DATA_DIR, seed, ollama_model="granite-embedding:278m", cache_dir="./emb_cache_granite")

    # ---- V extraction ----
    V_gcca = {
        lang: gcca.projs[lang].weight.detach().cpu().numpy().T
        for lang in gcca.projs
    }
    
    name_to_V = {
        "gcca" : V_gcca
    }

    # ---- Retrieval groups ----
    print(f"------{exp_number}----------Evaluation------{exp_number}----------")
    print("--------Eval Data Retreival--------")
    retrieval_groups = extract_parallel_maxcover(
        EVAL_SENTENCES_DIR,
        EVAL_LINKS_DIR,
        languages,
        n_sentences=num_sentences_for_retreival,
        min_langs=5,
        fill_missing=None,
    )
    retrieval_groups = remove_nones_parallel(retrieval_groups)

    # ---- Embedder with cache ----
    print("--------Eval Data Embedding--------")
    embed_base = OllamaEmbedder(model="granite-embedding:278m")
    cache_dir = os.path.abspath("./emb_cache_granite")
    cache = DiskEmbeddingCache(cache_dir)
    embed_src = CachedEmbedder(embed_base, cache)
    embed_fn = torch_embedder_to_numpy(embed_src)

    # ---- Eval ----
    print("--------Running Eval--------")
    out_path = run_full_eval(
        exp_number=exp_number,
        name_to_V=name_to_V,
        embed_fn=embed_fn,
        projection_mode="subspace_coords",
        retrieval_groups=retrieval_groups,
        retrieval_langs=None,  # only correct if retrieval_groups are (lang,sent) tuples
        retrieval_K=10,
        retrieval_trials=500,
        seed=seed,
    )

    print(f"----------------Finished Experiment #{exp_number}, Wrote: {out_path}----------------")
    return out_path


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run universal_embeddings experiment.")

    parser.add_argument("--exp", type=str, default=0, help="Experiment number")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--data_size", type=int, default=500, help="Training data subset size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs/runs")

    parser.add_argument("--n_min", type=int, default=1, help="Geometric: n_min")
    parser.add_argument("--n_max", type=int, default=30, help="Geometric: n_max")
    parser.add_argument("--K", type=int, default=50, help="Geometric: K (subspace dim)")
    parser.add_argument("--r", type=int, default=128, help="Geometric: r")

    parser.add_argument(
        "--n_sent_ret",
        type=int,
        default=20,
        help="Number of sentence groups used for retrieval set construction",
    )

    args = parser.parse_args()

    # Basic sanity checks
    if args.data_size <= 0:
        raise ValueError("--data_size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.n_min < 0:
        raise ValueError("--n_min must be >= 0")
    if args.n_max <= args.n_min:
        raise ValueError("--n_max must be > --n_min")
    if args.K <= 0:
        raise ValueError("--K must be > 0")
    if args.r <= 0:
        raise ValueError("--r must be > 0")
    if args.n_sent_ret <= 0:
        raise ValueError("--n_sent_ret must be > 0")

    run_experiment(
        exp_number=args.exp,
        seed=args.seed,
        data_size=args.data_size,
        epochs=args.epochs,
        n_min=args.n_min,
        n_max=args.n_max,
        K_geo=args.K,
        r=args.r,
        num_sentences_for_retreival=args.n_sent_ret,
    )


if __name__ == "__main__":
    main()
