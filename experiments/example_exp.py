import torch
import numpy as np

from models.geometric import run_geometric_training_example
from models.Infonce import run_inforce_training_example
from models.pair_wise import run_pairwise_training_example
from models.supcon import run_supcon_training_example
from eval.process_tatoeba import extract_parallel_maxcover
from eval.eval_runner import run_full_eval
from eval.embedder import OllamaEmbedder, DiskEmbeddingCache, CachedEmbedder

EXP_NUMBER = 0




DATA_DIR = "data/merged.tsv"
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_SENTENCES_DIR = "data/eval/sentences.csv"
EVAL_LINKS_DIR = "data/eval/links.csv"

SEED = 12

# --- choose real hyperparams ---
SIZE = 50
BATCH = 256
D = 128  # let supcon auto-detect if you want
K_grad = 50
EPOCHS = 5
lr = 1e-3
tau = 0.07

# --- geometric (NO trailing commas) ---
pair_subset_size = 10
sentence_subset_size = 20
K_geo = 50
r = 128
n_min = 0
n_max = 0

num_sentences_for_retreival = 20

# Training
model = run_inforce_training_example(
        "E:/thesis_work/nllb_sampled/merged.tsv", subset_size=100, d=10, k=11
    )
inforce = run_inforce_training_example(DATA_DIR, SEED, SIZE, D, K_grad, DEVICE_STR)
pairwise = run_pairwise_training_example(DATA_DIR, SEED, SIZE, D, K_grad, DEVICE_STR)

supcon, languages = run_supcon_training_example(
    tsv_path=DATA_DIR,
    seed=SEED,
    subset_size=SIZE,
    d=D,
    k=K_grad,
    batch_size_pairs=BATCH,
    epochs=EPOCHS,
    lr=lr,
    tau=tau,
    device=DEVICE_STR,
)

geometric = run_geometric_training_example(
    tsv_path=DATA_DIR,
    seed=SEED,
    pair_subset_size=pair_subset_size,
    sentence_subset_size=sentence_subset_size,
    K=K_geo,
    r=r,
    n_min=n_min,
    n_max=n_max,
    NUM_RUNS=EPOCHS,
)

# V extraction
V_inforce = inforce.proj.weight.detach().float().cpu().T
V_pairwise = pairwise.proj.weight.detach().float().cpu().T
V_supcon = supcon.proj.weight.detach().float().cpu().T
V_geometric = torch.tensor(geometric, dtype=torch.float32).cpu()

name_to_V = {
    "infonce": V_inforce,
    "pairwise": V_pairwise,
    "supcon": V_supcon,
    "geometric": V_geometric,
}

retrieval_groups = extract_parallel_maxcover(
    EVAL_SENTENCES_DIR,
    EVAL_LINKS_DIR,
    languages,
    n_sentences=num_sentences_for_retreival,
    min_langs=5,
    fill_missing=None,
)


def torch_embedder_to_numpy(embedder):
    def _embed_fn(sentences: list[str]) -> np.ndarray:
        with torch.no_grad():
            x = embedder(sentences)
            return x.detach().cpu().numpy().astype(np.float32)

    return _embed_fn


embed_base = OllamaEmbedder(model="granite-embedding:278m")
cache = DiskEmbeddingCache("./emb_cache_granite278m")
embed_src = CachedEmbedder(embed_base, cache)
embed_fn = torch_embedder_to_numpy(embed_src)

out_path = run_full_eval(
    exp_number=EXP_NUMBER,
    name_to_V=name_to_V,
    embed_fn=embed_fn,
    projection_mode="subspace_coords",
    retrieval_groups=retrieval_groups,
    retrieval_langs=None,  # only correct if retrieval_groups are (lang,sent) tuples
    retrieval_K=10,
    retrieval_trials=500,
    seed=SEED,
)

print(f"Finished Experiment #{EXP_NUMBER}, Wrote: {out_path}")
