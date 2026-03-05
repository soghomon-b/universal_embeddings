from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np


# ----------------------------
# Utilities
# ----------------------------


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [d] or [n,d]
    b: [m,d]
    returns: [m] if a is [d], else [n,m]
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        return b @ a
    return a @ b.T


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------
# Projection into "universal space"
# ----------------------------



def project_to_universal(
    X: np.ndarray,
    V: np.ndarray,
    mode: str = "subspace_coords",
) -> np.ndarray:
    """
    Project base embeddings X into a universal space using V.

    X: [n, d]
    V:
      - If mode="linear": V is [d, du] (a linear map W)
      - If mode="subspace_coords": V is [d, k] (a basis); output is coordinates [n, k]
      - If mode="subspace_recon": V is [d, k] (a basis); output is reconstruction [n, d]
      - If V is None: no projection (base embeddings)

    Returns: projected embeddings, L2-normalized row-wise.
    """

    X = np.asarray(X, dtype=np.float32)

    # --- BASE CASE: no projection ---
    if V is None:
        return l2_normalize(X, axis=-1)

    V = np.asarray(V, dtype=np.float32)

    if mode == "linear":
        Z = X @ V  # [n, du]

    elif mode == "subspace_coords":
        Z = X @ V  # coordinates in basis, [n, k]

    elif mode == "subspace_recon":
        # reconstruction back in d-dim: (X V) V^T
        Z = (X @ V) @ V.T  # [n, d]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return l2_normalize(Z, axis=-1)


# ----------------------------
# Data model
# ----------------------------

Sentence = str
Lang = str
ParallelGroup = Sequence[Sentence]  # as user said: list of sentences
ParallelGroupWithLang = Sequence[Tuple[Lang, Sentence]]  # optional richer format


@dataclass
class RetrievalTrialResult:
    correct_top1: bool
    rank: int  # 1-based rank of the true positive
    group_id: int
    anchor_idx: int
    pos_idx: int
    candidate_indices: List[
        Tuple[int, int]
    ]  # list of (group_id, sent_idx) for the candidate pool


@dataclass
class EvalReport:
    accuracy_at_1: float
    mrr: float
    recall_at_k: Dict[int, float]
    n_trials: int
    details: Optional[List[RetrievalTrialResult]] = None


# ----------------------------
# Core evaluator
# ----------------------------


class UniversalEmbeddingRetrievalEvaluator:
    """
    Implements your evaluation:
      - pick anchor from a group
      - build candidate set of size K with 1 positive + K-1 negatives (from other groups)
      - optional language constraints if langs are provided
      - embed -> project -> cosine retrieval
      - compute mean metrics
    """

    def __init__(
        self,
        V: np.ndarray,
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        projection_mode: str = "subspace_coords",
        batch_size: int = 64,
    ):
        """
        V: projection matrix / basis
        embed_fn: function mapping list[str] -> np.ndarray [n, d]
                 (You plug in your LLM encoder here.)
        projection_mode: see project_to_universal
        """
        self.V = np.asarray(V, dtype=np.float32)
        self.embed_fn = embed_fn
        self.projection_mode = projection_mode
        self.batch_size = batch_size

    def _batched_embed(self, sentences: List[str]) -> np.ndarray:
        outs = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = sentences[i : i + self.batch_size]
            e = self.embed_fn(chunk)
            e = np.asarray(e, dtype=np.float32)
            outs.append(e)
        return np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)

    def _prepare_groups(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        langs: Optional[Sequence[Sequence[str]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[str]]]]:
        """
        Accepts either:
          - groups: List[List[str]] and optional langs: same shape
          - OR groups: List[List[(lang, sent)]] with langs=None
        Returns:
          sentences_groups, langs_groups (or None)
        """
        # Case 1: already given as (lang, sentence)
        if (
            groups
            and len(groups) > 0
            and len(groups[0]) > 0
            and isinstance(groups[0][0], tuple)
        ):
            sent_groups: List[List[str]] = []
            lang_groups: List[List[str]] = []
            for g in groups:  # type: ignore
                lgs, sgs = zip(*g)  # type: ignore
                sent_groups.append(list(sgs))
                lang_groups.append(list(lgs))
            return sent_groups, lang_groups

        # Case 2: plain sentences with optional langs
        sent_groups = [list(g) for g in groups]  # type: ignore
        if langs is None:
            return sent_groups, None
        lang_groups = [list(lg) for lg in langs]
        if len(lang_groups) != len(sent_groups):
            raise ValueError("langs must have same number of groups as groups")
        for i, (g, lg) in enumerate(zip(sent_groups, lang_groups)):
            if len(g) != len(lg):
                raise ValueError(f"langs group {i} must match sentence group length")
        return sent_groups, lang_groups

    def _index_all_sentences(
        self,
        sent_groups: List[List[str]],
        lang_groups: Optional[List[List[str]]],
    ) -> Tuple[List[Tuple[int, int]], List[str], Optional[List[str]]]:
        """
        Flatten everything:
          flat_indices: list of (group_id, sent_idx)
          flat_sentences: list of sentences aligned to flat_indices
          flat_langs: optional list of langs aligned
        """
        flat_indices: List[Tuple[int, int]] = []
        flat_sentences: List[str] = []
        flat_langs: Optional[List[str]] = [] if lang_groups is not None else None

        for gi, g in enumerate(sent_groups):
            for si, s in enumerate(g):
                flat_indices.append((gi, si))
                flat_sentences.append(s)
                if flat_langs is not None:
                    flat_langs.append(lang_groups[gi][si])  # type: ignore

        return flat_indices, flat_sentences, flat_langs

    def _sample_candidate_pool(
        self,
        *,
        sent_groups: List[List[str]],
        lang_groups: Optional[List[List[str]]],
        anchor: Tuple[int, int],
        K: int,
        hard_negatives: bool,
        hard_pool_size: int,
        base_embs_flat: Optional[np.ndarray],
        flat_indices: List[Tuple[int, int]],
        flat_langs: Optional[List[str]],
        rng: random.Random,
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        """
        Returns:
          candidate_indices: length K, includes 1 positive
          pos_index: (group_id, sent_idx) of the positive
        """
        anchor_g, anchor_s = anchor
        anchor_lang = None
        if lang_groups is not None:
            anchor_lang = lang_groups[anchor_g][anchor_s]

        # Choose positive from same group but different language if possible
        pos_candidates = [
            (anchor_g, j) for j in range(len(sent_groups[anchor_g])) if j != anchor_s
        ]
        if lang_groups is not None:
            pos_candidates = [
                (anchor_g, j)
                for (anchor_g, j) in pos_candidates
                if lang_groups[anchor_g][j] != anchor_lang
            ]
        if not pos_candidates:
            raise RuntimeError(
                "No valid positive candidate (need at least 2 sentences per group, "
                "and if using langs, at least 2 different langs)."
            )
        pos = rng.choice(pos_candidates)

        # Build negative pool: from other groups
        neg_pool: List[Tuple[int, int]] = []
        for gi, si in flat_indices:
            if gi == anchor_g:
                continue  # exclude same meaning group
            if lang_groups is not None:
                # enforce: negatives not in anchor language
                if lang_groups[gi][si] == anchor_lang:
                    continue
                # optional extra: avoid positive language too (often cleaner)
                if lang_groups[gi][si] == lang_groups[pos[0]][pos[1]]:
                    continue
            neg_pool.append((gi, si))

        if len(neg_pool) < K - 1:
            raise RuntimeError(
                f"Not enough negatives to sample {K-1} items. "
                f"Have {len(neg_pool)} valid negatives."
            )

        # Optionally: hard negatives by base embedding similarity to anchor
        if hard_negatives:
            if base_embs_flat is None:
                raise ValueError(
                    "hard_negatives=True requires base_embs_flat (precomputed base embeddings)."
                )

            # locate anchor in flat list
            anchor_flat_idx = flat_indices.index(anchor)
            a_vec = l2_normalize(base_embs_flat[anchor_flat_idx], axis=-1)

            # compute similarity to all negs
            neg_flat_idxs = [flat_indices.index(x) for x in neg_pool]
            neg_vecs = l2_normalize(base_embs_flat[neg_flat_idxs], axis=-1)
            sims = cosine_sim(a_vec, neg_vecs)  # [len(neg_pool)]

            # take top hard_pool_size and sample from them
            hard_pool_size = min(hard_pool_size, len(neg_pool))
            top_idx = np.argpartition(-sims, hard_pool_size - 1)[:hard_pool_size]
            hard_pool = [neg_pool[i] for i in top_idx.tolist()]
            negatives = rng.sample(hard_pool, K - 1)
        else:
            negatives = rng.sample(neg_pool, K - 1)

        # Candidate pool = [positive + negatives], shuffle
        candidates = [pos] + negatives
        rng.shuffle(candidates)

        # Optional: ensure all candidate languages are distinct (stronger control)
        if lang_groups is not None:
            langs_in_candidates = [lang_groups[gi][si] for (gi, si) in candidates]
            # If duplicates happen, you can resample a few times; we keep it simple:
            # (This constraint can be too strict depending on your data.)
            # If you want it strict, uncomment the resampling loop below.

            # for _ in range(20):
            #     if len(set(langs_in_candidates)) == len(langs_in_candidates):
            #         break
            #     negatives = rng.sample(neg_pool, K - 1)
            #     candidates = [pos] + negatives
            #     rng.shuffle(candidates)
            #     langs_in_candidates = [lang_groups[gi][si] for (gi, si) in candidates]

        return candidates, pos

    def evaluate(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        K: int = 10,
        n_trials: int = 1000,
        seed: int = 0,
        hard_negatives: bool = False,
        hard_pool_size: int = 200,
        return_details: bool = False,
        recall_ks: Sequence[int] = (1, 3, 5),
        precomputed_base_embeddings: Optional[np.ndarray] = None,
    ) -> EvalReport:
        """
        groups: list of parallel groups (same meaning).
                Either List[List[str]] OR List[List[(lang, sent)]]
        langs: optional languages with same shape as groups if groups are plain sentences.
        K: candidate set size per trial (1 positive + K-1 negatives)
        n_trials: number of retrieval trials
        hard_negatives: sample negatives from top similar ones (requires base embeddings)
        precomputed_base_embeddings:
              if provided, must be aligned to the flattened sentence list in groups,
              shape [N_total, d]. Used for hard negative mining.
              If not provided and hard_negatives=True, we will compute it once using embed_fn.
        """
        assert K >= 2, "K must be >= 2"
        rng = random.Random(seed)

        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        flat_indices, flat_sentences, flat_langs = self._index_all_sentences(
            sent_groups, lang_groups
        )

        # Precompute base embeddings if needed (for hard negative mining)
        base_embs_flat = None
        if hard_negatives:
            if precomputed_base_embeddings is not None:
                base_embs_flat = np.asarray(
                    precomputed_base_embeddings, dtype=np.float32
                )
            else:
                base_embs_flat = self._batched_embed(flat_sentences)
            if base_embs_flat.shape[0] != len(flat_sentences):
                raise ValueError(
                    "precomputed_base_embeddings must match total number of sentences."
                )

        # Metrics accumulators
        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}
        details: List[RetrievalTrialResult] = []

        # Helper to draw a valid anchor (must have a positive available)
        valid_anchors: List[Tuple[int, int]] = []
        for gi, g in enumerate(sent_groups):
            if len(g) < 2:
                continue
            if lang_groups is None:
                valid_anchors.extend([(gi, si) for si in range(len(g))])
            else:
                # need at least two distinct langs within group to allow pos with different lang
                if len(set(lang_groups[gi])) >= 2:
                    valid_anchors.extend([(gi, si) for si in range(len(g))])

        if not valid_anchors:
            raise RuntimeError(
                "No valid anchors. Ensure each group has >=2 sentences "
                "and (if using langs) at least 2 different languages per group."
            )

        # Run trials
        for t in range(n_trials):
            anchor = rng.choice(valid_anchors)
            anchor_g, anchor_s = anchor

            candidates, pos = self._sample_candidate_pool(
                sent_groups=sent_groups,
                lang_groups=lang_groups,
                anchor=anchor,
                K=K,
                hard_negatives=hard_negatives,
                hard_pool_size=hard_pool_size,
                base_embs_flat=base_embs_flat,
                flat_indices=flat_indices,
                flat_langs=flat_langs,
                rng=rng,
            )

            # Collect sentences for embedding
            anchor_sentence = sent_groups[anchor_g][anchor_s]
            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]

            # Embed (base)
            base = self._batched_embed([anchor_sentence] + candidate_sentences)
            anchor_base = base[0:1]  # [1,d]
            cand_base = base[1:]  # [K,d]

            # Project to universal space & normalize
            anchor_u = project_to_universal(
                anchor_base, self.V, mode=self.projection_mode
            )[
                0
            ]  # [du]
            cand_u = project_to_universal(
                cand_base, self.V, mode=self.projection_mode
            )  # [K,du]

            # Cosine similarity (since vectors are normalized, dot = cosine)
            sims = cosine_sim(anchor_u, cand_u)  # [K]

            # Rank (descending similarity)
            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

            # Find rank of true positive
            rank = 1 + ranked_candidates.index(pos)
            is_top1 = rank == 1

            correct1 += int(is_top1)
            mrr_sum += 1.0 / rank
            for k in recall_ks:
                if rank <= k:
                    recall_hits[k] += 1

            if return_details:
                details.append(
                    RetrievalTrialResult(
                        correct_top1=is_top1,
                        rank=rank,
                        group_id=anchor_g,
                        anchor_idx=anchor_s,
                        pos_idx=pos[1],
                        candidate_indices=candidates,
                    )
                )

        acc1 = correct1 / n_trials
        mrr = mrr_sum / n_trials
        recall = {k: recall_hits[k] / n_trials for k in recall_ks}

        return EvalReport(
            accuracy_at_1=acc1,
            mrr=mrr,
            recall_at_k=recall,
            n_trials=n_trials,
            details=details if return_details else None,
        )


# ----------------------------
# Example usage (you replace embed_fn)
# ----------------------------


def dummy_embed_fn(sentences: List[str]) -> np.ndarray:
    """
    Replace this with your real LLM embedder.
    This dummy version is deterministic-ish and only for demonstrating the pipeline.
    """
    # e.g., hash -> pseudo-random vector
    d = 768
    out = np.zeros((len(sentences), d), dtype=np.float32)
    for i, s in enumerate(sentences):
        rs = np.random.RandomState(abs(hash(s)) % (2**32))
        out[i] = rs.normal(size=d).astype(np.float32)
    return l2_normalize(out, axis=-1)


if __name__ == "__main__":
    set_seed(0)

    # Suppose your universal "V" is a basis [d,k] (subspace) or a linear map [d,du]
    d = 768
    k = 256
    V = np.random.randn(d, k).astype(np.float32)
    # (If V is meant to be a basis, you might orthonormalize it in your own pipeline.)

    # groups: list of lists of parallel sentences (same meaning, different languages)
    groups = [
        ["Hello world (en)", "Bonjour le monde (fr)", "Hola mundo (es)"],
        ["How are you? (en)", "Comment ça va? (fr)", "¿Cómo estás? (es)"],
        ["I like coffee (en)", "J'aime le café (fr)", "Me gusta el café (es)"],
    ]

    # Optional language labels (lets you enforce “not same language” constraints properly)
    langs = [
        ["en", "fr", "es"],
        ["en", "fr", "es"],
        ["en", "fr", "es"],
    ]

    evaluator = UniversalEmbeddingRetrievalEvaluator(
        V=V,
        embed_fn=dummy_embed_fn,
        projection_mode="subspace_coords",  # or "linear", "subspace_recon"
        batch_size=64,
    )

    report = evaluator.evaluate(
        groups,
        langs=langs,  # omit if you truly don't have language ids
        K=3,  # candidate pool size per trial
        n_trials=200,  # increase for real experiments
        seed=123,
        hard_negatives=False,  # set True + provide lots of data if you want hard negatives
        recall_ks=(1, 3, 5),
        return_details=False,
    )

    print("Accuracy@1:", report.accuracy_at_1)
    print("MRR:", report.mrr)
    print("Recall@k:", report.recall_at_k)
