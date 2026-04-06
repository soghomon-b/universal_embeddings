from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any


from models.muse import BitextSentenceEncoder
from models.ot import SinkhornOT
from models.dvcca import CachedBitextDVCCA
import numpy as np
import torch
import torch.nn.functional as F
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
      - Base case: V is None (or a 0-d array from None) => no projection

    Returns: projected embeddings, L2-normalized row-wise.
    """
    X = np.asarray(X, dtype=np.float32)

    # --- BASE CASE: no projection ---
    # Handle both Python None and accidental np.array(None) / 0-d arrays.
    if V is None:
        return l2_normalize(X, axis=-1)

    # If V came in as a numpy scalar/0-d array (e.g., np.asarray(None) somewhere),
    # treat it as "no projection" as well.
    if isinstance(V, np.ndarray) and V.ndim == 0:
        return l2_normalize(X, axis=-1)

    # Now safely cast V to float32 matrix
    V = np.asarray(V, dtype=np.float32)

    # Extra safety: if casting still produced 0-d (shouldn't, but just in case)
    if V.ndim == 0:
        return l2_normalize(X, axis=-1)

    if mode == "linear":
        Z = X @ V  # [n, du]
    elif mode == "subspace_coords":
        Z = X @ V  # [n, k]
    elif mode == "subspace_recon":
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
        if isinstance(V, dict):
            self.V = {lang: np.asarray(mat, dtype=np.float32) for lang, mat in V.items()}
        elif isinstance(V, BitextSentenceEncoder) or isinstance(V, SinkhornOT) or isinstance(V, CachedBitextDVCCA):
            self.V = V
        else:
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
    def evaluate_2(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        n_trials: int = 500,
        K: int = 10,
        recall_ks: Sequence[int] = (1, 3, 5),
        seed: int = 0,
        hard_negatives: bool = False,
        hard_pool_size: int = 100,
        return_details: bool = False,
    ) -> EvalReport:
        """
        Same as evaluate(), but expects self.V to be a dict-like object:
            self.V[lang] -> projection matrix for that language

        Works for methods like GCCA / VecMap where each language has its own map.

        Assumes:
        - groups/langs provide language for each sentence
        - each self.V[lang] is compatible with project_to_universal(X, V_lang, ...)
        """
        sent_groups, lang_groups = self._prepare_groups(groups, langs)

        if lang_groups is None:
            raise ValueError(
                "evaluate_2 requires language information. "
                "Pass langs=... or use groups formatted as [(lang, sentence), ...]."
            )

        rng = random.Random(seed)

        flat_indices, flat_sentences, flat_langs = self._index_all_sentences(
            sent_groups, lang_groups
        )

        # Optional hard-negative precompute stays in base space, same as evaluate()
        base_embs_flat = None
        if hard_negatives:
            base_embs_flat = self._batched_embed(flat_sentences)
            base_embs_flat = l2_normalize(base_embs_flat, axis=-1)

        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}
        details: List[RetrievalTrialResult] = []

        for _ in range(n_trials):
            # sample anchor group with at least 2 sentences
            valid_groups = [gi for gi, g in enumerate(sent_groups) if len(g) >= 2]
            if not valid_groups:
                raise ValueError("Need at least one group with 2+ sentences.")
            anchor_g = rng.choice(valid_groups)

            # sample anchor sentence
            anchor_s = rng.randrange(len(sent_groups[anchor_g]))
            anchor = (anchor_g, anchor_s)

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

            # Sentences + langs
            anchor_sentence = sent_groups[anchor_g][anchor_s]
            anchor_lang = lang_groups[anchor_g][anchor_s]

            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]
            candidate_langs = [lang_groups[gi][si] for (gi, si) in candidates]

            # Embed in base space
            base = self._batched_embed([anchor_sentence] + candidate_sentences)
            anchor_base = base[0:1]   # [1, d]
            cand_base = base[1:]      # [K, d]

            # Project anchor with its own language matrix
            if anchor_lang not in self.V:
                raise KeyError(f"Missing projection for anchor language: {anchor_lang}")

            anchor_u = project_to_universal(
                anchor_base,
                self.V[anchor_lang],
                mode=self.projection_mode,
            )[0]  # [du]

            # Project candidates language-by-language
            cand_u = np.zeros((len(candidates), anchor_u.shape[0]), dtype=np.float32)

            unique_langs = sorted(set(candidate_langs))
            for lang in unique_langs:
                if lang not in self.V:
                    raise KeyError(f"Missing projection for candidate language: {lang}")

                idxs = [i for i, lg in enumerate(candidate_langs) if lg == lang]
                X_lang = cand_base[idxs]  # [m, d]

                Z_lang = project_to_universal(
                    X_lang,
                    self.V[lang],
                    mode=self.projection_mode,
                )  # [m, du]

                cand_u[idxs] = Z_lang

            # cosine similarity
            sims = cosine_sim(anchor_u, cand_u)  # [K]

            # rank
            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

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
    
    def _batched_muse_embed(self, sentences, *, encoder, device="cpu", batch_size=32):
        import torch
        all_z = []

        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]

                # adapt this to your model API
                z = encoder.encode_batch(batch, device=device)   # or your own wrapper

                if isinstance(z, torch.Tensor):
                    z = z.detach().cpu().numpy()

                all_z.append(z)

        return np.vstack(all_z).astype(np.float32)
    def evaluate_3(
        self,
        groups,
        *,
        encoder,
        langs=None,
        n_trials: int = 500,
        K: int = 10,
        recall_ks=(1, 3, 5),
        seed: int = 0,
        hard_negatives: bool = False,
        hard_pool_size: int = 100,
        return_details: bool = False,
        device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        rng = random.Random(seed)

        flat_indices, flat_sentences, flat_langs = self._index_all_sentences(
            sent_groups, lang_groups
        )

        base_embs_flat = None
        if hard_negatives:
            # for MUSE, the encoder output itself is the base/universal space
            base_embs_flat = self._batched_muse_embed(
                flat_sentences, encoder=encoder, device=device
            )
            base_embs_flat = l2_normalize(base_embs_flat, axis=-1)

        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}
        details = []

        valid_anchors = []
        for gi, g in enumerate(sent_groups):
            if len(g) < 2:
                continue
            if lang_groups is None:
                valid_anchors.extend([(gi, si) for si in range(len(g))])
            else:
                if len(set(lang_groups[gi])) >= 2:
                    valid_anchors.extend([(gi, si) for si in range(len(g))])

        if not valid_anchors:
            raise RuntimeError("No valid anchors.")

        for _ in range(n_trials):
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

            anchor_sentence = sent_groups[anchor_g][anchor_s]
            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]

            Z = self._batched_muse_embed(
                [anchor_sentence] + candidate_sentences,
                encoder=encoder,
                device=device,
            )
            Z = l2_normalize(Z, axis=-1)

            anchor_u = Z[0]
            cand_u = Z[1:]

            sims = cosine_sim(anchor_u, cand_u)
            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

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
    def evaluate_4(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        ot_model,
        langs: Optional[Sequence[Sequence[str]]] = None,
        n_trials: int = 500,
        K: int = 10,
        recall_ks: Sequence[int] = (1, 3, 5),
        seed: int = 0,
        hard_negatives: bool = False,
        hard_pool_size: int = 100,
        return_details: bool = False,
        transport_candidates: bool = False,
        device: Optional[str] = None,
    ) -> EvalReport:
        """
        OT-specific retrieval evaluation.

        For each trial:
        1. sample anchor + candidate pool (1 positive, K-1 negatives)
        2. embed anchor and candidates in base space
        3. transport anchor toward candidate pool with Sinkhorn OT
        4. rank candidates by cosine similarity to transported anchor

        Args:
            ot_model:
                instance of SinkhornOT (or compatible model)
            transport_candidates:
                if True, also transport candidates toward the transported anchor.
                Usually keep this False and only transport the anchor.
        """
        if K < 2:
            raise ValueError("K must be >= 2")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        rng = random.Random(seed)

        flat_indices, flat_sentences, flat_langs = self._index_all_sentences(
            sent_groups, lang_groups
        )

        # Hard negatives are mined in base space, same as evaluate() / evaluate_2()
        base_embs_flat = None
        if hard_negatives:
            base_embs_flat = self._batched_embed(flat_sentences)
            base_embs_flat = l2_normalize(base_embs_flat, axis=-1)

        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}
        details: List[RetrievalTrialResult] = []

        valid_anchors: List[Tuple[int, int]] = []
        for gi, g in enumerate(sent_groups):
            if len(g) < 2:
                continue
            if lang_groups is None:
                valid_anchors.extend((gi, si) for si in range(len(g)))
            else:
                if len(set(lang_groups[gi])) >= 2:
                    valid_anchors.extend((gi, si) for si in range(len(g)))

        if not valid_anchors:
            raise RuntimeError(
                "No valid anchors. Ensure each group has >= 2 sentences "
                "and, if using langs, at least 2 distinct languages per group."
            )

        ot_model = ot_model.to(device)
        ot_model.eval()

        for _ in range(n_trials):
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

            anchor_sentence = sent_groups[anchor_g][anchor_s]
            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]

            # Base embeddings
            base = self._batched_embed([anchor_sentence] + candidate_sentences)
            base = torch.as_tensor(base, dtype=torch.float32, device=device)

            anchor_base = base[0:1]   # [1, d]
            cand_base = base[1:]      # [K, d]

            # Transport anchor toward candidate pool
            anchor_ot = ot_model(anchor_base, cand_base)   # [1, d]

            # Usually candidates stay in base space
            if transport_candidates:
                cand_cmp = ot_model(cand_base, anchor_base.expand(cand_base.size(0), -1))
            else:
                cand_cmp = cand_base

            # Normalize for cosine similarity
            anchor_ot = F.normalize(anchor_ot, dim=-1)
            cand_cmp = F.normalize(cand_cmp, dim=-1)

            sims = (anchor_ot @ cand_cmp.T).squeeze(0).detach().cpu().numpy()  # [K]

            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

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
    def evaluate_5(
        self,
        groups,
        *,
        dvcca_model,
        langs=None,
        n_trials: int = 500,
        K: int = 10,
        recall_ks=(1, 3, 5),
        seed: int = 0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        rng = random.Random(seed)

        flat_indices, flat_sentences, flat_langs = self._index_all_sentences(
            sent_groups, lang_groups
        )

        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}

        valid_anchors = []
        for gi, g in enumerate(sent_groups):
            if len(g) < 2:
                continue
            if lang_groups is None:
                valid_anchors.extend([(gi, si) for si in range(len(g))])
            else:
                if len(set(lang_groups[gi])) >= 2:
                    valid_anchors.extend([(gi, si) for si in range(len(g))])

        if not valid_anchors:
            raise RuntimeError("No valid anchors.")

        dvcca_model.eval()

        for _ in range(n_trials):
            anchor = rng.choice(valid_anchors)
            anchor_g, anchor_s = anchor

            candidates, pos = self._sample_candidate_pool(
                sent_groups=sent_groups,
                lang_groups=lang_groups,
                anchor=anchor,
                K=K,
                hard_negatives=False,
                hard_pool_size=100,
                base_embs_flat=None,
                flat_indices=flat_indices,
                flat_langs=flat_langs,
                rng=rng,
            )

            anchor_sentence = sent_groups[anchor_g][anchor_s]
            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]

            Z = dvcca_model.encode_shared_texts(
                [anchor_sentence] + candidate_sentences,
                device=device,
                view="src",
            )

            if isinstance(Z, np.ndarray):
                Z = torch.from_numpy(Z).to(device)

            Z = F.normalize(Z, dim=-1)

            anchor_u = Z[0:1]          # [1, d]
            cand_u = Z[1:]             # [K, d]

            sims = (anchor_u @ cand_u.T).squeeze(0).detach().cpu().numpy()
            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

            rank = 1 + ranked_candidates.index(pos)
            correct1 += int(rank == 1)
            mrr_sum += 1.0 / rank

            for k in recall_ks:
                if rank <= k:
                    recall_hits[k] += 1

        acc1 = correct1 / n_trials
        mrr = mrr_sum / n_trials
        recall = {k: recall_hits[k] / n_trials for k in recall_ks}

        return EvalReport(
            accuracy_at_1=acc1,
            mrr=mrr,
            recall_at_k=recall,
            n_trials=n_trials,
            details=None,
        )
    def evaluate_6(
        self,
        groups,
        *,
        sue_model,
        embedder,
        langs=None,
        n_trials: int = 500,
        K: int = 10,
        recall_ks=(1, 3, 5),
        seed: int = 0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        rng = random.Random(seed)

        correct1 = 0
        mrr_sum = 0.0
        recall_hits = {k: 0 for k in recall_ks}

        valid_anchors = []
        for gi, g in enumerate(sent_groups):
            if len(g) < 2:
                continue
            if lang_groups is None:
                valid_anchors.extend([(gi, si) for si in range(len(g))])
            else:
                if len(set(lang_groups[gi])) >= 2:
                    valid_anchors.extend([(gi, si) for si in range(len(g))])

        if not valid_anchors:
            raise RuntimeError("No valid anchors.")

        sue_model.eval()

        for _ in range(n_trials):
            anchor = rng.choice(valid_anchors)
            anchor_g, anchor_s = anchor

            candidates, pos = self._sample_candidate_pool(
                sent_groups=sent_groups,
                lang_groups=lang_groups,
                anchor=anchor,
                K=K,
                hard_negatives=False,
                hard_pool_size=100,
                base_embs_flat=None,
                flat_indices=None,
                flat_langs=None,
                rng=rng,
            )

            anchor_sentence = sent_groups[anchor_g][anchor_s]
            candidate_sentences = [sent_groups[gi][si] for (gi, si) in candidates]

            # Step 1: base embeddings
            X = embedder([anchor_sentence] + candidate_sentences).to(device)

            # split anchor vs candidates
            anchor_x = X[0:1]
            cand_x = X[1:]

            # Step 2: SUE transform (use src side)
            anchor_z = sue_model.transform_src(anchor_x)
            cand_z = sue_model.transform_src(cand_x)

            # normalize
            anchor_z = F.normalize(anchor_z, dim=-1)
            cand_z = F.normalize(cand_z, dim=-1)

            # similarity + ranking
            sims = (anchor_z @ cand_z.T).squeeze(0).cpu().numpy()
            order = np.argsort(-sims)
            ranked_candidates = [candidates[i] for i in order]

            rank = 1 + ranked_candidates.index(pos)
            is_top1 = rank == 1

            correct1 += int(is_top1)
            mrr_sum += 1.0 / rank

            for k in recall_ks:
                if rank <= k:
                    recall_hits[k] += 1

        acc1 = correct1 / n_trials
        mrr = mrr_sum / n_trials
        recall = {k: recall_hits[k] / n_trials for k in recall_ks}

        return EvalReport(
            accuracy_at_1=acc1,
            mrr=mrr,
            recall_at_k=recall,
            n_trials=n_trials,
            details=None,
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

    if isinstance(evaluator.V, dict):
        report = evaluator.evaluate_2(groups, langs=langs, K=3, n_trials=200, seed=123, hard_negatives=False, recall_ks=(1,3,5), return_details=False)
    else: 
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
