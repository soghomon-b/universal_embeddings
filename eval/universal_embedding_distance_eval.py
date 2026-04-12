from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.dvcca import CachedBitextDVCCA
from models.muse import BitextSentenceEncoder
from models.ot import SinkhornOT


# ----------------------------
# Utilities
# ----------------------------


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        return b @ a
    return a @ b.T


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Projection into universal space
# ----------------------------


def project_to_universal(
    X: np.ndarray,
    V: np.ndarray,
    mode: str = "subspace_coords",
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)

    if V is None:
        return l2_normalize(X, axis=-1)

    if isinstance(V, np.ndarray) and V.ndim == 0:
        return l2_normalize(X, axis=-1)

    V = np.asarray(V, dtype=np.float32)

    if V.ndim == 0:
        return l2_normalize(X, axis=-1)

    if mode == "linear":
        Z = X @ V
    elif mode == "subspace_coords":
        Z = X @ V
    elif mode == "subspace_recon":
        Z = (X @ V) @ V.T
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return l2_normalize(Z, axis=-1)


# ----------------------------
# Data model
# ----------------------------

Sentence = str
Lang = str
ParallelGroup = Sequence[Sentence]
ParallelGroupWithLang = Sequence[Tuple[Lang, Sentence]]


@dataclass
class DistancePairResult:
    group_id: int
    sent_i: int
    sent_j: int
    lang_i: Optional[str]
    lang_j: Optional[str]
    cosine_similarity: float
    cosine_distance: float


@dataclass
class RandomPairResult:
    flat_i: int
    flat_j: int
    group_i: int
    group_j: int
    sent_i: int
    sent_j: int
    lang_i: Optional[str]
    lang_j: Optional[str]
    cosine_similarity: float
    cosine_distance: float


@dataclass
class GroupDistanceSummary:
    group_id: int
    n_sentences: int
    n_pairs: int
    avg_cosine_similarity: float
    avg_cosine_distance: float


@dataclass
class DistanceEvalReport:
    avg_cosine_similarity: float
    avg_cosine_distance: float
    std_cosine_distance: float
    min_cosine_distance: float
    max_cosine_distance: float

    random_avg_cosine_similarity: float
    random_avg_cosine_distance: float
    random_std_cosine_distance: float
    random_min_cosine_distance: float
    random_max_cosine_distance: float

    similarity_gap: float
    distance_gap: float

    n_groups: int
    n_pairs: int
    n_random_pairs: int

    group_summaries: Optional[List[GroupDistanceSummary]] = None
    details: Optional[List[DistancePairResult]] = None
    random_details: Optional[List[RandomPairResult]] = None


# ----------------------------
# Core evaluator
# ----------------------------


class UniversalEmbeddingDistanceEvaluator:
    """
    Computes within-group semantic tightness in universal space.

    Each group contains 2+ sentences with the same meaning (often across languages).
    For each group:
      - embed all sentences
      - map them into universal space
      - compute cosine distance for all valid within-group pairs

    Also computes random cross-group pair similarity as a baseline:
      - sample pairs from different groups
      - compare same-meaning vs random-meaning similarity

    A useful universal space should show:
      within-group similarity >> random-pair similarity
    """

    def __init__(
        self,
        V: Union[np.ndarray, Dict[str, np.ndarray], BitextSentenceEncoder, SinkhornOT, CachedBitextDVCCA],
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        projection_mode: str = "subspace_coords",
        batch_size: int = 64,
        random_pair_sample_size: int = 10000,
    ):
        if isinstance(V, dict):
            self.V = {lang: np.asarray(mat, dtype=np.float32) for lang, mat in V.items()}
        elif isinstance(V, (BitextSentenceEncoder, SinkhornOT, CachedBitextDVCCA)):
            self.V = V
        else:
            self.V = np.asarray(V, dtype=np.float32)

        self.embed_fn = embed_fn
        self.projection_mode = projection_mode
        self.batch_size = batch_size
        self.random_pair_sample_size = random_pair_sample_size

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

    def _all_valid_pairs(
        self,
        sent_groups: List[List[str]],
        lang_groups: Optional[List[List[str]]],
        require_different_langs: bool,
    ) -> List[List[Tuple[int, int]]]:
        all_pairs: List[List[Tuple[int, int]]] = []
        for gi, g in enumerate(sent_groups):
            pairs = []
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if require_different_langs and lang_groups is not None:
                        if lang_groups[gi][i] == lang_groups[gi][j]:
                            continue
                    pairs.append((i, j))
            all_pairs.append(pairs)
        return all_pairs

    def _safe_cosine_stats(self, sims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sims = np.clip(sims.astype(np.float32), -1.0, 1.0)
        dists = np.maximum(0.0, 1.0 - sims)
        return sims, dists

    def _sample_random_cross_group_pairs(
        self,
        all_embeddings: List[np.ndarray],
        all_metadata: List[Tuple[int, int, Optional[str]]],  # (group_id, sent_idx, lang)
        num_samples: Optional[int] = None,
    ) -> List[RandomPairResult]:
        if num_samples is None:
            num_samples = self.random_pair_sample_size

        n = len(all_embeddings)
        if n < 2:
            return []

        results: List[RandomPairResult] = []
        max_attempts = max(10 * num_samples, 1000)
        attempts = 0

        while len(results) < num_samples and attempts < max_attempts:
            attempts += 1
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue

            group_i, sent_i, lang_i = all_metadata[i]
            group_j, sent_j, lang_j = all_metadata[j]

            if group_i == group_j:
                continue

            sim = float(np.clip(np.dot(all_embeddings[i], all_embeddings[j]), -1.0, 1.0))
            dist = float(max(0.0, 1.0 - sim))

            results.append(
                RandomPairResult(
                    flat_i=i,
                    flat_j=j,
                    group_i=group_i,
                    group_j=group_j,
                    sent_i=sent_i,
                    sent_j=sent_j,
                    lang_i=lang_i,
                    lang_j=lang_j,
                    cosine_similarity=sim,
                    cosine_distance=dist,
                )
            )

        return results

    def _finalize_report(
        self,
        pair_results: List[DistancePairResult],
        group_summaries: List[GroupDistanceSummary],
        random_pair_results: List[RandomPairResult],
        return_details: bool,
        return_group_summaries: bool,
        return_random_details: bool,
    ) -> DistanceEvalReport:
        if not pair_results:
            raise RuntimeError(
                "No valid sentence pairs found. Ensure each group has at least 2 sentences "
                "and, if requiring different languages, at least 2 different languages."
            )

        similarities = np.array([x.cosine_similarity for x in pair_results], dtype=np.float32)
        similarities, distances = self._safe_cosine_stats(similarities)

        if random_pair_results:
            random_similarities = np.array([x.cosine_similarity for x in random_pair_results], dtype=np.float32)
            random_similarities, random_distances = self._safe_cosine_stats(random_similarities)
        else:
            random_similarities = np.array([], dtype=np.float32)
            random_distances = np.array([], dtype=np.float32)

        random_avg_similarity = float(random_similarities.mean()) if len(random_similarities) else float("nan")
        random_avg_distance = float(random_distances.mean()) if len(random_distances) else float("nan")
        random_std_distance = float(random_distances.std()) if len(random_distances) else float("nan")
        random_min_distance = float(random_distances.min()) if len(random_distances) else float("nan")
        random_max_distance = float(random_distances.max()) if len(random_distances) else float("nan")

        avg_similarity = float(similarities.mean())
        avg_distance = float(distances.mean())

        return DistanceEvalReport(
            avg_cosine_similarity=avg_similarity,
            avg_cosine_distance=avg_distance,
            std_cosine_distance=float(distances.std()),
            min_cosine_distance=float(distances.min()),
            max_cosine_distance=float(distances.max()),
            random_avg_cosine_similarity=random_avg_similarity,
            random_avg_cosine_distance=random_avg_distance,
            random_std_cosine_distance=random_std_distance,
            random_min_cosine_distance=random_min_distance,
            random_max_cosine_distance=random_max_distance,
            similarity_gap=float(avg_similarity - random_avg_similarity) if len(random_similarities) else float("nan"),
            distance_gap=float(random_avg_distance - avg_distance) if len(random_distances) else float("nan"),
            n_groups=len(group_summaries),
            n_pairs=len(pair_results),
            n_random_pairs=len(random_pair_results),
            group_summaries=group_summaries if return_group_summaries else None,
            details=pair_results if return_details else None,
            random_details=random_pair_results if return_random_details else None,
        )

    def evaluate(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            base = self._batched_embed(group)
            Z = project_to_universal(base, self.V, mode=self.projection_mode)

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, None if lang_groups is None else lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=None if lang_groups is None else lang_groups[gi][i],
                        lang_j=None if lang_groups is None else lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )

    def evaluate_2(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        if lang_groups is None:
            raise ValueError("evaluate_2 requires language information.")

        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            base = self._batched_embed(group)

            if len(group) == 0:
                continue

            first_lang = lang_groups[gi][0]
            if first_lang not in self.V:
                raise KeyError(f"Missing projection for language: {first_lang}")

            sample_Z = project_to_universal(
                base[0:1],
                self.V[first_lang],
                mode=self.projection_mode,
            )
            Z = np.zeros((len(group), sample_Z.shape[1]), dtype=np.float32)

            for lang in sorted(set(lang_groups[gi])):
                if lang not in self.V:
                    raise KeyError(f"Missing projection for language: {lang}")
                idxs = [idx for idx, lg in enumerate(lang_groups[gi]) if lg == lang]
                Z_lang = project_to_universal(base[idxs], self.V[lang], mode=self.projection_mode)
                Z[idxs] = Z_lang

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=lang_groups[gi][i],
                        lang_j=lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )

    def _batched_muse_embed(self, sentences, *, encoder, device="cpu", batch_size=32):
        all_z = []
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                z = encoder.encode_batch(batch, device=device)
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
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            Z = self._batched_muse_embed(group, encoder=encoder, device=device)
            Z = l2_normalize(Z, axis=-1)

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, None if lang_groups is None else lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=None if lang_groups is None else lang_groups[gi][i],
                        lang_j=None if lang_groups is None else lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )

    def evaluate_4(
        self,
        groups,
        *,
        ot_model,
        langs=None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> DistanceEvalReport:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        ot_model = ot_model.to(device)
        ot_model.eval()

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            base_np = self._batched_embed(group)
            base = torch.as_tensor(base_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                Z = ot_model(base, base)
                Z = F.normalize(Z, dim=-1).detach().cpu().numpy()

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, None if lang_groups is None else lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=None if lang_groups is None else lang_groups[gi][i],
                        lang_j=None if lang_groups is None else lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )

    def evaluate_5(
        self,
        groups,
        *,
        dvcca_model,
        langs=None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        dvcca_model.eval()

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            Z = dvcca_model.encode_shared_texts(group, device=device, view="src")
            if isinstance(Z, np.ndarray):
                Z = torch.from_numpy(Z).to(device)
            Z = F.normalize(Z, dim=-1).detach().cpu().numpy()

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, None if lang_groups is None else lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=None if lang_groups is None else lang_groups[gi][i],
                        lang_j=None if lang_groups is None else lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )

    def evaluate_6(
        self,
        groups,
        *,
        sue_model,
        embedder,
        langs=None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
        return_random_details: bool = False,
        random_pair_sample_size: Optional[int] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        all_embeddings: List[np.ndarray] = []
        all_metadata: List[Tuple[int, int, Optional[str]]] = []

        sue_model.eval()

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            X = embedder(group)
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            X = X.to(device)

            with torch.no_grad():
                Z = sue_model.transform_src(X)
                Z = F.normalize(Z, dim=-1).detach().cpu().numpy()

            for idx in range(len(group)):
                all_embeddings.append(Z[idx])
                all_metadata.append((gi, idx, None if lang_groups is None else lang_groups[gi][idx]))

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.clip(np.dot(Z[i], Z[j]), -1.0, 1.0))
                dist = float(max(0.0, 1.0 - sim))
                group_sims.append(sim)
                group_distances.append(dist)
                pair_results.append(
                    DistancePairResult(
                        group_id=gi,
                        sent_i=i,
                        sent_j=j,
                        lang_i=None if lang_groups is None else lang_groups[gi][i],
                        lang_j=None if lang_groups is None else lang_groups[gi][j],
                        cosine_similarity=sim,
                        cosine_distance=dist,
                    )
                )

            group_summaries.append(
                GroupDistanceSummary(
                    group_id=gi,
                    n_sentences=len(group),
                    n_pairs=len(pairs),
                    avg_cosine_similarity=float(np.mean(group_sims)),
                    avg_cosine_distance=float(np.mean(group_distances)),
                )
            )

        random_pair_results = self._sample_random_cross_group_pairs(
            all_embeddings,
            all_metadata,
            num_samples=random_pair_sample_size,
        )

        return self._finalize_report(
            pair_results,
            group_summaries,
            random_pair_results,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
            return_random_details=return_random_details,
        )


# ----------------------------
# Example usage
# ----------------------------


def dummy_embed_fn(sentences: List[str]) -> np.ndarray:
    d = 768
    out = np.zeros((len(sentences), d), dtype=np.float32)
    for i, s in enumerate(sentences):
        rs = np.random.RandomState(abs(hash(s)) % (2**32))
        out[i] = rs.normal(size=d).astype(np.float32)
    return l2_normalize(out, axis=-1)


if __name__ == "__main__":
    set_seed(0)

    d = 768
    k = 256
    V = np.random.randn(d, k).astype(np.float32)

    groups = [
        ["Hello world (en)", "Bonjour le monde (fr)", "Hola mundo (es)"],
        ["How are you? (en)", "Comment ça va? (fr)", "¿Cómo estás? (es)"],
        ["I like coffee (en)", "J'aime le café (fr)", "Me gusta el café (es)"],
        ["The weather is nice today (en)", "Il fait beau aujourd'hui (fr)", "Hace buen tiempo hoy (es)"],
    ]

    langs = [
        ["en", "fr", "es"],
        ["en", "fr", "es"],
        ["en", "fr", "es"],
        ["en", "fr", "es"],
    ]

    evaluator = UniversalEmbeddingDistanceEvaluator(
        V=V,
        embed_fn=dummy_embed_fn,
        projection_mode="subspace_coords",
        batch_size=64,
        random_pair_sample_size=5000,
    )

    if isinstance(evaluator.V, dict):
        report = evaluator.evaluate_2(
            groups,
            langs=langs,
            require_different_langs=True,
            return_group_summaries=True,
            return_details=False,
            return_random_details=False,
        )
    else:
        report = evaluator.evaluate(
            groups,
            langs=langs,
            require_different_langs=True,
            return_group_summaries=True,
            return_details=False,
            return_random_details=False,
        )

    print("[same-meaning pairs]")
    print("Avg cosine similarity:", report.avg_cosine_similarity)
    print("Avg cosine distance:", report.avg_cosine_distance)
    print("Std cosine distance:", report.std_cosine_distance)
    print("Min cosine distance:", report.min_cosine_distance)
    print("Max cosine distance:", report.max_cosine_distance)
    print("Pairs:", report.n_pairs)

    print()
    print("[random cross-group pairs]")
    print("Avg cosine similarity:", report.random_avg_cosine_similarity)
    print("Avg cosine distance:", report.random_avg_cosine_distance)
    print("Std cosine distance:", report.random_std_cosine_distance)
    print("Min cosine distance:", report.random_min_cosine_distance)
    print("Max cosine distance:", report.random_max_cosine_distance)
    print("Random pairs:", report.n_random_pairs)

    print()
    print("[separation]")
    print("Similarity gap (same - random):", report.similarity_gap)
    print("Distance gap (random - same):", report.distance_gap)