from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    n_groups: int
    n_pairs: int
    group_summaries: Optional[List[GroupDistanceSummary]] = None
    details: Optional[List[DistancePairResult]] = None


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
      - compute cosine distance for all valid pairs
    Returns a report similar in spirit to EvalReport from retrieval.
    """

    def __init__(
        self,
        V: np.ndarray,
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        projection_mode: str = "subspace_coords",
        batch_size: int = 64,
    ):
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

    def _finalize_report(
        self,
        pair_results: List[DistancePairResult],
        group_summaries: List[GroupDistanceSummary],
        return_details: bool,
        return_group_summaries: bool,
    ) -> DistanceEvalReport:
        if not pair_results:
            raise RuntimeError(
                "No valid sentence pairs found. Ensure each group has at least 2 sentences "
                "and, if requiring different languages, at least 2 different languages."
            )

        distances = np.array([x.cosine_distance for x in pair_results], dtype=np.float32)
        similarities = np.array([x.cosine_similarity for x in pair_results], dtype=np.float32)

        return DistanceEvalReport(
            avg_cosine_similarity=float(similarities.mean()),
            avg_cosine_distance=float(distances.mean()),
            std_cosine_distance=float(distances.std()),
            min_cosine_distance=float(distances.min()),
            max_cosine_distance=float(distances.max()),
            n_groups=len(group_summaries),
            n_pairs=len(pair_results),
            group_summaries=group_summaries if return_group_summaries else None,
            details=pair_results if return_details else None,
        )

    def evaluate(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            base = self._batched_embed(group)
            Z = project_to_universal(base, self.V, mode=self.projection_mode)

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
        )

    def evaluate_2(
        self,
        groups: Sequence[ParallelGroup] | Sequence[ParallelGroupWithLang],
        *,
        langs: Optional[Sequence[Sequence[str]]] = None,
        require_different_langs: bool = True,
        return_details: bool = False,
        return_group_summaries: bool = True,
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        if lang_groups is None:
            raise ValueError("evaluate_2 requires language information.")

        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

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

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            Z = self._batched_muse_embed(group, encoder=encoder, device=device)
            Z = l2_normalize(Z, axis=-1)

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
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
        device: Optional[str] = None,
    ) -> DistanceEvalReport:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

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

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

        dvcca_model.eval()

        for gi, group in enumerate(sent_groups):
            pairs = all_pairs[gi]
            if not pairs:
                continue

            Z = dvcca_model.encode_shared_texts(group, device=device, view="src")
            if isinstance(Z, np.ndarray):
                Z = torch.from_numpy(Z).to(device)
            Z = F.normalize(Z, dim=-1).detach().cpu().numpy()

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> DistanceEvalReport:
        sent_groups, lang_groups = self._prepare_groups(groups, langs)
        all_pairs = self._all_valid_pairs(sent_groups, lang_groups, require_different_langs)

        pair_results: List[DistancePairResult] = []
        group_summaries: List[GroupDistanceSummary] = []

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

            group_distances = []
            group_sims = []
            for i, j in pairs:
                sim = float(np.dot(Z[i], Z[j]))
                dist = float(1.0 - sim)
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

        return self._finalize_report(
            pair_results,
            group_summaries,
            return_details=return_details,
            return_group_summaries=return_group_summaries,
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
    ]

    langs = [
        ["en", "fr", "es"],
        ["en", "fr", "es"],
        ["en", "fr", "es"],
    ]

    evaluator = UniversalEmbeddingDistanceEvaluator(
        V=V,
        embed_fn=dummy_embed_fn,
        projection_mode="subspace_coords",
        batch_size=64,
    )

    if isinstance(evaluator.V, dict):
        report = evaluator.evaluate_2(
            groups,
            langs=langs,
            require_different_langs=True,
            return_group_summaries=True,
            return_details=False,
        )
    else:
        report = evaluator.evaluate(
            groups,
            langs=langs,
            require_different_langs=True,
            return_group_summaries=True,
            return_details=False,
        )

    print("Avg cosine similarity:", report.avg_cosine_similarity)
    print("Avg cosine distance:", report.avg_cosine_distance)
    print("Std cosine distance:", report.std_cosine_distance)
    print("Pairs:", report.n_pairs)
