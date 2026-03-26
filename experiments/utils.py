import torch
import numpy as np
# -----------------------------
# Utilities
# -----------------------------
def remove_nones_parallel(parallel_lists):
    return [
        [s for s in group if s is not None]
        for group in parallel_lists
        if any(s is not None for s in group)
    ]

def clean_parallel_lang_sentence(parallel_lists):
    return [
        [(lang, s) for (lang, s) in group if s is not None and s != ""]
        for group in parallel_lists
        if any(s is not None and s != "" for (_, s) in group)
    ]


def torch_embedder_to_numpy(embedder):
    def _embed_fn(sentences):
        cleaned = []
        for i, s in enumerate(sentences):
            if s is None:
                raise ValueError(
                    f"Got None sentence at index {i} in batch of size {len(sentences)}"
                )
            if not isinstance(s, str):
                s = str(s)
            s = s.strip()
            if not s:
                raise ValueError(f"Got empty sentence at index {i}")
            cleaned.append(s)

        with torch.no_grad():
            x = embedder(cleaned)
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.asarray(x, dtype=np.float32)

    return _embed_fn
