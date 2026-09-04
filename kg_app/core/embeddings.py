import hashlib
import math
import os
import re

_MODEL = None


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", str(text or ""))
        if token.strip()
    ]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """
    Lightweight local embedding fallback.
    It is deterministic, normalized, and dependency-free, so retrieval still works
    even when a transformer model is not installed.
    """
    vector = [0.0] * dimensions
    tokens = _tokens(text)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

        # Include small bigram-style character features for fuzzy semantic overlap.
        for i in range(max(0, len(token) - 2)):
            gram = token[i : i + 3]
            gram_digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            gram_index = int.from_bytes(gram_digest[:4], "big") % dimensions
            gram_sign = 1.0 if gram_digest[4] % 2 == 0 else -1.0
            vector[gram_index] += 0.25 * gram_sign

    return _normalise(vector)


def _load_sentence_transformer():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_name = os.environ.get("KG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(model_name)
        return _MODEL
    except Exception as error:
        print(f"Embedding model unavailable, using local fallback: {error}")
        _MODEL = False
        return None


def embed_text(text: str) -> list[float]:
    backend = os.environ.get("KG_EMBEDDING_BACKEND", "auto").strip().lower()
    if backend in {"auto", "sentence-transformers", "sentence_transformers", "bge", "e5"}:
        model = _load_sentence_transformer()
        if model:
            vector = model.encode(str(text or ""), normalize_embeddings=True)
            return [float(value) for value in vector.tolist()]
    return _hash_embedding(text)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left[:size]))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right[:size]))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
