from __future__ import annotations

import hashlib
from typing import List


class DeterministicEmbedding:
    """Deterministic hash-based embeddings for local development."""

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest]