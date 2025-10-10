from __future__ import annotations

from typing import Dict, Iterable, List, Protocol

from src.adapters.pinecone_client import PineconeIndexProtocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:  # pragma: no cover - interface
        ...


class ResponseGenerator(Protocol):
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


class RagService:
    """Handles multi-tenant retrieval over the vector store."""

    def __init__(
        self,
        pinecone_index: PineconeIndexProtocol,
        embedder: EmbeddingProvider,
        response_generator: ResponseGenerator,
    ) -> None:
        self._index = pinecone_index
        self._embedder = embedder
        self._response_generator = response_generator

    def answer_query(self, context: Dict[str, str], query: str, history: List[Dict[str, str]]) -> str:
        org_id = context.get("org_id", "default_org")
        branch_id = context.get("branch_id", "default_branch")
        filter_payload = {
            "$and": [
                {"org_id": org_id},
                {"branch_id": branch_id},
            ]
        }
        vector = self._embedder.embed(query)
        result = self._index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            filter=filter_payload,
            namespace=f"{org_id}::{branch_id}",
        )
        matches = result.get("matches", [])
        context_snippets = self._extract_context(matches)
        if not context_snippets:
            return "I could not find information for that request."
        prompt = self._build_prompt(query=query, context_snippets=context_snippets, history=history)
        return self._response_generator.generate(prompt).strip()

    def _extract_context(self, matches: Iterable[Dict]) -> List[str]:
        snippets: List[str] = []
        for match in matches:
            metadata = match.get("metadata")
            if not isinstance(metadata, dict):
                continue
            text = metadata.get("text")
            if text:
                snippets.append(str(text))
        return snippets

    def _build_prompt(self, *, query: str, context_snippets: List[str], history: List[Dict[str, str]]) -> str:
        context_block = "\n\n".join(context_snippets)
        if history:
            history_lines = []
            for message in history[-8:]:
                role = message.get("role", "user")
                content = message.get("content", "")
                if content:
                    history_lines.append(f"{role}: {content}")
            history_block = "\n".join(history_lines) if history_lines else "None."
        else:
            history_block = "None."

        prompt = (
            "You are a helpful sales assistant crafting concise, accurate answers.\n"
            "Use ONLY the provided context snippets to answer the user's question.\n"
            "If the context does not contain the answer, say you do not have that information.\n\n"
            f"Context Snippets:\n{context_block}\n\n"
            f"Conversation History:\n{history_block}\n\n"
            f"User Question:\n{query}\n\n"
            "Answer:"
        )
        return prompt
