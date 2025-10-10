from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping

from src.orchestrator.intents import Intent

TokenVector = Dict[str, float]

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


DEFAULT_TRAINING_DATA: Mapping[Intent, List[str]] = {
    Intent.BOOKING: [
        "book an appointment",
        "schedule a consultation for me",
        "set up a meeting tomorrow",
        "reserve a time slot",
        "I want to get on the calendar",
        "book me for next week",
        "arrange a visit with your team",
    ],
    Intent.CANCEL_BOOKING: [
        "cancel my appointment",
        "I need to reschedule our call",
        "please move my booking",
        "change the meeting time",
        "call off the appointment",
        "rebook the consultation for later",
        "push back our meeting",
    ],
    Intent.PURCHASE_INTEREST: [
        "I'm interested in buying your product",
        "what does it cost",
        "tell me about pricing",
        "I'd like to purchase your services",
        "how much is the plan",
        "looking into your packages",
        "give me a quote for the installation",
    ],
    Intent.RAG_INFO: [
        "tell me more about your services",
        "what does your company do",
        "give me general information",
        "how does the process work",
        "share details about your offerings",
        "provide information on the business",
        "what support do you provide",
    ],
}


class SemanticIntentClassifier:
    """Lightweight tf-idf intent classifier without brittle keyword rules."""

    def __init__(
        self,
        training_data: Mapping[Intent, Iterable[str]] | None = None,
        similarity_threshold: float = 0.12,
    ) -> None:
        self._training_data = self._normalize_training_data(training_data)
        self._idf = self._build_idf(self._training_data)
        self._intent_profiles = self._build_intent_profiles(self._training_data)
        self._threshold = similarity_threshold

    def classify(self, state) -> Intent:
        query = getattr(state, "user_query", "") or ""
        vector = self._vectorize(query)
        if not vector:
            return Intent.RAG_INFO

        best_intent = Intent.RAG_INFO
        best_score = 0.0
        for intent, profile in self._intent_profiles.items():
            if not profile:
                continue
            score = self._cosine_similarity(vector, profile)
            if score > best_score:
                best_intent = intent
                best_score = score

        if best_score < self._threshold:
            return Intent.RAG_INFO
        return best_intent

    def _normalize_training_data(
        self, training_data: Mapping[Intent, Iterable[str]] | None
    ) -> Dict[Intent, List[str]]:
        if training_data is None:
            return {intent: list(samples) for intent, samples in DEFAULT_TRAINING_DATA.items()}
        normalised: Dict[Intent, List[str]] = {}
        for intent in Intent:
            examples = list(training_data.get(intent, []))
            if examples:
                normalised[intent] = examples
        if not normalised:
            raise ValueError("Training data must provide at least one example")
        return normalised

    def _build_idf(self, training_data: Mapping[Intent, Iterable[str]]) -> Dict[str, float]:
        doc_counts: Counter[str] = Counter()
        total_docs = 0

        for phrases in training_data.values():
            for phrase in phrases:
                tokens = set(self._tokenize(phrase))
                if not tokens:
                    continue
                doc_counts.update(tokens)
                total_docs += 1

        if total_docs == 0:
            return {}

        idf: Dict[str, float] = {}
        for token, count in doc_counts.items():
            idf[token] = math.log((1 + total_docs) / (1 + count)) + 1.0
        return idf

    def _build_intent_profiles(
        self, training_data: Mapping[Intent, Iterable[str]]
    ) -> Dict[Intent, TokenVector]:
        profiles: Dict[Intent, TokenVector] = {}
        for intent, phrases in training_data.items():
            summed: defaultdict[str, float] = defaultdict(float)
            example_count = 0
            for phrase in phrases:
                vec = self._vectorize(phrase)
                if not vec:
                    continue
                example_count += 1
                for term, weight in vec.items():
                    summed[term] += weight
            if example_count == 0:
                profiles[intent] = {}
                continue
            profiles[intent] = {term: weight / example_count for term, weight in summed.items()}
        return profiles

    def _vectorize(self, text: str) -> TokenVector:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf = Counter(tokens)
        vector: TokenVector = {}
        length = float(len(tokens))
        for token, term_count in tf.items():
            idf = self._idf.get(token)
            if idf is None:
                continue
            vector[token] = (term_count / length) * idf
        return vector

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _cosine_similarity(a: TokenVector, b: TokenVector) -> float:
        dot = sum(a_val * b.get(term, 0.0) for term, a_val in a.items())
        if dot == 0.0:
            return 0.0
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
