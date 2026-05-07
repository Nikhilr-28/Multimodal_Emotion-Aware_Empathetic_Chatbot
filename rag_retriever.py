import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_emotion(emotion: str) -> str:
    return (emotion or "").strip().lower()


def format_query_text(emotion: str, text: str) -> str:
    return f"Emotion: {normalize_emotion(emotion)}\nContext: {(text or '').strip()}"


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_train_examples(train_examples_path: str) -> List[dict]:
    with open(train_examples_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_record(example: dict) -> dict:
    input_to_model = example.get("input_to_model", {})
    emotion = normalize_emotion(input_to_model.get("emotion", ""))
    context = (input_to_model.get("text", "") or "").strip()
    target = (example.get("target", "") or "").strip()
    return {
        "example_id": example.get("example_id"),
        "conv_id": example.get("conv_id", ""),
        "utterance_idx": example.get("utterance_idx", -1),
        "emotion": emotion,
        "context": context,
        "target": target,
        "retrieval_text": format_query_text(emotion, context),
    }


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_metadata_by_emotion(index_dir: str, emotion: str, metadata: List[dict]) -> None:
    emotion_dir = os.path.join(index_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    metadata_path = os.path.join(emotion_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata_by_emotion(index_dir: str, emotion: str) -> List[dict]:
    metadata_path = os.path.join(index_dir, emotion, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class EmotionIndex:
    emotion: str
    index: faiss.Index
    metadata: List[dict]


class EmotionFaissRetriever:
    def __init__(
        self,
        train_examples_path: str,
        index_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        force_rebuild: bool = False,
        batch_size: int = 64,
    ) -> None:
        self.train_examples_path = train_examples_path
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.encoder = SentenceTransformer(embedding_model)
        self.indexes_by_emotion: Dict[str, EmotionIndex] = {}
        self.build_or_load_indexes_by_emotion(force_rebuild=force_rebuild)

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return _l2_normalize(vectors)

    def _emotion_paths(self, emotion: str) -> tuple[str, str]:
        emotion_dir = os.path.join(self.index_dir, emotion)
        return (
            os.path.join(emotion_dir, "index.faiss"),
            os.path.join(emotion_dir, "metadata.json"),
        )

    def _build_from_examples(self, examples: List[dict]) -> None:
        grouped_records: Dict[str, List[dict]] = {}
        for example in examples:
            record = _build_record(example)
            if not record["emotion"] or not record["context"] or not record["target"]:
                continue
            grouped_records.setdefault(record["emotion"], []).append(record)

        os.makedirs(self.index_dir, exist_ok=True)
        model_info_path = os.path.join(self.index_dir, "model_info.json")
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_model": self.embedding_model_name}, f, indent=2)

        for emotion, records in grouped_records.items():
            texts = [r["retrieval_text"] for r in records]
            vectors = self._encode(texts)
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

            emotion_dir = os.path.join(self.index_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            index_path = os.path.join(emotion_dir, "index.faiss")
            faiss.write_index(index, index_path)
            save_metadata_by_emotion(index_dir=self.index_dir, emotion=emotion, metadata=records)

            self.indexes_by_emotion[emotion] = EmotionIndex(
                emotion=emotion,
                index=index,
                metadata=records,
            )

    def build_or_load_indexes_by_emotion(self, force_rebuild: bool = False) -> None:
        if force_rebuild or not os.path.isdir(self.index_dir):
            examples = load_train_examples(self.train_examples_path)
            self._build_from_examples(examples)
            return

        loaded_any = False
        for emotion in sorted(os.listdir(self.index_dir)):
            emotion_dir = os.path.join(self.index_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            index_path, metadata_path = self._emotion_paths(emotion)
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                continue
            index = faiss.read_index(index_path)
            metadata = load_metadata_by_emotion(self.index_dir, emotion)
            self.indexes_by_emotion[emotion] = EmotionIndex(
                emotion=emotion, index=index, metadata=metadata
            )
            loaded_any = True

        if not loaded_any:
            examples = load_train_examples(self.train_examples_path)
            self._build_from_examples(examples)

    def retrieve_examples(
        self,
        query: str,
        emotion: str,
        k: int = 3,
        max_example_chars: Optional[int] = None,
    ) -> List[dict]:
        normalized_emotion = normalize_emotion(emotion)
        if normalized_emotion not in self.indexes_by_emotion:
            return []

        emotion_index = self.indexes_by_emotion[normalized_emotion]
        if emotion_index.index.ntotal == 0:
            return []

        query_text = format_query_text(normalized_emotion, query)
        query_vec = self._encode([query_text])
        top_k = min(max(k, 1), emotion_index.index.ntotal)
        scores, idxs = emotion_index.index.search(query_vec, top_k)

        results: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            record = dict(emotion_index.metadata[idx])
            if max_example_chars and max_example_chars > 0:
                record["context"] = truncate_text(record["context"], max_example_chars)
                record["target"] = truncate_text(record["target"], max_example_chars)
            record["score"] = float(score)
            results.append(record)
        return results

