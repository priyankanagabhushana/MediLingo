from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ARTIFACT_ROOT,
    PROJECT_ROOT,
    configure_environment,
    read_jsonl,
    retrieval_entity_tokens,
)
from verifier import verify_translation


class RetrievalIndex:
    def __init__(self, index_dir: Path | None = None):
        configure_environment()
        self.index_dir = index_dir or ARTIFACT_ROOT / "rag"
        self.tm_vectors: np.ndarray | None = None
        self.tm_metadata: list[dict[str, Any]] = []
        self.glossary: list[dict[str, Any]] = []
        self.encoder = None
        self.embedding_model_name = (
            os.environ.get(
                "MEDILINGO_EMBEDDING_MODEL_DIR",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip()
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._load_metadata()

    def _load_metadata(self) -> None:
        vector_path = self.index_dir / "translation_memory_embeddings.npy"
        metadata_path = self.index_dir / "translation_memory_metadata.jsonl"
        if vector_path.exists() and metadata_path.exists():
            self.tm_vectors = np.load(vector_path, mmap_mode="r")
            self.tm_metadata = read_jsonl(metadata_path)
        self.glossary = read_jsonl(self.index_dir / "glossary.jsonl")

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if self._cuda_available() else "cpu"
            self.encoder = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=os.environ["TRANSFORMERS_CACHE"],
                device=device,
            )
        return self.encoder

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _entity_keys(values: list[str]) -> set[str]:
        return {
            re.sub(r"[^a-z0-9]+", "", value.lower())
            for value in values
            if re.sub(r"[^a-z0-9]+", "", value.lower())
        }

    @staticmethod
    def _product_keys(values: list[str]) -> set[str]:
        # Ignore abbreviations such as VTE, DVT, and PE when deciding whether
        # two sentences mention conflicting products. Arixtra and Quixidar,
        # however, remain distinct product keys.
        products = set()
        for value in values:
            bare = value.rstrip("s")
            if bare.isupper() or any(char.isdigit() for char in bare):
                continue
            key = re.sub(r"[^a-z0-9]+", "", value.lower())
            if len(key) >= 4:
                products.add(key)
        return products

    def _row_entity_values(self, row: dict[str, Any]) -> list[str]:
        return row.get("entity_tokens") or retrieval_entity_tokens(row.get("source", ""))

    def _row_entity_keys(self, row: dict[str, Any]) -> set[str]:
        return self._entity_keys(self._row_entity_values(row))

    def search_translation_memory(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.65,
        entity_filter: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.tm_vectors is None or not self.tm_metadata:
            return [], {
                "retrieval_gate": "no_index",
                "similarity_threshold": similarity_threshold,
                "entity_filter": entity_filter,
                "entity_conflicts_filtered": 0,
                "best_raw_similarity": None,
                "best_kept_similarity": None,
            }
        query_vector = self._get_encoder().encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        scores = np.asarray(self.tm_vectors @ query_vector).reshape(-1)
        candidate_limit = min(len(scores), max(max(1, top_k) * 8, 64))
        top_indices = np.argsort(-scores)[:candidate_limit]
        query_entity_values = retrieval_entity_tokens(query)
        query_entities = self._entity_keys(query_entity_values)
        query_products = self._product_keys(query_entity_values)
        results: list[dict[str, Any]] = []
        conflicts = 0
        below_threshold = 0
        for index in top_indices:
            row = dict(self.tm_metadata[int(index)])
            score = float(scores[int(index)])
            row_values = self._row_entity_values(row)
            row_entities = self._entity_keys(row_values)
            row_products = self._product_keys(row_values)
            entity_match = bool(query_entities and row_entities and query_entities.intersection(row_entities))
            product_conflict = bool(
                query_products and row_products and not query_products.intersection(row_products)
            )
            generic_entity_conflict = bool(
                not query_products and query_entities and row_entities and not entity_match
            )
            if entity_filter and (product_conflict or generic_entity_conflict):
                conflicts += 1
                continue
            if score < similarity_threshold:
                below_threshold += 1
                continue
            row["similarity"] = score
            row["entity_match"] = entity_match if query_entities else None
            results.append(row)
            if len(results) >= max(1, top_k):
                break
        best_raw_similarity = float(scores[int(top_indices[0])]) if len(top_indices) else None
        best_kept_similarity = results[0]["similarity"] if results else None
        diagnostics = {
            "retrieval_gate": "evidence_kept" if results else "no_sufficient_non_conflicting_evidence",
            "similarity_threshold": similarity_threshold,
            "entity_filter": entity_filter,
            "query_entities": sorted(query_entities),
            "entity_conflicts_filtered": conflicts,
            "below_threshold_candidates": below_threshold,
            "best_raw_similarity": best_raw_similarity,
            "best_kept_similarity": best_kept_similarity,
        }
        return results, diagnostics

    def search_glossary(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        lowered = query.lower()
        direct = [
            row for row in self.glossary
            if row.get("source_term", "").lower() in lowered
        ]
        return direct[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        glossary_top_k: int = 8,
        similarity_threshold: float = 0.65,
        entity_filter: bool = True,
    ) -> dict[str, Any]:
        translation_memory, diagnostics = self.search_translation_memory(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            entity_filter=entity_filter,
        )
        glossary = self.search_glossary(query, top_k=glossary_top_k)
        diagnostics["glossary_hits"] = len(glossary)
        return {
            "translation_memory": translation_memory,
            "glossary": glossary,
            "diagnostics": diagnostics,
        }


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    for prefix in ["German translation:", "Übersetzung:", "Translation:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # A model can occasionally continue by copying a retrieved bilingual example.
    # Cut only explicit reference labels; preserve ordinary German lists and paragraphs.
    for marker in ["\n- English:", "\nEnglish:", "\nRetrieved translation-memory", "\nRetrieved terminology:"]:
        marker_index = text.find(marker)
        if marker_index > 0:
            text = text[:marker_index].rstrip()
    return text.strip()


class TranslationRuntime:
    def __init__(
        self,
        model_id: str,
        adapter_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
    ):
        configure_environment()
        self.model_id = model_id
        self.adapter_dir = self._resolve_adapter(adapter_dir)
        self.index = RetrievalIndex(Path(index_dir) if index_dir else None)
        self.model = None
        self.tokenizer = None
        self.device = None

    @staticmethod
    def _resolve_adapter(adapter_dir: str | Path | None) -> Path | None:
        if not adapter_dir:
            return None
        path = Path(adapter_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.exists() else None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "cache_dir": os.environ["TRANSFORMERS_CACHE"],
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        if self.adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_dir),
                is_trainable=False,
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _chat_prompt(self, messages: list[dict[str, str]], generation: bool) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": generation}
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _build_messages(
        self,
        source: str,
        retrieved: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        system = (
            "You are a precise English-to-German medical-information translator "
            "for administrative use. Translate only. Do not diagnose, prescribe, "
            "explain, or reveal reasoning. Preserve medicine names, numbers, units, "
            "dosage, warnings, negation, and formatting. Retrieved material is "
            "reference-only: never reproduce it, list it, or add a second translation. "
            "Return exactly one German translation of the marked source text. "
            "Do not output English, headings, bullets, notes, citations, or explanations."
        )
        if retrieved:
            examples = "\n".join(
                f"- English: {item.get('source', '')[:350]}\n  German: {item.get('target', '')[:350]}"
                for item in retrieved.get("translation_memory", [])
            )
            glossary = "\n".join(
                f"- {item.get('source_term', '')} -> {item.get('target_term', '')}"
                for item in retrieved.get("glossary", [])
            )
            context = (
                "\n\n<reference_memory>\n"
                + (examples or "- none")
                + "\n</reference_memory>\n"
                + "<reference_terminology>\n"
                + (glossary or "- none")
                + "\n</reference_terminology>"
            )
        else:
            context = ""
        user = (
            "Use the reference material only to choose terminology and style; "
            "it is not part of the text to translate. Translate exactly the text "
            "between <source_text> and </source_text> into German. Your entire "
            "response must contain only that one German translation; do not quote, "
            "copy, or continue the references."
            + context
            + "\n\n<source_text>\n"
            + source
            + "\n</source_text>\nNow provide only the German translation."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def translate(
        self,
        source: str,
        use_rag: bool = True,
        top_k: int = 4,
        glossary_top_k: int = 8,
        similarity_threshold: float = 0.65,
        entity_filter: bool = True,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        self.load()
        if use_rag:
            retrieved = self.index.retrieve(
                source,
                top_k=top_k,
                glossary_top_k=glossary_top_k,
                similarity_threshold=similarity_threshold,
                entity_filter=entity_filter,
            )
        else:
            retrieved = {
                "translation_memory": [],
                "glossary": [],
                "diagnostics": {
                    "retrieval_gate": "disabled",
                    "similarity_threshold": similarity_threshold,
                    "entity_filter": entity_filter,
                    "entity_conflicts_filtered": 0,
                    "best_raw_similarity": None,
                    "best_kept_similarity": None,
                    "glossary_hits": 0,
                },
            }
        has_evidence = bool(retrieved["translation_memory"] or retrieved["glossary"])
        messages = self._build_messages(source, retrieved if use_rag and has_evidence else None)
        prompt = self._chat_prompt(messages, generation=True)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        import torch

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        translation = clean_generation(self.tokenizer.decode(generated, skip_special_tokens=True))
        verification = verify_translation(source, translation)
        return {
            "source": source,
            "translation": translation,
            "model_id": self.model_id,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "rag_enabled": use_rag,
            "rag_used": bool(use_rag and has_evidence),
            "retrieved": retrieved,
            "verification": verification,
        }

    def close(self) -> None:
        try:
            import torch

            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
