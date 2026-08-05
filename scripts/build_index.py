from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ARTIFACT_ROOT,
    PROCESSED_ROOT,
    PROJECT_ROOT,
    configure_environment,
    ensure_project_dirs,
    now_utc,
    read_jsonl,
    retrieval_entity_tokens,
    write_json,
    write_jsonl,
)


CURATED_GLOSSARY = [
    ("active ingredient", "Wirkstoff"),
    ("side effects", "Nebenwirkungen"),
    ("dosage", "Dosierung"),
    ("dose", "Dosis"),
    ("contraindications", "Gegenanzeigen"),
    ("warning", "Warnung"),
    ("prescription", "Verschreibung"),
    ("tablet", "Tablette"),
    ("tablets", "Tabletten"),
    ("capsule", "Kapsel"),
    ("capsules", "Kapseln"),
    ("solution", "Lösung"),
    ("take", "einnehmen"),
    ("once daily", "einmal täglich"),
    ("twice daily", "zweimal täglich"),
    ("after meals", "nach den Mahlzeiten"),
    ("before meals", "vor den Mahlzeiten"),
    ("adults", "Erwachsene"),
    ("children", "Kinder"),
    ("do not take", "nicht einnehmen"),
    ("keep out of reach of children", "für Kinder unzugänglich aufbewahren"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=50000)
    parser.add_argument("--include-emea", action="store_true")
    parser.add_argument("--input-train-file", default="data/processed/ahazeemi_train.jsonl")
    parser.add_argument("--output-dir", default="artifacts/rag")
    return parser.parse_args()


def encode_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _cuda_available() else "cpu"
    encoder = SentenceTransformer(
        model_name,
        cache_folder=os.environ["TRANSFORMERS_CACHE"],
        device=device,
    )
    vectors = encoder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_translation_memory(rows: list[dict[str, Any]], output_dir: Path, model_name: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [row["source"] for row in rows]
    vectors = encode_texts(texts, model_name)
    np.save(output_dir / "translation_memory_embeddings.npy", vectors)
    write_jsonl(
        output_dir / "translation_memory_metadata.jsonl",
        [
            {
                "source": row["source"],
                "target": row["target"],
                "dataset": row.get("dataset", "unknown"),
                "split": row.get("split", "train"),
                "row_id": row.get("row_id"),
                "source_hash": row.get("source_hash"),
                "provenance": row.get("dataset", "unknown"),
                "entity_tokens": retrieval_entity_tokens(row["source"]),
            }
            for row in rows
        ],
    )
    return {
        "rows": len(rows),
        "dimensions": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "embedding_model": model_name,
        "index": str(output_dir / "translation_memory_embeddings.npy"),
    }


def build_emea_index(rows: list[dict[str, Any]], output_dir: Path, model_name: str) -> dict[str, Any]:
    if not rows:
        return {"status": "not_available", "rows": 0}
    emea_dir = output_dir / "emea_reference"
    emea_dir.mkdir(parents=True, exist_ok=True)
    texts = [row["source"] for row in rows]
    vectors = encode_texts(texts, model_name)
    np.save(emea_dir / "embeddings.npy", vectors)
    write_jsonl(emea_dir / "metadata.jsonl", rows)
    return {
        "status": "saved",
        "rows": len(rows),
        "dimensions": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "index": str(emea_dir / "embeddings.npy"),
    }


def main() -> None:
    args = parse_args()
    configure_environment()
    ensure_project_dirs()

    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    input_train = Path(args.input_train_file)
    if not input_train.is_absolute():
        input_train = PROJECT_ROOT / input_train
    input_train = input_train.resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if PROJECT_ROOT not in input_train.parents or PROJECT_ROOT not in output_dir.parents:
        raise ValueError("Index input and output must remain inside the project")
    rows = read_jsonl(input_train)[: args.max_examples]
    if not rows:
        raise RuntimeError("Prepared training rows are missing.")
    rag_dir = output_dir
    rag_dir.mkdir(parents=True, exist_ok=True)

    tm_info = build_translation_memory(rows, rag_dir, embedding_model)
    glossary_path = rag_dir / "glossary.jsonl"
    glossary_rows = [
        {
            "source_term": source,
            "target_term": target,
            "example": f"{source} -> {target}",
            "source_dataset": "curated_medical_terminology",
            "confidence": "human-curated-for-demo",
        }
        for source, target in CURATED_GLOSSARY
    ]
    write_jsonl(glossary_path, glossary_rows)

    emea_info = {"status": "not_requested"}
    if args.include_emea:
        emea_info = build_emea_index(
            read_jsonl(PROCESSED_ROOT / "emea_external.jsonl"),
            rag_dir,
            embedding_model,
        )

    manifest = {
        "created_at_utc": now_utc(),
        "translation_memory": tm_info,
        "glossary": {
            "rows": len(glossary_rows),
            "path": str(glossary_path),
            "source_type": "curated_medical_terminology",
        },
        "emea_reference": emea_info,
        "evaluation_warning": (
            "The EMEA reference index must not be used when reporting EMEA metrics "
            "on the same rows; it is for UI/reference demonstration only."
        ),
    }
    write_json(rag_dir / "index_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
