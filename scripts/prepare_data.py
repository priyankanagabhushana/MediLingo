from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_ROOT,
    PROCESSED_ROOT,
    RAW_ROOT,
    configure_environment,
    ensure_project_dirs,
    dosage_tokens,
    medicine_name_tokens,
    negation_markers,
    warning_markers,
    normalize_text,
    number_tokens,
    read_jsonl,
    source_hash,
    unit_tokens,
    write_json,
    write_jsonl,
)


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen_exact: set[tuple[str, str]] = set()
    seen_source: collections.Counter[str] = collections.Counter()
    result: list[dict[str, Any]] = []
    duplicates = 0

    for row in rows:
        source = normalize_text(row.get("source"))
        target = normalize_text(row.get("target"))
        if not source or not target:
            continue
        key = (source.lower(), target.lower())
        seen_source[source.lower()] += 1
        if key in seen_exact:
            duplicates += 1
            continue
        seen_exact.add(key)
        row = dict(row)
        row["source"] = source
        row["target"] = target
        row["source_hash"] = source_hash(source)
        result.append(row)

    repeated_sources = sum(1 for count in seen_source.values() if count > 1)
    return result, {
        "input_rows": len(rows),
        "output_rows": len(result),
        "exact_duplicate_rows_removed": duplicates,
        "repeated_source_strings": repeated_sources,
    }


def near_duplicate_sample(rows: list[dict[str, Any]], sample_size: int = 500) -> dict[str, Any]:
    """Bounded near-duplicate audit using prefix/length buckets."""
    sample = rows[:sample_size]
    buckets: dict[tuple[str, int], list[tuple[int, str]]] = collections.defaultdict(list)
    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(sample):
        normalized = re.sub(r"[^a-z0-9]+", " ", row["source"].lower()).strip()
        if len(normalized) < 40:
            continue
        words = normalized.split()
        prefix = " ".join(words[:5])
        length_bucket = len(normalized) // 40
        key = (prefix, length_bucket)
        for other_index, other in buckets[key][-20:]:
            ratio = difflib.SequenceMatcher(a=normalized, b=other).ratio()
            if ratio >= 0.96 and normalized != other:
                pairs.append(
                    {
                        "sample_index_a": index,
                        "sample_index_b": other_index,
                        "similarity": round(ratio, 4),
                    }
                )
                if len(pairs) >= 100:
                    return {
                        "sample_size": len(sample),
                        "near_duplicate_pairs_capped_at": 100,
                        "pairs": pairs,
                    }
        buckets[key].append((index, normalized))
    return {"sample_size": len(sample), "near_duplicate_pairs": len(pairs), "pairs": pairs}


def enrich_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["source_numbers"] = number_tokens(item["source"])
        item["target_numbers"] = number_tokens(item["target"])
        item["source_units"] = unit_tokens(item["source"])
        item["target_units"] = unit_tokens(item["target"])
        item["source_negation"] = negation_markers(item["source"], "en")
        item["target_negation"] = negation_markers(item["target"], "de")
        item["source_dosage"] = dosage_tokens(item["source"])
        item["target_dosage"] = dosage_tokens(item["target"])
        item["source_medicine_names"] = medicine_name_tokens(item["source"])
        item["target_medicine_names"] = medicine_name_tokens(item["target"])
        item["source_warnings"] = warning_markers(item["source"], "en")
        item["target_warnings"] = warning_markers(item["target"], "de")
        enriched.append(item)
    return enriched


def load_ahazeemi_split(name: str) -> list[dict[str, Any]]:
    return read_jsonl(RAW_ROOT / "ahazeemi_processed" / f"{name}.jsonl")


def load_emea_rows() -> tuple[list[dict[str, Any]], str]:
    candidates = [
        (RAW_ROOT / "emea_opus" / "emea_en_de.jsonl", "emea_raw_or_opus"),
        (
            RAW_ROOT / "emea_huggingface_repackage" / "emea_en_de.jsonl",
            "emea_huggingface_repackage",
        ),
    ]
    for path, source_type in candidates:
        rows = read_jsonl(path)
        if rows:
            return rows, source_type
    return [], "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-examples", type=int, default=50000)
    parser.add_argument("--train-output", default="ahazeemi_train.jsonl")
    parser.add_argument("--audit-output", default="data_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_environment()
    ensure_project_dirs()
    random_seed = 42
    rng = random.Random(random_seed)

    raw_train = load_ahazeemi_split("train")
    raw_dev = load_ahazeemi_split("validation") or load_ahazeemi_split("dev")
    raw_test = load_ahazeemi_split("test")

    train, train_audit = deduplicate(raw_train)
    dev, dev_audit = deduplicate(raw_dev)
    test, test_audit = deduplicate(raw_test)

    rng.shuffle(train)
    if args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive")
    train = train[: args.max_train_examples]

    train_output = Path(args.train_output)
    if not train_output.is_absolute():
        train_output = PROCESSED_ROOT / train_output
    train_output = train_output.resolve()
    if PROCESSED_ROOT.resolve() not in train_output.parents:
        raise ValueError("--train-output must remain inside data/processed")

    audit_output = Path(args.audit_output)
    if not audit_output.is_absolute():
        audit_output = ARTIFACT_ROOT / audit_output
    audit_output = audit_output.resolve()
    if ARTIFACT_ROOT.resolve() not in audit_output.parents:
        raise ValueError("--audit-output must remain inside artifacts")

    train = enrich_patterns(train)
    dev = enrich_patterns(dev)
    test = enrich_patterns(test)

    write_jsonl(train_output, train)
    write_jsonl(PROCESSED_ROOT / "ahazeemi_dev.jsonl", dev)
    write_jsonl(PROCESSED_ROOT / "ahazeemi_test.jsonl", test)

    emea_rows, emea_source_type = load_emea_rows()
    emea_rows, emea_audit = deduplicate(emea_rows)
    emea_rows = enrich_patterns(emea_rows[:2000])
    document_ids = [row.get("document_id") for row in emea_rows if row.get("document_id")]
    document_id_available = bool(document_ids)
    write_jsonl(PROCESSED_ROOT / "emea_external.jsonl", emea_rows)

    patterns = {
        "numbers": r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+)?(?!\w)",
        "units": "mg, g, kg, mcg, µg, ml, mL, l, %, mmHg, mmol, mol, tablets, capsules, drops, days, hours, weeks",
        "english_negation": "not, no, without, never, must not, do not, cannot, contraindicated",
        "german_negation": "nicht, kein, keine, keinen, ohne, niemals, verboten, untersagt",
        "dosage": "numeric expressions paired with medical units or tablet/capsule/drop counts",
        "warnings": "warning, caution, do not take/use, keep out of reach, contraindicated, allergic, side effects",
        "medicine_names": "conservative drug-like suffix/product-name heuristic; not a clinical NER model",
    }
    write_json(ARTIFACT_ROOT / "medical_patterns.json", patterns)

    audit = {
        "seed": random_seed,
        "training_policy": f"deterministic shuffle seed 42, capped at {args.max_train_examples} examples",
        "ahazeemi": {
            "train": train_audit,
            "dev": dev_audit,
            "test": test_audit,
            "selected_train_rows": len(train),
            "selected_dev_rows": len(dev),
            "selected_test_rows": len(test),
            "train_near_duplicate_sample": near_duplicate_sample(train),
        },
        "emea": {
            "source_type": emea_source_type,
            "rows": len(emea_rows),
            "audit": emea_audit,
            "document_id_available": document_id_available,
            "document_id_count": len(set(document_ids)),
            "split_policy": (
                "document-level metadata is available for inspection"
                if document_id_available
                else "sentence-level external evaluation; document IDs were not exposed"
            ),
        },
        "files": {
            "train": str(train_output),
            "dev": str(PROCESSED_ROOT / "ahazeemi_dev.jsonl"),
            "test": str(PROCESSED_ROOT / "ahazeemi_test.jsonl"),
            "emea": str(PROCESSED_ROOT / "emea_external.jsonl"),
        },
    }
    write_json(audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
