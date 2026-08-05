from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_ROOT,
    MODEL_ROOT,
    PROCESSED_ROOT,
    PROJECT_ROOT,
    REPORT_ROOT,
    configure_environment,
    ensure_project_dirs,
    now_utc,
    read_jsonl,
    write_json,
    write_jsonl,
)
from runtime import TranslationRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "gemma4"], required=True)
    parser.add_argument("--condition", default="base,sft,rag")
    parser.add_argument("--dataset", choices=["test", "dev", "emea"], default="test")
    parser.add_argument("--max-examples", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--index-dir", default="artifacts/rag")
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--no-entity-filter", action="store_true")
    return parser.parse_args()


def model_config(name: str) -> tuple[str, Path]:
    if name == "qwen3":
        return "Qwen/Qwen3-4B", MODEL_ROOT / "qwen3-4b-medical-lora"
    return "google/gemma-4-E2B-it", MODEL_ROOT / "gemma4-e2b-medical-lora"


def evaluate_condition(
    model_id: str,
    adapter_dir: Path | None,
    rows: list[dict[str, Any]],
    condition: str,
    max_new_tokens: int,
    index_dir: Path,
    similarity_threshold: float,
    entity_filter: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_rag = condition == "rag"
    runtime = TranslationRuntime(
        model_id,
        adapter_dir=adapter_dir,
        index_dir=index_dir,
    )
    predictions: list[str] = []
    references: list[str] = []
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for index, row in enumerate(rows):
            result = runtime.translate(
                row["source"],
                use_rag=use_rag,
                similarity_threshold=similarity_threshold,
                entity_filter=entity_filter,
                max_new_tokens=max_new_tokens,
            )
            result.update(
                {
                    "row_id": row.get("row_id", index),
                    "reference": row["target"],
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "condition": condition,
                }
            )
            predictions.append(result["translation"])
            references.append(row["target"])
            outputs.append(result)
            if (index + 1) % 25 == 0:
                print(f"{condition}: {index + 1}/{len(rows)}")
    finally:
        runtime.close()

    elapsed = time.perf_counter() - started
    try:
        import sacrebleu

        chrf_score = sacrebleu.corpus_chrf(predictions, [references]).score
        bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    except Exception as exc:
        chrf_score = None
        bleu_score = None
        metric_error = repr(exc)
    else:
        metric_error = None

    verification = [item["verification"] for item in outputs]
    diagnostics = [
        item.get("retrieved", {}).get("diagnostics", {})
        for item in outputs
    ]
    similarity_values = [
        float(item["best_raw_similarity"])
        for item in diagnostics
        if item.get("best_raw_similarity") is not None
    ]
    metrics = {
        "created_at_utc": now_utc(),
        "condition": condition,
        "examples": len(outputs),
        "chrf": chrf_score,
        "bleu": bleu_score,
        "number_preservation_rate": _rate(verification, "number_preserved"),
        "unit_preservation_rate": _rate(verification, "unit_preserved"),
        "negation_preservation_rate": _rate(
            verification,
            "negation_present_when_expected",
        ),
        "medicine_name_preservation_rate": _rate(
            verification,
            "medicine_name_preserved",
        ),
        "dosage_preservation_rate": _rate(verification, "dosage_preserved"),
        "warning_preservation_rate": _rate(verification, "warning_preserved"),
        "warning_rate": (
            sum(bool(item["warnings"]) for item in verification) / len(verification)
            if verification
            else None
        ),
        "rag_used_rate": (
            sum(bool(item.get("rag_used")) for item in outputs) / len(outputs)
            if outputs
            else None
        ),
        "retrieval_no_evidence_rate": (
            sum(
                item.get("retrieval_gate") == "no_sufficient_non_conflicting_evidence"
                for item in diagnostics
            ) / len(outputs)
            if outputs
            else None
        ),
        "entity_conflicts_filtered": sum(
            int(item.get("entity_conflicts_filtered", 0)) for item in diagnostics
        ),
        "avg_best_raw_similarity": (
            round(sum(similarity_values) / len(similarity_values), 4)
            if similarity_values
            else None
        ),
        "similarity_threshold": similarity_threshold,
        "entity_filter": entity_filter,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_example": round(elapsed / len(outputs), 3) if outputs else None,
        "metric_error": metric_error,
    }
    return outputs, metrics


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 4)


def write_report_files(
    model: str,
    dataset: str,
    comparison: dict[str, Any],
    run_name: str,
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "condition",
        "examples",
        "chrf",
        "bleu",
        "number_preservation_rate",
        "unit_preservation_rate",
        "dosage_preservation_rate",
        "medicine_name_preservation_rate",
        "negation_preservation_rate",
        "warning_preservation_rate",
        "warning_rate",
        "rag_used_rate",
        "retrieval_no_evidence_rate",
        "entity_conflicts_filtered",
        "avg_best_raw_similarity",
        "elapsed_seconds",
        "seconds_per_example",
    ]
    csv_path = REPORT_ROOT / f"{run_name}_{dataset}_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for condition, metrics in comparison["conditions"].items():
            writer.writerow(
                {
                    field: metrics.get(field) if field != "condition" else condition
                    for field in metric_fields
                }
            )

    md_path = REPORT_ROOT / f"{run_name}_{dataset}_comparison.md"
    lines = [
        f"# {run_name} comparison on {dataset}",
        "",
        f"- Model: {comparison['model_id']}",
        f"- Run name: {run_name}",
        f"- Rows: {comparison['rows']}",
        f"- Similarity gate: {comparison['similarity_threshold']}",
        f"- Entity filtering: {comparison['entity_filter']}",
        "",
        "| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | RAG used | Conflicts filtered | Seconds/example |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in comparison["conditions"].items():
        lines.append(
            "| {condition} | {chrf} | {bleu} | {numbers} | {units} | {dosage} | "
            "{medicine} | {negation} | {warnings} | {rag_used} | {conflicts} | {latency} |".format(
                condition=condition,
                chrf=metrics.get("chrf"),
                bleu=metrics.get("bleu"),
                numbers=metrics.get("number_preservation_rate"),
                units=metrics.get("unit_preservation_rate"),
                dosage=metrics.get("dosage_preservation_rate"),
                medicine=metrics.get("medicine_name_preservation_rate"),
                negation=metrics.get("negation_preservation_rate"),
                warnings=metrics.get("warning_preservation_rate"),
                rag_used=metrics.get("rag_used_rate"),
                conflicts=metrics.get("entity_conflicts_filtered"),
                latency=metrics.get("seconds_per_example"),
            )
        )
    lines.extend(
        [
            "",
            "ChrF and BLEU are automatic translation metrics. The preservation "
            "columns are conservative deterministic checks and require human review.",
            "The system is an administrative healthcare translation aid, not a clinical decision system.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_environment()
    ensure_project_dirs()
    model_id, configured_adapter = model_config(args.model)

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else configured_adapter
    if not adapter_dir.is_absolute():
        adapter_dir = PROJECT_ROOT / adapter_dir
    adapter_dir = adapter_dir.resolve()
    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir
    index_dir = index_dir.resolve()
    if PROJECT_ROOT not in adapter_dir.parents or PROJECT_ROOT not in index_dir.parents:
        raise ValueError("Adapter and index paths must remain inside the project")
    run_name = args.run_name or args.model

    data_path = {
        "test": PROCESSED_ROOT / "ahazeemi_test.jsonl",
        "dev": PROCESSED_ROOT / "ahazeemi_dev.jsonl",
        "emea": PROCESSED_ROOT / "emea_external.jsonl",
    }[args.dataset]
    rows = read_jsonl(data_path)
    if not rows:
        raise RuntimeError(f"No prepared evaluation rows found at {data_path}")
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    conditions = [item.strip() for item in args.condition.split(",") if item.strip()]
    comparison: dict[str, Any] = {
        "created_at_utc": now_utc(),
        "model": args.model,
        "model_id": model_id,
        "dataset": args.dataset,
        "run_name": run_name,
        "adapter_dir": str(adapter_dir) if adapter_dir.exists() else None,
        "index_dir": str(index_dir),
        "similarity_threshold": args.similarity_threshold,
        "entity_filter": not args.no_entity_filter,
        "rows": len(rows),
        "conditions": {},
    }
    example_bundle: list[dict[str, Any]] = []

    for condition in conditions:
        if condition not in {"base", "sft", "rag"}:
            raise ValueError(f"Unknown condition: {condition}")
        if condition in {"sft", "rag"} and not adapter_dir.exists():
            raise RuntimeError(
                f"{condition} requested but adapter is missing at {adapter_dir}. "
                "Run train_sft.py first or pass --adapter-dir."
            )
        outputs, metrics = evaluate_condition(
            model_id,
            adapter_dir if condition in {"sft", "rag"} else None,
            rows,
            condition,
            args.max_new_tokens,
            index_dir,
            args.similarity_threshold,
            not args.no_entity_filter,
        )
        output_path = ARTIFACT_ROOT / f"{run_name}_{args.dataset}_{condition}_outputs.jsonl"
        write_jsonl(output_path, outputs)
        metrics["outputs_path"] = str(output_path)
        comparison["conditions"][condition] = metrics
        for item in outputs[:20]:
            example_bundle.append(
                {
                    "model": args.model,
                    "run_name": run_name,
                    "condition": condition,
                    "source": item["source"],
                    "reference": item["reference"],
                    "translation": item["translation"],
                    "verification": item["verification"],
                    "retrieved": item["retrieved"],
                }
            )

    comparison_path = ARTIFACT_ROOT / f"{run_name}_{args.dataset}_comparison.json"
    write_json(comparison_path, comparison)
    write_report_files(args.model, args.dataset, comparison, run_name)
    examples_path = ARTIFACT_ROOT / "examples.json"
    existing_examples = []
    if examples_path.exists():
        try:
            existing_examples = json.loads(examples_path.read_text(encoding="utf-8"))
        except Exception:
            existing_examples = []
    write_json(examples_path, existing_examples + example_bundle)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
