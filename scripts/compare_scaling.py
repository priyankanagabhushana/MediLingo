from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, now_utc, write_json


FIELDS = [
    "chrf",
    "bleu",
    "number_preservation_rate",
    "unit_preservation_rate",
    "dosage_preservation_rate",
    "medicine_name_preservation_rate",
    "negation_preservation_rate",
    "warning_preservation_rate",
    "rag_used_rate",
    "retrieval_no_evidence_rate",
    "entity_conflicts_filtered",
    "seconds_per_example",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expanded", required=True)
    parser.add_argument("--output-json", default="artifacts/qwen3_data_scaling.json")
    parser.add_argument("--output-md", default="reports/qwen3_data_scaling.md")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if PROJECT_ROOT not in path.parents:
        raise ValueError("All report paths must remain inside the project")
    return path


def main() -> None:
    args = parse_args()
    baseline_path = project_path(args.baseline)
    expanded_path = project_path(args.expanded)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
    conditions = sorted(
        set(baseline.get("conditions", {})).intersection(expanded.get("conditions", {}))
    )
    result: dict[str, Any] = {
        "created_at_utc": now_utc(),
        "baseline": str(baseline_path),
        "expanded": str(expanded_path),
        "baseline_rows": baseline.get("rows"),
        "expanded_rows": expanded.get("rows"),
        "conditions": {},
    }
    for condition in conditions:
        before = baseline["conditions"][condition]
        after = expanded["conditions"][condition]
        result["conditions"][condition] = {
            field: {
                "baseline": before.get(field),
                "expanded": after.get(field),
                "delta": (
                    round(after[field] - before[field], 6)
                    if isinstance(before.get(field), (int, float))
                    and isinstance(after.get(field), (int, float))
                    else None
                ),
            }
            for field in FIELDS
        }

    output_json = project_path(args.output_json)
    output_md = project_path(args.output_md)
    write_json(output_json, result)
    lines = [
        "# Qwen3 data-scaling comparison",
        "",
        f"- Baseline: `{baseline_path}` ({baseline.get('rows')} evaluation rows)",
        f"- Expanded: `{expanded_path}` ({expanded.get('rows')} evaluation rows)",
        "- The adapter, retrieval index, and evaluation row count are recorded in the source JSON files.",
        "",
        "| Condition | ChrF baseline | ChrF expanded | Δ ChrF | BLEU baseline | BLEU expanded | Δ BLEU | Conflicts filtered (expanded) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        values = result["conditions"][condition]
        def value(field: str, key: str) -> Any:
            return values[field][key]
        lines.append(
            f"| {condition} | {value('chrf', 'baseline')} | {value('chrf', 'expanded')} | "
            f"{value('chrf', 'delta')} | {value('bleu', 'baseline')} | {value('bleu', 'expanded')} | "
            f"{value('bleu', 'delta')} | {value('entity_conflicts_filtered', 'expanded')} |"
        )
    lines.extend(
        [
            "",
            "A positive ChrF/BLEU delta is useful only when preservation checks and human-readable examples remain stable.",
            "The experiment is a data-and-training comparison, not a clinical safety certification.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
