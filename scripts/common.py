from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
REPORT_ROOT = PROJECT_ROOT / "reports"
MODEL_ROOT = PROJECT_ROOT / "models"
CACHE_ROOT = PROJECT_ROOT / ".cache"


def configure_environment() -> None:
    """Force all model/dataset caches into the dedicated project folder."""
    paths = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_DATASETS_CACHE": CACHE_ROOT / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    for key, value in paths.items():
        os.environ[key] = str(value)


def ensure_project_dirs() -> None:
    for path in [
        DATA_ROOT,
        RAW_ROOT,
        PROCESSED_ROOT,
        ARTIFACT_ROOT,
        REPORT_ROOT,
        MODEL_ROOT,
        CACHE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    """Extract English/German text from common HF parallel-corpus schemas."""
    translation = row.get("translation")
    if isinstance(translation, dict):
        source = translation.get("en") or translation.get("eng")
        target = translation.get("de") or translation.get("deu")
        if source and target:
            return normalize_text(source), normalize_text(target)

    for source_key, target_key in [
        ("en", "de"),
        ("eng", "deu"),
        ("source_text", "target_text"),
        ("source", "target"),
    ]:
        source = row.get(source_key)
        target = row.get(target_key)
        if source and target:
            return normalize_text(source), normalize_text(target)

    return None


def number_tokens(text: str) -> list[str]:
    return re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+)?(?!\w)", text)


def unit_tokens(text: str) -> list[str]:
    pattern = (
        r"(?<!\w)(?:mg|g|kg|mcg|µg|μg|ml|mL|l|%|mmHg|mmol|mol|"
        r"tablets?|capsules?|drops?|days?|hours?|weeks?)(?!\w)"
    )
    return [token.lower() for token in re.findall(pattern, text, flags=re.IGNORECASE)]


def negation_markers(text: str, language: str) -> list[str]:
    if language == "de":
        markers = r"\b(?:nicht|kein(?:e|en|er|es)?|ohne|niemals|verboten|untersagt)\b"
    else:
        markers = r"\b(?:not|no|without|never|must not|do not|cannot|contraindicated)\b"
    return re.findall(markers, text, flags=re.IGNORECASE)


def dosage_tokens(text: str) -> list[str]:
    """Return normalized numeric dosage expressions such as 10 mg or 5 tablets."""
    pattern = (
        r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|g|kg|mcg|\u00b5g|\u03bcg|ml|mL|l|%|"
        r"mmHg|mmol|mol|tablets?|capsules?|drops?)(?!\w)"
    )
    return [
        re.sub(r"\s+", "", token.lower().replace(",", "."))
        for token in re.findall(pattern, text, flags=re.IGNORECASE)
    ]


def warning_markers(text: str, language: str) -> list[str]:
    if language == "de":
        markers = (
            r"\b(?:warnung|achtung|nicht einnehmen|nicht verwenden|nicht nehmen|"
            r"nicht anwenden|darf nicht eingenommen werden|darf nicht verwendet werden|"
            r"nicht eingenommen werden|nicht verwendet werden|f\u00fcr kinder unzug\u00e4nglich|"
            r"sofort \u00e4rztliche hilfe|gegenanzeige|allergisch|bei nebenwirkungen)\b"
        )
    else:
        markers = (
            r"\b(?:warning|caution|do not take|do not use|keep out of reach|"
            r"seek medical help immediately|contraindicated|allergic|"
            r"side effects)\b"
        )
    return re.findall(markers, text, flags=re.IGNORECASE)


def medicine_name_tokens(text: str) -> list[str]:
    """Conservative medicine-name heuristic for preservation auditing."""
    suffixes = (
        "cillin", "mycin", "cycline", "pril", "sartan", "olol", "azole",
        "statin", "caine", "mab", "vir", "zepam", "pine", "oxetine",
        "triptan", "gliptin", "nib", "lukast",
    )
    stop = {
        "take", "give", "use", "once", "twice", "daily", "after", "before",
        "with", "without", "the", "this", "medicine", "medication", "tablet",
        "tablets", "capsule", "capsules", "solution", "warning", "adults",
        "children", "keep", "reach", "sight",
    }
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,}\b", text)
    found: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stop:
            continue
        drug_like = any(lowered.endswith(suffix) for suffix in suffixes)
        product_like = any(char.isdigit() for char in token) or "-" in token
        if drug_like or product_like:
            found.append(token)
    dosage_pattern = (
        r"\b([A-Za-z][A-Za-z0-9-]{3,})\b\s+\d+(?:[.,]\d+)?\s*"
        r"(?:mg|g|kg|mcg|\u00b5g|\u03bcg|ml|mL|l|%|mmHg|mmol|mol|tablets?|capsules?|drops?)\b"
    )
    for match in re.finditer(dosage_pattern, text, flags=re.IGNORECASE):
        candidate = match.group(1)
        if candidate.lower() not in stop:
            found.append(candidate)
    return list(dict.fromkeys(found))


def retrieval_entity_tokens(text: str) -> list[str]:
    """Extract conservative product/entity tokens for retrieval safety.

    This is intentionally separate from the fidelity heuristic. Retrieval should
    notice product names such as Arixtra and Quixidar even when they do not match
    one of the drug-name suffixes used by the validation checker.
    """
    stop = {
        "After", "Before", "Children", "Daily", "During", "For", "From",
        "Keep", "Once", "Rare", "Take", "Table", "The", "This", "Twice",
        "Use", "When", "What", "You",
    }
    found = list(medicine_name_tokens(text))
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9-]{3,}\b", text):
        token = match.group(0)
        prefix = text[: match.start()].rstrip()
        # Common sentence starters are excluded, while an unusual first word
        # such as Arixtra is retained as a possible product name.
        if token in stop:
            continue
        if prefix and prefix[-1] in ".!?":
            continue
        found.append(token)
    for match in re.finditer(r"\b[A-Z]{2,}[A-Za-z0-9-]*\b", text):
        found.append(match.group(0))
    return list(dict.fromkeys(found))


def source_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()[:16]
