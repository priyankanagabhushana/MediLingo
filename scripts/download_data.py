from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from common import (
    ARTIFACT_ROOT,
    PROJECT_ROOT,
    RAW_ROOT,
    configure_environment,
    ensure_project_dirs,
    extract_pair,
    now_utc,
    sha256_file,
    write_json,
    write_jsonl,
)


AHazeemi_ID = "ahazeemi/opus-medical-en-de"
EMEA_HF_ID = "qanastek/EMEA-V3"
OPUS_PAGE = "https://opus.nlpl.eu/datasets/EMEA"
OPUS_BASE = "https://object.pouta.csc.fi/OPUS-EMEA/v3"


def download_url(url: str, destination: Path, timeout: int = 90) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"url": url, "destination": str(destination), "status": "failed"}
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MediLingo/0.2"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        result["status"] = "downloaded"
        result["bytes"] = destination.stat().st_size
        result["sha256"] = sha256_file(destination)
    except Exception as exc:
        result["error"] = repr(exc)
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
    return result


def save_hf_readme(repo_id: str, target_dir: Path, repo_type: str) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "README.md"
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type=repo_type,
            local_dir=str(target_dir),
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        if Path(downloaded).resolve() != target.resolve() and Path(downloaded).exists():
            shutil.copy2(downloaded, target)
        return {"status": "saved", "path": str(target)}
    except Exception as exc:
        return {"status": "unavailable", "error": repr(exc)}


def extract_hf_dataset() -> dict[str, Any]:
    from datasets import load_dataset

    destination = RAW_ROOT / "ahazeemi_processed"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset_id": AHazeemi_ID,
        "source_type": "ahazeemi_processed",
        "downloaded_at_utc": now_utc(),
        "splits": {},
    }

    for split in ["train", "validation", "dev", "test"]:
        try:
            dataset = load_dataset(
                AHazeemi_ID,
                split=split,
                cache_dir=os.environ["HF_DATASETS_CACHE"],
            )
        except Exception as exc:
            manifest["splits"][split] = {"status": "unavailable", "error": repr(exc)}
            continue

        rows = []
        for index, row in enumerate(dataset):
            pair = extract_pair(dict(row))
            if pair is None:
                continue
            source, target = pair
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "source_language": "en",
                    "target_language": "de",
                    "row_id": index,
                    "dataset": AHazeemi_ID,
                    "split": split,
                }
            )
        output = destination / f"{split}.jsonl"
        write_jsonl(output, rows)
        manifest["splits"][split] = {
            "status": "saved",
            "rows": len(rows),
            "path": str(output),
            "sha256": sha256_file(output),
        }

    manifest["readme"] = save_hf_readme(AHazeemi_ID, destination, "dataset")
    write_json(destination / "source_info.json", manifest)
    return manifest


def parse_moses_archive(archive_path: Path, output_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "archive": str(archive_path),
        "status": "failed",
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            lower = {name: name.lower() for name in names}
            source_candidates = [
                name for name in names
                if re.search(r"(?:\.|[_-])en(?:\.txt)?$", lower[name])
                or lower[name].endswith(".en")
            ]
            target_candidates = [
                name for name in names
                if re.search(r"(?:\.|[_-])de(?:\.txt)?$", lower[name])
                or lower[name].endswith(".de")
            ]

            if not source_candidates or not target_candidates:
                for name in names:
                    if ".en" in lower[name] and not lower[name].endswith(".xml"):
                        source_candidates.append(name)
                    if ".de" in lower[name] and not lower[name].endswith(".xml"):
                        target_candidates.append(name)

            if not source_candidates or not target_candidates:
                result["error"] = f"Could not identify EN/DE files in {names[:20]}"
                return result

            source_name = source_candidates[0]
            target_name = target_candidates[0]
            source_lines = archive.read(source_name).decode("utf-8", errors="replace").splitlines()
            target_lines = archive.read(target_name).decode("utf-8", errors="replace").splitlines()
            rows = []
            for index, (source, target) in enumerate(zip(source_lines, target_lines)):
                source = re.sub(r"\s+", " ", source).strip()
                target = re.sub(r"\s+", " ", target).strip()
                if source and target:
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "source_language": "en",
                            "target_language": "de",
                            "row_id": index,
                            "dataset": "OPUS-EMEA",
                            "split": "external",
                            "document_id": None,
                            "metadata_status": "sentence-level-unless-raw-document-id-is-recovered",
                        }
                    )
            write_jsonl(output_path, rows)
            result.update(
                {
                    "status": "saved",
                    "source_member": source_name,
                    "target_member": target_name,
                    "rows": len(rows),
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                }
            )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def download_raw_emea() -> dict[str, Any]:
    destination = RAW_ROOT / "emea_opus"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_type": "emea_raw_or_opus",
        "opus_page": OPUS_PAGE,
        "downloaded_at_utc": now_utc(),
        "attempts": [],
    }

    candidates = [
        ("moses_en_de", f"{OPUS_BASE}/moses/de-en.txt.zip"),
        ("xml_en_de", f"{OPUS_BASE}/xml/de-en.xml.gz"),
        ("raw_en_de", f"{OPUS_BASE}/raw/de-en.zip"),
    ]
    archive_path = None
    for name, url in candidates:
        suffix = Path(url).suffix or ".bin"
        local = destination / f"{name}{suffix}"
        result = download_url(url, local)
        manifest["attempts"].append(result)
        if result.get("status") == "downloaded":
            if name == "moses_en_de":
                archive_path = local
                break

    if archive_path is not None:
        manifest["parsed"] = parse_moses_archive(
            archive_path,
            destination / "emea_en_de.jsonl",
        )
    else:
        manifest["parsed"] = {
            "status": "not_available",
            "note": "Raw OPUS archive was not retrieved; fallback will be attempted.",
        }

    write_json(destination / "source_info.json", manifest)
    return manifest


def fallback_hf_emea() -> dict[str, Any]:
    destination = RAW_ROOT / "emea_huggingface_repackage"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset_id": EMEA_HF_ID,
        "source_type": "emea_huggingface_repackage",
        "downloaded_at_utc": now_utc(),
        "status": "starting",
    }
    save_hf_readme(EMEA_HF_ID, destination, "dataset")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            EMEA_HF_ID,
            split="train",
            streaming=True,
            cache_dir=os.environ["HF_DATASETS_CACHE"],
        )
        rows = []
        max_rows = 2000
        for row in dataset:
            row = dict(row)
            lang = str(row.get("lang", "")).lower().replace("_", "-")
            pair = extract_pair(row)
            if pair is None:
                source = row.get("source_text")
                target = row.get("target_text")
                if source and target:
                    pair = (str(source).strip(), str(target).strip())
            if pair is None:
                continue
            if lang and not any(token in lang for token in ["en-de", "en_de", "en/de"]):
                continue
            rows.append(
                {
                    "source": pair[0],
                    "target": pair[1],
                    "source_language": "en",
                    "target_language": "de",
                    "row_id": len(rows),
                    "dataset": EMEA_HF_ID,
                    "split": "external",
                    "document_id": row.get("document_id") or row.get("doc_id") or row.get("id"),
                    "metadata_status": "repackaged_fields_must_be_verified",
                    "raw_lang": row.get("lang"),
                }
            )
            if len(rows) >= max_rows:
                break

        output = destination / "emea_en_de.jsonl"
        write_jsonl(output, rows)
        manifest.update(
            {
                "status": "saved",
                "rows": len(rows),
                "path": str(output),
                "sha256": sha256_file(output),
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc)})

    write_json(destination / "source_info.json", manifest)
    return manifest


def main() -> None:
    configure_environment()
    ensure_project_dirs()
    ahazeemi = extract_hf_dataset()
    emea_raw = download_raw_emea()
    raw_jsonl = RAW_ROOT / "emea_opus" / "emea_en_de.jsonl"
    if not raw_jsonl.exists():
        emea_fallback = fallback_hf_emea()
    else:
        emea_fallback = {"status": "not_needed", "note": "Raw OPUS EMEA rows available."}

    manifest = {
        "created_at_utc": now_utc(),
        "project_root": str(PROJECT_ROOT),
        "ahazeemi": ahazeemi,
        "emea_raw": emea_raw,
        "emea_fallback": emea_fallback,
    }
    write_json(ARTIFACT_ROOT / "download_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
