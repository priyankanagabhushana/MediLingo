from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

from common import ARTIFACT_ROOT, PROJECT_ROOT, RAW_ROOT, configure_environment, ensure_project_dirs, now_utc, write_json


def main() -> None:
    configure_environment()
    ensure_project_dirs()
    xml_path = RAW_ROOT / "emea_opus" / "xml_de_en.xml.gz"
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    groups = 0
    links = 0
    documents: Counter[str] = Counter()
    sample = []
    current = None

    with gzip.open(xml_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "<linkGrp" in line:
                groups += 1
                to_doc = re.search(r'toDoc="([^"]+)"', line)
                from_doc = re.search(r'fromDoc="([^"]+)"', line)
                current = {
                    "toDoc": to_doc.group(1) if to_doc else None,
                    "fromDoc": from_doc.group(1) if from_doc else None,
                }
                documents[current["toDoc"] or "unknown"] += 1
                if len(sample) < 10:
                    sample.append(current)
            if "<link " in line:
                links += 1

    report = {
        "created_at_utc": now_utc(),
        "xml_path": str(xml_path),
        "xces_link_groups": groups,
        "xces_links": links,
        "unique_to_documents": len(documents),
        "sample_document_pairs": sample,
        "document_metadata_available_in_raw_xces": groups > 0,
        "plain_text_alignment_warning": (
            "The Moses plain-text rows do not have a guaranteed one-to-one order with "
            "all XCES links because the XCES contains one-to-many/many-to-one alignments. "
            "Document-level splitting requires reconstructing rows from the raw XML documents."
        ),
    }
    write_json(ARTIFACT_ROOT / "emea_xml_metadata.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
