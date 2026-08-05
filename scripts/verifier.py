from __future__ import annotations

import re
from collections import Counter
from typing import Any

from common import (
    dosage_tokens,
    medicine_name_tokens,
    negation_markers,
    number_tokens,
    unit_tokens,
    warning_markers,
)


def multiset_equal(left: list[str], right: list[str]) -> bool:
    return Counter(token.lower().replace(",", ".") for token in left) == Counter(
        token.lower().replace(",", ".") for token in right
    )


def _normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def medicine_names_preserved(source_names: list[str], target: str) -> bool:
    if not source_names:
        return True
    target_normalized = _normalized_identifier(target)
    return all(
        _normalized_identifier(name) in target_normalized for name in source_names
    )


def verify_translation(source: str, translation: str) -> dict[str, Any]:
    source_numbers = number_tokens(source)
    target_numbers = number_tokens(translation)
    source_units = unit_tokens(source)
    target_units = unit_tokens(translation)
    source_negation = negation_markers(source, "en")
    target_negation = negation_markers(translation, "de")
    source_dosage = dosage_tokens(source)
    target_dosage = dosage_tokens(translation)
    source_medicine_names = medicine_name_tokens(source)
    target_medicine_names = medicine_name_tokens(translation)
    source_warnings = warning_markers(source, "en")
    target_warnings = warning_markers(translation, "de")

    medicine_preserved = medicine_names_preserved(
        source_medicine_names, translation
    )
    dosage_preserved = multiset_equal(source_dosage, target_dosage)
    warning_preserved = not source_warnings or bool(target_warnings)

    warnings: list[str] = []
    if not multiset_equal(source_numbers, target_numbers):
        warnings.append("number_mismatch")
    if source_units and not multiset_equal(source_units, target_units):
        warnings.append("unit_mismatch")
    if source_negation and not target_negation:
        warnings.append("possible_negation_loss")
    if not medicine_preserved:
        warnings.append("possible_medicine_name_loss")
    if not dosage_preserved:
        warnings.append("dosage_mismatch")
    if not warning_preserved:
        warnings.append("possible_warning_loss")

    return {
        "safe_for_automatic_use": not warnings,
        "warnings": warnings,
        "source_numbers": source_numbers,
        "target_numbers": target_numbers,
        "source_units": source_units,
        "target_units": target_units,
        "source_negation": source_negation,
        "target_negation": target_negation,
        "source_dosage": source_dosage,
        "target_dosage": target_dosage,
        "source_medicine_names": source_medicine_names,
        "target_medicine_names": target_medicine_names,
        "source_warnings": source_warnings,
        "target_warnings": target_warnings,
        "number_preserved": multiset_equal(source_numbers, target_numbers),
        "unit_preserved": multiset_equal(source_units, target_units),
        "negation_present_when_expected": not source_negation or bool(target_negation),
        "medicine_name_preserved": medicine_preserved,
        "dosage_preserved": dosage_preserved,
        "warning_preserved": warning_preserved,
    }
