# MediLingo project overview

MediLingo is an English-to-German medical-information translation assistant for administrative communication. Its focus is faithful translation, local provenance, and conservative handling of safety-critical details.

## System design

- A small locally runnable language model produces a draft translation.
- Translation memory and a medical terminology glossary provide reusable context.
- Retrieval augments the draft with approved local references.
- Deterministic checks compare numbers, units, medicines, warnings, dosage details, and negation.
- Reports record timing, validation results, and failure cases for later review.

## Evaluation approach

The project separates training material from an external English-German evaluation sample. Data preparation, duplicate handling, checksums, and measured outputs are recorded in the `artifacts/` and `reports/` directories.

## Safety boundary

The system supports administrative information workflows. It does not diagnose, prescribe, or replace review by a qualified healthcare professional. Public or de-identified text should be used for local experiments, and generated translations should be checked before use.

## Reproducibility

Training, preparation, evaluation, and container instructions are kept in the repository so that a run can be repeated without relying on an undocumented local state.
