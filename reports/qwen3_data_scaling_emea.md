# Qwen3 data-scaling comparison

- Baseline: `/home/sreenath/research-space/SovereignMedTranslate/artifacts/qwen3_50k_gated400_emea_comparison.json` (400 evaluation rows)
- Expanded: `/home/sreenath/research-space/SovereignMedTranslate/artifacts/qwen3_100k_gated400_emea_comparison.json` (400 evaluation rows)
- The adapter, retrieval index, and evaluation row count are recorded in the source JSON files.

| Condition | ChrF baseline | ChrF expanded | Δ ChrF | BLEU baseline | BLEU expanded | Δ BLEU | Conflicts filtered (expanded) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 60.0740215512238 | 60.0740215512238 | 0.0 | 30.595030793825334 | 30.595030793825334 | 0.0 | 0 |
| rag | 73.73347794400713 | 78.67292274330327 | 4.939445 | 52.101473197779384 | 59.87228494082227 | 7.770812 | 1718 |
| sft | 68.61086164724635 | 69.29557859813333 | 0.684717 | 43.14139864566407 | 43.73576342663574 | 0.594365 | 0 |

A positive ChrF/BLEU delta is useful only when preservation checks and human-readable examples remain stable.
The experiment is a data-and-training comparison, not a clinical safety certification.
