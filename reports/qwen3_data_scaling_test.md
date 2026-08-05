# Qwen3 data-scaling comparison

- Baseline: `/home/sreenath/research-space/SovereignMedTranslate/artifacts/qwen3_50k_gated400_test_comparison.json` (400 evaluation rows)
- Expanded: `/home/sreenath/research-space/SovereignMedTranslate/artifacts/qwen3_100k_gated400_test_comparison.json` (400 evaluation rows)
- The adapter, retrieval index, and evaluation row count are recorded in the source JSON files.

| Condition | ChrF baseline | ChrF expanded | Δ ChrF | BLEU baseline | BLEU expanded | Δ BLEU | Conflicts filtered (expanded) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 55.30111973872434 | 55.30111973872434 | 0.0 | 29.90005513371611 | 29.90005513371611 | 0.0 | 0 |
| rag | 64.87927713459783 | 67.05600626847684 | 2.176729 | 44.060942592787974 | 45.99485967026761 | 1.933917 | 4308 |
| sft | 62.63911303998969 | 63.91093812388947 | 1.271825 | 38.57575679675408 | 39.657984603727975 | 1.082228 | 0 |

A positive ChrF/BLEU delta is useful only when preservation checks and human-readable examples remain stable.
The experiment is a data-and-training comparison, not a clinical safety certification.
