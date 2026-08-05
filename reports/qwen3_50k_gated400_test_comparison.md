# qwen3_50k_gated400 comparison on test

- Model: Qwen/Qwen3-4B
- Run name: qwen3_50k_gated400
- Rows: 400
- Similarity gate: 0.65
- Entity filtering: True

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | RAG used | Conflicts filtered | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 55.30111973872434 | 29.90005513371611 | 0.9775 | 0.875 | 0.9925 | 0.8525 | 0.9925 | 0.9675 | 0.0 | 0 | 1.655 |
| sft | 62.63911303998969 | 38.57575679675408 | 0.93 | 0.8825 | 0.985 | 0.8575 | 0.995 | 0.97 | 0.0 | 0 | 2.015 |
| rag | 64.87927713459783 | 44.060942592787974 | 0.9275 | 0.8875 | 0.99 | 0.855 | 0.995 | 0.97 | 0.8775 | 6312 | 2.099 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative healthcare translation aid, not a clinical decision system.
