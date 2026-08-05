# qwen3 comparison on test

- Model: Qwen/Qwen3-4B
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 56.12011511713446 | 34.952316336703355 | 0.98 | 0.82 | 1.0 | 0.94 | 0.96 | 0.94 | 2.194 |
| sft | 60.15954554334204 | 34.64274809426894 | 0.94 | 0.84 | 1.0 | 0.92 | 0.96 | 0.94 | 2.435 |
| rag | 61.66656267014729 | 38.02168417785614 | 0.96 | 0.84 | 1.0 | 0.9 | 0.96 | 0.94 | 2.756 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
