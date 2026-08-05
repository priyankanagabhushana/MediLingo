# qwen3 comparison on emea

- Model: Qwen/Qwen3-4B
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 56.96113028400248 | 33.663438159223325 | 1.0 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.046 |
| sft | 64.89562569429933 | 44.128203895929545 | 0.94 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.22 |
| rag | 68.11886021388258 | 49.21021928072785 | 0.96 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.499 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
