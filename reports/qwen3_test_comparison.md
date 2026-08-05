# qwen3 comparison on test

- Model: Qwen/Qwen3-4B
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 55.893851322687674 | 31.614542418333862 | 0.96 | 0.82 | 1.0 | 0.92 | 0.94 | 0.94 | 2.202 |
| sft | 60.0864696477725 | 33.749053241320844 | 0.94 | 0.82 | 0.96 | 0.92 | 0.96 | 0.94 | 2.454 |
| rag | 60.54285795234606 | 38.43195296016254 | 0.96 | 0.84 | 1.0 | 0.92 | 0.94 | 0.94 | 2.766 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
