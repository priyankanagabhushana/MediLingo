# gemma4 comparison on test

- Model: google/gemma-4-E2B-it
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 59.48902941433643 | 33.902542978438355 | 0.98 | 0.82 | 1.0 | 0.92 | 0.96 | 0.94 | 2.102 |
| sft | 59.35430181791108 | 35.22194308447099 | 0.94 | 0.84 | 0.98 | 0.9 | 0.96 | 0.94 | 2.822 |
| rag | 48.02486107542509 | 13.622651694281968 | 0.74 | 0.82 | 0.92 | 0.94 | 0.96 | 0.94 | 6.432 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
