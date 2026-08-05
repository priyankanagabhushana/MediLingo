# gemma4 comparison on test

- Model: google/gemma-4-E2B-it
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 58.881293375244304 | 34.71386040686329 | 0.98 | 0.82 | 1.0 | 0.92 | 0.96 | 0.94 | 2.425 |
| sft | 58.9415003221094 | 35.688386544061885 | 0.96 | 0.84 | 1.0 | 0.9 | 0.96 | 0.94 | 2.885 |
| rag | 56.3211641274468 | 35.79609228119892 | 0.92 | 0.84 | 0.98 | 0.9 | 0.96 | 0.94 | 2.876 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
