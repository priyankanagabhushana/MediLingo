# gemma4 comparison on emea

- Model: google/gemma-4-E2B-it
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 61.11253617555709 | 36.38122630588284 | 0.98 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.172 |
| sft | 65.25456680949269 | 43.607605730811116 | 0.98 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.463 |
| rag | 68.45872101939403 | 47.84220474866133 | 0.98 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.661 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
