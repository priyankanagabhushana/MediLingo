# qwen3_100k_gated400 comparison on test

- Model: Qwen/Qwen3-4B
- Run name: qwen3_100k_gated400
- Rows: 400
- Similarity gate: 0.65
- Entity filtering: True

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | RAG used | Conflicts filtered | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 55.30111973872434 | 29.90005513371611 | 0.9775 | 0.875 | 0.9925 | 0.8525 | 0.9925 | 0.9675 | 0.0 | 0 | 1.662 |
| sft | 63.91093812388947 | 39.657984603727975 | 0.9325 | 0.8825 | 0.9825 | 0.8525 | 0.9925 | 0.97 | 0.0 | 0 | 2.031 |
| rag | 67.05600626847684 | 45.99485967026761 | 0.925 | 0.8825 | 0.9875 | 0.85 | 0.995 | 0.97 | 0.93 | 4308 | 2.2 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative healthcare translation aid, not a clinical decision system.
