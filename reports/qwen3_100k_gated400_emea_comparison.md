# qwen3_100k_gated400 comparison on emea

- Model: Qwen/Qwen3-4B
- Run name: qwen3_100k_gated400
- Rows: 400
- Similarity gate: 0.65
- Entity filtering: True

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | RAG used | Conflicts filtered | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 60.0740215512238 | 30.595030793825334 | 0.9925 | 0.8325 | 1.0 | 0.695 | 0.995 | 0.9725 | 0.0 | 0 | 1.653 |
| sft | 69.29557859813333 | 43.73576342663574 | 0.965 | 0.8625 | 0.9875 | 0.6925 | 0.995 | 0.9725 | 0.0 | 0 | 1.926 |
| rag | 78.67292274330327 | 59.87228494082227 | 0.965 | 0.86 | 0.985 | 0.69 | 0.9925 | 0.9725 | 0.9925 | 1718 | 2.062 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative healthcare translation aid, not a clinical decision system.
