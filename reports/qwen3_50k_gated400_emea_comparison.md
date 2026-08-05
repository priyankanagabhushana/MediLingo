# qwen3_50k_gated400 comparison on emea

- Model: Qwen/Qwen3-4B
- Run name: qwen3_50k_gated400
- Rows: 400
- Similarity gate: 0.65
- Entity filtering: True

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | RAG used | Conflicts filtered | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 60.0740215512238 | 30.595030793825334 | 0.9925 | 0.8325 | 1.0 | 0.695 | 0.995 | 0.9725 | 0.0 | 0 | 1.636 |
| sft | 68.61086164724635 | 43.14139864566407 | 0.9525 | 0.865 | 0.9825 | 0.69 | 0.995 | 0.9725 | 0.0 | 0 | 1.901 |
| rag | 73.73347794400713 | 52.101473197779384 | 0.9475 | 0.8625 | 0.99 | 0.685 | 0.9875 | 0.9725 | 0.985 | 3288 | 2.033 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative healthcare translation aid, not a clinical decision system.
