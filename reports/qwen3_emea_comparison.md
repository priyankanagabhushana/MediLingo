# qwen3 comparison on emea

- Model: Qwen/Qwen3-4B
- Rows: 50

| Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine names | Negation | Warnings preserved | Seconds/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 57.095953102711896 | 34.50576311247183 | 1.0 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.033 |
| sft | 65.13136124467026 | 44.7883877128932 | 0.98 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.255 |
| rag | 66.36881787034834 | 47.40750995937742 | 0.98 | 0.78 | 1.0 | 0.84 | 1.0 | 0.98 | 2.512 |

ChrF and BLEU are automatic translation metrics. The preservation columns are conservative deterministic checks and require human review.
The system is an administrative translation aid, not a clinical decision system.
