# MediLingo

MediLingo is an isolated, reproducible English-to-German medical-information translation assistant for administrative use. It is not a clinical decision system and does not provide medical advice.

## What is implemented

- Qwen3-4B BF16 LoRA/SFT on medical parallel text.
- Gemma 4 E2B BF16 LoRA/SFT as a controlled second-model comparison.
- A local translation memory and medical terminology glossary.
- Retrieval-augmented translation with local provenance.
- Deterministic checks for numbers, units, dosage, medicine names, warnings, and negation.
- External evaluation on a separate English-German OPUS EMEA sample.
- Streamlit UI, Docker packaging, and Cloud Run notes.
- JSON, JSONL, CSV, and Markdown reports with timing and failure records.

The central engineering question is: can a small, locally runnable model translate specialized medical information while remaining auditable, resource-aware, and conservative about safety-critical details?

## Data and provenance

### Ahazeemi medical corpus

- Source: ahazeemi/opus-medical-en-de.
- Supplied train/dev/test: 248,099 / 2,000 / 2,000 rows.
- Deterministic first pilot: 50,000 training rows, seed 42.
- Expanded Qwen run: 100,000 training rows, same seed and preparation rules; the adapter is trained separately from the 50k adapter.
- Stored under data/raw/ahazeemi_processed and data/processed.
- Exact row counts, duplicate analysis, extracted medical patterns, and checksums are in:
  - artifacts/data_audit.json
  - artifacts/medical_patterns.json
  - artifacts/download_manifest.json
  - data/raw/ahazeemi_processed/README.md

The preparation step removed 435 exact duplicate training rows before selecting the 50,000-row subset. The 100,000-row preparation is stored separately and contains exactly 100,000 rows under the same deterministic policy. The supplied dev and test sets were kept complete; the original pilot sampled 50 rows, while the expanded Qwen comparison samples 400 rows deterministically.

### OPUS EMEA

- Raw English-German Moses archive stored in data/raw/emea_opus/moses_en_de.zip.
- Raw XCES XML stored in data/raw/emea_opus/xml_de_en.xml.gz.
- Parsed aligned Moses rows: 1,108,752.
- XCES metadata: 1,163,348 links across 1,939 document groups.
- The Moses rows do not expose a direct document ID, so the current EMEA evaluation is explicitly sentence-level. The XML metadata is retained for future document-level reconstruction.
- EMEA is not mixed into SFT training.
- Its retrieval/reference index is separate from the training translation memory.

URLs, dates, versions, checksums, extraction notes, and the fallback policy are recorded in artifacts/download_manifest.json and data/raw/emea_opus/source_info.json.

## Retrieval design

Two local evidence sources are used:

1. Translation memory: English source, approved German target, dataset, split, row ID, hash, provenance, and similarity.
2. Medical glossary: source term, German term, example, source dataset, and confidence.

The index uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 embeddings and a NumPy cosine-similarity search. Test rows are not inserted into the training translation-memory index. EMEA is indexed separately. The expanded RAG path oversamples candidates, filters conflicting product/entity tokens such as Arixtra versus Quixidar, and applies a minimum similarity threshold (0.65 by default). If no non-conflicting candidate survives, the model translates without translation-memory evidence and the UI reports that decision.

The model prompt marks retrieved material as reference-only and marks the source with source_text tags. The answer must contain exactly one German translation. scripts/runtime.py also removes only explicit copied-reference labels such as - English:; ordinary German lists and paragraphs are preserved.

An initial Gemma RAG run exposed why this matters: the model sometimes continued by copying retrieved examples into its answer. That first run is preserved as *_initial_prompt.*. After the prompt and cleanup fix, the same Gemma test RAG condition improved from ChrF 48.02/BLEU 13.62 to ChrF 56.32/BLEU 35.80. This is recorded as a reproducibility and prompt-engineering lesson, not hidden.

## Expanded 400-example Qwen results

The original 50-example pilot remains the historical baseline. With the new gate enabled, a 400-example Qwen 50k comparison produced:

| Dataset | Condition | ChrF | BLEU | RAG used | Conflicts filtered | sec/example |
|---|---|---:|---:|---:|---:|---:|
| ahazeemi test | Base | 55.30 | 29.90 | 0% | 0 | 1.655 |
| ahazeemi test | SFT | 62.64 | 38.58 | 0% | 0 | 2.015 |
| ahazeemi test | SFT + gated RAG | 64.88 | 44.06 | 87.8% | 6,312 | 2.099 |
| EMEA external | Base | 60.07 | 30.60 | 0% | 0 | 1.636 |
| EMEA external | SFT | 68.61 | 43.14 | 0% | 0 | 1.901 |
| EMEA external | SFT + gated RAG | 73.73 | 52.10 | 98.5% | 3,288 | 2.033 |

The separate 100k Qwen run is being evaluated against this baseline; its result should be treated as an experiment, not assumed to be better merely because it uses more data.

## Training

### Qwen3

- Base: Qwen/Qwen3-4B.
- BF16 LoRA/SFT, seed 42, maximum sequence length 1,024.
- LoRA rank 16, alpha 32, dropout 0.05.
- Attention and MLP projections: q/k/v/o and gate/up/down.
- 50,000 examples, capped at 3,000 steps.
- Adapter: models/qwen3-4b-medical-lora.
- Expanded adapter run: 100,000 examples, capped at 6,000 steps, with the same seed/hyperparameters and a separate output directory.
- Training metadata: models/qwen3-4b-medical-lora/training_metadata.json.
- Training log: logs/qwen3_training.log.
- Final training loss: 0.6925; final dev loss: 0.6354.
- Training runtime: about 8,383 seconds on Thor.

### Gemma 4

- Base: google/gemma-4-E2B-it.
- Same data subset, evaluation rows, prompt structure, RAG index, seed, and LoRA hyperparameters.
- Adapter: models/gemma4-e2b-medical-lora.
- Full pilot completed at 3,000 steps.
- Final training loss: 0.83094; final dev loss: 0.7524.
- Training runtime: about 9,208 seconds on Thor.
- Gemma uses multimodal wrapper modules in some towers. The first PEFT smoke failed because it attempted to target Gemma4ClippableLinear wrappers. The project trainer now selects only native torch.nn.Linear q/k/v/o and gate/up/down projections under model.language_model.layers.*. The one-step smoke then passed, followed by the full pilot.
- Initial and corrected smoke details are in artifacts/gemma4_compatibility.json.

The SSH environment did not contain an HF_TOKEN; the model was nevertheless already resolvable from the project-local cache. The browser login was not read or copied into SSH.

## Corrected evaluation results

These are reproducible 50-example pilot slices, not claims about the complete corpora. Values are rates except ChrF, BLEU, and latency.

| Dataset | Model | Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine | Negation | Warnings | sec/example |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ahazeemi test | Qwen3 | base | 55.89 | 31.61 | 0.96 | 0.82 | 1.00 | 0.92 | 0.94 | 0.94 | 2.20 |
| ahazeemi test | Qwen3 | SFT | 60.09 | 33.75 | 0.94 | 0.82 | 0.96 | 0.92 | 0.96 | 0.94 | 2.45 |
| ahazeemi test | Qwen3 | SFT + RAG | 60.54 | 38.43 | 0.96 | 0.84 | 1.00 | 0.92 | 0.94 | 0.94 | 2.77 |
| ahazeemi test | Gemma 4 | base | 58.88 | 34.71 | 0.98 | 0.82 | 1.00 | 0.92 | 0.96 | 0.94 | 2.43 |
| ahazeemi test | Gemma 4 | SFT | 58.94 | 35.69 | 0.96 | 0.84 | 1.00 | 0.90 | 0.96 | 0.94 | 2.89 |
| ahazeemi test | Gemma 4 | SFT + RAG | 56.32 | 35.80 | 0.92 | 0.84 | 0.98 | 0.90 | 0.96 | 0.94 | 2.88 |
| EMEA external | Qwen3 | base | 57.10 | 34.51 | 1.00 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.03 |
| EMEA external | Qwen3 | SFT | 65.13 | 44.79 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.26 |
| EMEA external | Qwen3 | SFT + RAG | 66.37 | 47.41 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.51 |
| EMEA external | Gemma 4 | base | 61.11 | 36.38 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.17 |
| EMEA external | Gemma 4 | SFT | 65.25 | 43.61 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.46 |
| EMEA external | Gemma 4 | SFT + RAG | 68.46 | 47.84 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.66 |

Reports and current raw outputs are in reports/qwen3_test_comparison.*, reports/qwen3_emea_comparison.*, reports/gemma4_test_comparison.*, reports/gemma4_emea_comparison.*, artifacts/*_comparison.json, and artifacts/*_outputs.jsonl. Earlier prompt-version snapshots are intentionally retained with the suffix _initial_prompt.

Automatic fidelity checks are conservative heuristics and do not replace expert review. A validation warning means the result should be reviewed, not that the model is safe.

## Run it on Thor

From the project root:

    .venv/bin/python scripts/system_report.py
    .venv/bin/python scripts/download_data.py
    .venv/bin/python scripts/prepare_data.py
    .venv/bin/python scripts/build_index.py --include-emea
    .venv/bin/python scripts/train_sft.py --model qwen3 --max-steps 3000 --max-train-examples 50000
    .venv/bin/python scripts/prepare_data.py --max-train-examples 100000 --train-output ahazeemi_train_100k.jsonl --audit-output data_audit_100k.json
    .venv/bin/python scripts/build_index.py --max-examples 100000 --input-train-file data/processed/ahazeemi_train_100k.jsonl --output-dir artifacts/rag_100k --include-emea
    .venv/bin/python scripts/train_sft.py --model qwen3 --max-steps 6000 --max-train-examples 100000 --train-file data/processed/ahazeemi_train_100k.jsonl --output-dir models/qwen3-4b-medical-lora-100k
    .venv/bin/python scripts/train_sft.py --model gemma4 --max-steps 3000 --max-train-examples 50000 --output-dir models/gemma4-e2b-medical-lora
    .venv/bin/python scripts/evaluate.py --model qwen3 --condition base,sft,rag --dataset test --max-examples 50 --max-new-tokens 128

The project keeps Hugging Face and Torch caches under .cache. No system-wide package installation is required.

## UI and deployment

The project UI is a Streamlit app in ui/app.py. It offers:

- English input and English-to-German selection.
- Local research mode can compare Qwen3 and Gemma 4 base/SFT adapters; the public Cloud Run demo presents only the Qwen3-4B 100k SFT adapter.
- RAG on/off, translation-memory choice, top-k control, and minimum similarity threshold.
- A switch to block conflicting medicine/entity evidence before it reaches the model.
- Translation output.
- Retrieved examples, terminology, provenance, and validation flags.
- Preloaded medical examples.

Port 8501 was already occupied by an unrelated existing process, so the verified project UI runs at:

    .venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8511

Health check:

    curl -fsS http://127.0.0.1:8511/_stcore/health

The local Docker image is medilingo:local. It contains both LoRA adapters, uses the Streamlit health endpoint, and has been tested with a local container health check. See CLOUD_RUN.md.

The public MediLingo demo is deployed and verified at https://medilingo-osqskujnua-ez.a.run.app. It uses Cloud Run Gen2 in project `retailmind-497115` and region `europe-west4`, with one NVIDIA L4 GPU, 4 vCPUs, and 16 GiB RAM. The service is public, capped at one instance, and configured with zero minimum instances so it scales down when idle. Streamlit uses concurrency 80 so its parallel static assets load correctly. The first request after scale-to-zero can take several minutes while the image and Qwen3 model initialize. Production use would still need authentication, audit logging, request limits, data retention controls, and human review.

## Project framing

A defensible summary is:

> I adapted two small open models for English-German medical-information translation, then added local retrieval so the system could consult approved terminology and previous translation examples. I evaluated base, SFT, and SFT plus RAG separately on the original test slice and on an external EMEA slice. I also checked details that must not change: medicine names, numbers, units, dosage, warnings, and negation.

The most credible research point is that the answer is conditional:

- SFT gives the largest consistent gain over base on the external EMEA sample.
- RAG helps both models on EMEA after prompt control.
- RAG is not automatically beneficial on every in-domain slice; Gemma test RAG is a useful counterexample.
- A healthcare translation system needs provenance, failure checks, reproducible splits, human review, and resource measurements—not only one aggregate score.

Good next research steps are document-level EMEA reconstruction using XML metadata, larger test sets and confidence intervals, expert medical terminology annotation, leakage-resistant retrieval splits, structured constrained decoding for numbers and dosage, human preference evaluation, and eventually preference/RL training with a carefully validated reward.
