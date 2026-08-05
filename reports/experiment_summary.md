# MediLingo experiment summary

Date: 2026-08-05
Host: Thor, NVIDIA Thor, approximately 128 GB unified memory
Scope: isolated project at /home/sreenath/research-space/SovereignMedTranslate

## One-sentence story

This project asks whether a small open model can translate English medical information into German locally, while a translation memory and glossary provide auditable terminology evidence and deterministic checks catch changes to numbers, dosage, units, medicines, warnings, and negation.

## What was built

1. Prepared ahazeemi/opus-medical-en-de without mixing EMEA into SFT.
2. Selected 50,000 training examples deterministically with seed 42 and kept complete supplied dev/test files.
3. Downloaded and checksummed OPUS EMEA Moses data and XCES document metadata.
4. Kept EMEA as external evaluation/reference data. Because row-level Moses document IDs are unavailable, current EMEA results are sentence-level.
5. Built a NumPy cosine translation-memory index and a separate EMEA reference index.
6. Built a medical glossary and conservative fidelity verifier.
7. Trained Qwen3-4B and Gemma 4 E2B LoRA adapters for the same 3,000-step pilot.
8. Compared base, SFT, and SFT+RAG under a corrected shared prompt.
9. Expanded the Qwen evaluation to 400 in-domain and 400 external EMEA examples.
10. Added entity-aware retrieval filtering and similarity gating, then started a separate 100,000-example Qwen adapter run.
11. Built and health-tested a Streamlit UI and Docker image.

## Data story

The ahazeemi source supplied 248,099 train rows, 2,000 dev rows, and 2,000 test rows. Duplicate auditing removed 435 exact duplicate training rows before the deterministic 50,000-row selection. Medical-pattern extraction records medicine-like names, numbers, dosage expressions, units, warnings, and negation cues.

The OPUS EMEA Moses archive has 1,108,752 aligned sentence rows. The retained XCES XML has 1,163,348 links across 1,939 document groups. The XML proves that document metadata exists in the source, but the parsed Moses rows do not carry direct IDs that can safely be joined in this run. Therefore the report does not claim document-level splitting; that is a future improvement.

## Training story

### Qwen3-4B

Qwen3 was the first model because it is small enough to run locally and supports the intended non-thinking translation mode. It used BF16 LoRA/SFT with seed 42, max length 1,024, rank 16, alpha 32, dropout 0.05, q/k/v/o and gate/up/down targets, 50,000 examples, and 3,000 steps.

The adapter is models/qwen3-4b-medical-lora. Its final training loss was 0.6925 and final dev loss 0.6354. The recorded runtime was about 8,383 seconds.

### Gemma 4 E2B

Gemma 4 was a controlled comparison, not an attempt to claim that the newest model must win. The first PEFT smoke failed because the generic target selection reached Gemma4ClippableLinear modules in multimodal towers. The fix restricted LoRA to native language-tower torch.nn.Linear projections. A one-step smoke then passed, followed by the 3,000-step run.

The adapter is models/gemma4-e2b-medical-lora. The final training loss was 0.83094, final dev loss 0.7524, and recorded runtime about 9,208 seconds. This architecture lesson is valuable: a new open model is not plug-and-play just because it is small enough for the hardware.

## RAG story and the prompt lesson

The first RAG prompt placed retrieved bilingual examples before the source but did not delimit the final answer strongly enough. Gemma sometimes translated correctly and then continued by copying retrieved examples. Its first 50-row test RAG result was therefore misleadingly poor: ChrF 48.02 and BLEU 13.62.

That run is preserved as *_initial_prompt.*. The corrected prompt labels retrieved text as reference-only, places the real source inside source_text tags, instructs the model to output exactly one German translation, and removes only explicit copied-reference labels. The same Gemma test RAG run then produced ChrF 56.32 and BLEU 35.80. This is a concrete example of why RAG quality is a joint property of retrieval, prompt design, generation control, and validation.

## Corrected results

These are 50-example pilot slices. They are useful for demonstrating the pipeline, not statistical proof of superiority.

### ahazeemi test

| Model | Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine | Negation | Warnings | sec/example |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 | Base | 55.89 | 31.61 | 0.96 | 0.82 | 1.00 | 0.92 | 0.94 | 0.94 | 2.20 |
| Qwen3 | SFT | 60.09 | 33.75 | 0.94 | 0.82 | 0.96 | 0.92 | 0.96 | 0.94 | 2.45 |
| Qwen3 | SFT + RAG | 60.54 | 38.43 | 0.96 | 0.84 | 1.00 | 0.92 | 0.94 | 0.94 | 2.77 |
| Gemma 4 | Base | 58.88 | 34.71 | 0.98 | 0.82 | 1.00 | 0.92 | 0.96 | 0.94 | 2.43 |
| Gemma 4 | SFT | 58.94 | 35.69 | 0.96 | 0.84 | 1.00 | 0.90 | 0.96 | 0.94 | 2.89 |
| Gemma 4 | SFT + RAG | 56.32 | 35.80 | 0.92 | 0.84 | 0.98 | 0.90 | 0.96 | 0.94 | 2.88 |

### EMEA external, sentence-level

| Model | Condition | ChrF | BLEU | Numbers | Units | Dosage | Medicine | Negation | Warnings | sec/example |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 | Base | 57.10 | 34.51 | 1.00 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.03 |
| Qwen3 | SFT | 65.13 | 44.79 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.26 |
| Qwen3 | SFT + RAG | 66.37 | 47.41 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.51 |
| Gemma 4 | Base | 61.11 | 36.38 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.17 |
| Gemma 4 | SFT | 65.25 | 43.61 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.46 |
| Gemma 4 | SFT + RAG | 68.46 | 47.84 | 0.98 | 0.78 | 1.00 | 0.84 | 1.00 | 0.98 | 2.66 |

## Expanded 400-example Qwen comparison

The original 50-example pilot remains above for reproducibility. The following run evaluates 400 examples per condition with the new retrieval gate (minimum similarity 0.65, product/entity conflict filtering enabled). EMEA is sentence-level external evaluation.

### ahazeemi test, Qwen3 50k adapter and 50k memory

| Condition | ChrF | BLEU | Numbers | Units | Medicine | RAG used | Conflicts filtered | sec/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 55.30 | 29.90 | 0.978 | 0.875 | 0.853 | 0% | 0 | 1.655 |
| SFT | 62.64 | 38.58 | 0.930 | 0.883 | 0.858 | 0% | 0 | 2.015 |
| SFT + gated RAG | 64.88 | 44.06 | 0.928 | 0.888 | 0.855 | 87.8% | 6,312 | 2.099 |

### EMEA external, sentence-level

| Condition | ChrF | BLEU | Numbers | Units | Medicine | RAG used | Conflicts filtered | sec/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 60.07 | 30.60 | 0.993 | 0.833 | 0.695 | 0% | 0 | 1.636 |
| SFT | 68.61 | 43.14 | 0.953 | 0.865 | 0.690 | 0% | 0 | 1.901 |
| SFT + gated RAG | 73.73 | 52.10 | 0.948 | 0.863 | 0.685 | 98.5% | 3,288 | 2.033 |

The gate is simple enough to explain in a research discussion: retrieve several candidates, reject a candidate mentioning a conflicting product/entity, reject weak matches below the threshold, and let the model translate without a translation-memory example when no safe evidence remains. The 100k Qwen adapter is intentionally being evaluated separately rather than replacing these baselines.

## What the results mean

- SFT is the strongest consistent intervention over base on the external EMEA sample.
- RAG improves both models on EMEA after prompt control.
- RAG is not guaranteed to improve every in-domain slice; Gemma test RAG is a useful counterexample.
- Gemma has stronger base and external scores here, but the result is only a 50-row pilot and not a claim that Gemma is universally better.
- The safety-relevant fidelity heuristics are not a substitute for expert review.
- The main research contribution is the controlled comparison and the failure analysis, not a claim of beating a research group.

## Project demonstration

Use three short examples:

1. A normal medical sentence, showing base versus SFT.
2. A sentence with medicine name, dosage, number, and unit, showing the validation panel.
3. A warning/negation sentence such as Do not take Aspirin 10 mg., showing preserved Aspirin, 10 mg, and the German negation.

Then turn RAG on and show retrieved translation-memory examples, terminology used, provenance dataset/split/row IDs, validation results, and the difference between in-domain and EMEA behavior.

A good spoken conclusion is:

> I did not assume that retrieval or a newer model automatically improves translation. I made the comparison controlled, caught a real prompt failure, fixed it, retained the failed run, and measured both quality and preservation of details that matter in medical administration.

## Remaining research opportunities

- Reconstruct document-level EMEA groups using the retained XCES XML.
- Finish the 100k Qwen comparison and report confidence intervals for the 400-example runs.
- Add expert annotation for medical terminology, dosage normalization, and warning/negation categories.
- Test retrieval leakage and source/target near-duplicates more rigorously.
- Add constrained decoding or a structured post-editor for numbers and units.
- Compare terminology-aware reranking with the current embedding-only retrieval.
- Explore DPO or carefully graded RL only after the SFT/RAG baseline is stable.
- Measure energy, memory, latency distributions, and privacy controls for a healthcare administration.
- Add human-in-the-loop approval and audit trails before any operational use.

## Main artifacts

- README.md
- CLOUD_RUN.md
- artifacts/data_audit.json
- artifacts/download_manifest.json
- artifacts/gemma4_compatibility.json
- reports/experiment_summary.md
- reports/qwen3_test_comparison.*
- reports/qwen3_emea_comparison.*
- reports/gemma4_test_comparison.*
- reports/gemma4_emea_comparison.*
- artifacts/qwen3_50k_gated400_test_comparison.json
- artifacts/qwen3_50k_gated400_emea_comparison.json
- ui/app.py
- Dockerfile

No GCP deployment was performed because project, billing, credentials, and region were not provided. No files outside this project directory were modified.
