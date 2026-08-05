# MediLingo: The Simple Story Behind the Project

## Interview study guide for healthcare translation, SFT, LoRA, RAG, evaluation, GRPO, and distillation

Date of this snapshot: 5 August 2026

Project name shown to users: MediLingo

Technical project directory kept stable for reproducibility: `/home/sreenath/research-space/SovereignMedTranslate`

---

## How to use this guide

This guide is deliberately written as a story. The goal is not to make you memorize equations. The goal is to help you picture the system as a small translation office inside a hospital.

Whenever you feel lost, return to this picture:

> A hospital has an English-to-German translation desk. A trainee translator learns from many approved examples. A librarian brings similar approved examples and terminology when a new sentence arrives. A safety inspector checks medicine names, numbers, units, warnings, and words such as "not". A human reviewer remains responsible for the final answer.

The computer version of this office is MediLingo.

The most important sentence to remember for an project discussion is:

> I built a local English-to-German healthcare-information translation assistant. I first adapted small open models with supervised examples, then added a local translation memory and glossary. I evaluated base, fine-tuned, and retrieval-assisted versions on both an in-domain medical test set and a separate EMEA set, while checking details such as medicine names, numbers, dosage, units, warnings, and negation.

Do not describe MediLingo as a doctor, a clinical decision system, or a source of medical advice. It is an administrative translation prototype that requires human review.

---

# Part 1 - The whole story in one minute

Imagine that a Swiss hospital receives English medication information but needs to provide it in German.

A general translation model may understand everyday sentences, but healthcare text has details that are dangerous to change:

- `5 mg` must not become `50 mg`.
- `once daily` must not become `twice daily`.
- `Do not take...` must not lose the word `not`.
- `Arixtra` must not be replaced by a different product such as `Quixidar`.
- A warning must remain a warning.

So we built the project in layers.

1. We obtained English-German medical parallel data. Each row is an English sentence paired with its approved German translation.
2. We kept some data for learning, some for checking during training, and some for final testing.
3. We started with a pretrained Qwen3-4B model. It already knows general language, like a translator who knows English and German but has not yet studied this hospital's style.
4. We used supervised fine-tuning, or SFT, to show the model many correct medical translations.
5. We used LoRA, which adds a small set of trainable notes instead of rewriting the entire model.
6. We built a local translation memory and glossary. This is like giving the translator a librarian.
7. We added retrieval. Before translating, the system searches for similar approved examples and relevant terminology.
8. We added a safety gate. It rejects weak matches and evidence mentioning a conflicting medicine or entity.
9. We measured not only ChrF and BLEU, but also whether numbers, units, dosage, medicine names, warnings, and negation survived.
10. We built a Streamlit interface called MediLingo and prepared a Docker image for later deployment.

The 50k Qwen comparison on 400 examples produced these headline results:

| Dataset | Base ChrF | SFT ChrF | SFT + gated RAG ChrF | Base BLEU | SFT BLEU | SFT + gated RAG BLEU |
|---|---:|---:|---:|---:|---:|---:|
| Ahazeemi test | 55.30 | 62.64 | 64.88 | 29.90 | 38.58 | 44.06 |
| EMEA external | 60.07 | 68.61 | 73.73 | 30.60 | 43.14 | 52.10 |

[[PAGEBREAK]]

The detailed preservation and retrieval diagnostics for the 400-example 50k Qwen run are:

| Dataset | Condition | Numbers | Units | Dosage | Medicine | Negation | Warnings | RAG used | No evidence | Conflicts filtered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ahazeemi test | Base | 0.978 | 0.875 | 0.993 | 0.853 | 0.993 | 0.968 | 0% | 0% | 0 |
| Ahazeemi test | SFT | 0.930 | 0.883 | 0.985 | 0.858 | 0.995 | 0.970 | 0% | 0% | 0 |
| Ahazeemi test | SFT + gated RAG | 0.928 | 0.888 | 0.990 | 0.855 | 0.995 | 0.970 | 87.8% | 13.3% | 6,312 |
| EMEA external | Base | 0.993 | 0.833 | 1.000 | 0.695 | 0.995 | 0.973 | 0% | 0% | 0 |
| EMEA external | SFT | 0.953 | 0.865 | 0.983 | 0.690 | 0.995 | 0.973 | 0% | 0% | 0 |
| EMEA external | SFT + gated RAG | 0.948 | 0.863 | 0.990 | 0.685 | 0.988 | 0.973 | 98.5% | 1.8% | 3,288 |

These preservation values are rates, not clinical guarantees. For example, `0.978` means the heuristic check passed on 97.8% of the sampled cases for that property. The result still requires review.

Current run snapshot: at the latest document build, the separate Qwen 100k job was active at approximately 805 of 6,000 steps, with about four hours remaining. A project-local watcher is queued to evaluate the finished adapter on both 400-example datasets, build the scaling reports, restart the UI, and test the Docker image. The snapshot is intentionally dated because the run will continue after this PDF is printed.

The separate 100k Qwen training run is running as a controlled follow-up. We do not assume that more data must win. We measure it.

---

# Part 2 - Why healthcare translation is a useful problem

## The everyday problem

Suppose a hospital administrator receives an English medicine-information sentence:

> Do not take more than two tablets in 24 hours.

A useful German translation must preserve:

- the instruction is negative;
- the maximum quantity is two;
- the unit is tablets;
- the time window is 24 hours.

A fluent but incorrect translation can sound professional and still be unsafe. That is why a healthcare translation system needs more than a pleasant sentence.

## What MediLingo is and is not

MediLingo is intended for administrative healthcare information, such as:

- translating medicine-information text for staff review;
- helping prepare bilingual administrative documents;
- finding approved terminology used in previous translations;
- supporting multilingual document workflows;
- helping a human reviewer spot missing numbers or warnings.

MediLingo is not intended to:

- diagnose a patient;
- recommend treatment;
- choose a dose;
- replace a pharmacist, doctor, or professional translator;
- provide medical advice without review.

This distinction is important in an project discussion. The interesting research problem is reliable specialized language processing. The system is not being presented as clinically autonomous.

## Why local execution matters

The project runs on Thor, a machine with approximately 128 GB of memory and an NVIDIA Thor accelerator. Keeping the model and retrieval data local has practical benefits:

- sensitive text does not need to leave the local environment;
- experiments are repeatable without depending on a paid API;
- the team can measure memory, latency, and energy more directly;
- open model weights and code can be inspected;
- the system can be adapted to a public administration's infrastructure.

This is a data-control and deployment decision. It does not mean that a local model is automatically safe. Local execution still requires access controls, logging rules, human review, and careful data governance.

---

# Part 3 - The data story

## What is a dataset?

A dataset is a collection of examples used to teach or test a system. Think of it as a workbook.

For translation, one example usually has two sides:

- source: the English sentence;
- target: the German translation.

This is called parallel data because the two languages are aligned like two columns in the same workbook.

## The Ahazeemi medical corpus

The project uses `ahazeemi/opus-medical-en-de` for the initial English-German supervised training experiment.

The supplied split sizes were:

- train: 248,099 rows;
- dev: 2,000 rows;
- test: 2,000 rows.

The first pilot selected 50,000 training rows deterministically with seed 42. The expanded run prepared exactly 100,000 training rows in a separate file.

The complete supplied dev and test sets were preserved. The evaluation commands use deterministic samples of 400 rows for the expanded Qwen comparison so the run is much more informative than the original 50-row pilot while remaining practical on Thor.

## The EMEA corpus

EMEA is an English-German medical corpus from OPUS. It is kept separate from initial SFT training.

Why keep it separate? Imagine studying for an exam using one workbook and then testing yourself with a different workbook. If the same pages appear in both, the result can look better than the real ability.

The project stores EMEA as:

- external evaluation data;
- a separate reference or retrieval source;
- evidence for testing generalization beyond the Ahazeemi training source.

The downloaded OPUS material contains:

- 1,108,752 parsed aligned Moses sentence rows;
- retained XCES metadata with 1,163,348 links across 1,939 document groups.

The important limitation is that the parsed sentence rows do not expose a direct document ID that can be joined safely to the document metadata in this run. Therefore the present EMEA experiment is reported as sentence-level external evaluation. The XML metadata is retained for a future document-level reconstruction.

## Train, dev, and test: the three classrooms

Use this analogy:

- Train set: the exercises the apprentice studies.
- Dev set: the teacher's practice quiz used while adjusting the lesson.
- Test set: the final exam that should remain unseen until the end.

If test examples enter training or retrieval, it is like showing the final exam answers before the exam. The score is no longer trustworthy.

MediLingo therefore keeps test rows out of the training translation memory. EMEA is indexed separately from its held-out evaluation rows.

## Duplicate and near-duplicate checking

A duplicate is the same source sentence appearing more than once. A near-duplicate is almost the same sentence, perhaps with a small number or punctuation change.

Duplicates are dangerous because the model may appear to generalize when it has simply seen the answer already. They can also make a score look more stable than it really is.

The data preparation code:

- normalizes text for comparison;
- detects exact duplicates;
- records duplicate counts and data audit information;
- detects medical patterns for later validation;
- uses a deterministic selection policy.

The initial audit removed 435 exact duplicate training rows before selecting the 50,000-row pilot. The 100,000-row file and its audit are stored separately.

## What the preparation code extracts

The preparation step looks for patterns that matter in healthcare text:

- medicine-like names;
- numbers such as `5`, `7.5`, and `24`;
- dosage expressions such as `10 mg`;
- units such as `mg`, `ml`, `tablets`, and `hours`;
- warnings such as `do not`, `keep out of reach`, and `contraindicated`;
- negation such as `not`, `no`, and `without`.

This extraction is not a perfect medical named-entity recognizer. It is a conservative engineering check. Its purpose is to catch obvious changes that deserve human review.

## Provenance: the notebook attached to every example

Provenance means being able to answer:

- Where did this sentence come from?
- Which dataset was it in?
- Which split was it in?
- Which row or source file supplied it?
- Which retrieval index returned it?

A librarian who cannot tell you where a quote came from is not very useful. MediLingo keeps dataset names, split information, row IDs, hashes, download metadata, and checksums inside the project.

---

# Part 4 - The language-model basics

## What is NLP?

NLP means Natural Language Processing. It is the part of artificial intelligence that works with human language.

Examples include:

- translation;
- spelling correction;
- summarization;
- question answering;
- information extraction;
- sentiment analysis;
- speech and text interfaces.

MediLingo is an NLP translation system with retrieval and safety validation.

## What is a language model?

A language model is a system that has learned patterns in text. In simple terms, it repeatedly predicts what text should come next.

Imagine a reader who has read millions of pages. If you start a familiar sentence, the reader can guess likely continuations. A modern language model does this with numbers rather than human thoughts.

It does not automatically know that a translation is medically safe. It knows patterns. That is why we add specialized examples, evidence, and checks.

## What is a token?

A token is a small piece of text used internally by the model. A common word may be one token. A long word, number, punctuation mark, or unusual medicine name may be split into several pieces.

Analogy: instead of storing a whole sentence on one card, the model cuts it into small labeled cards. It reads and writes sequences of cards.

Tokens matter because:

- a maximum token limit controls how much text fits in one request;
- long documents can exceed the context window;
- unusual medicine names may be split awkwardly;
- output token limits affect how long a translation can be.

## What is a context window?

The context window is the amount of text the model can consider at one time. Think of the model's desk. If the desk holds only a few pages, a very long document must be split or summarized.

The Qwen training setup uses a maximum sequence length of approximately 1,024 tokens for the pilot. The UI uses a controlled maximum output length so it does not continue indefinitely.

## What is pretraining?

Pretraining is the large first education of a model. The model reads broad text and learns general language patterns before we receive it.

Qwen3-4B and Gemma 4 E2B are pretrained open models. They are not blank models. They already know general language, but they are not automatically experts in our exact healthcare translation style.

## What is a base model?

The base model is the original model before our medical SFT adapter is attached.

Analogy: it is a general translator who has not yet attended the hospital's terminology course.

The base condition is important because it tells us what improvement comes from training and retrieval. Without a base comparison, we cannot tell whether our work helped.

## What is inference?

Inference is using the trained model to produce an answer. Training is studying; inference is taking the exam or doing the real task.

The project measures inference latency because a translation that takes one second and one that takes one minute have different practical uses.

---

# Part 5 - Supervised fine-tuning and LoRA

## What is supervised fine-tuning, or SFT?

SFT stands for Supervised Fine-Tuning.

The teacher gives the apprentice a correct input and answer:

- English input: `The usual dose is 5 mg once daily.`
- Correct German target: the approved German translation.

The model is trained to make its output more like the target answer.

Analogy: a language student copies many answer sheets from a trusted teacher. Over time, the student learns the vocabulary, sentence patterns, and style.

SFT is easy to explain:

1. show the model an English source sentence;
2. show the correct German target;
3. compare what the model would write with the target;
4. adjust the trainable parameters so the model is more likely to produce the target next time;
5. repeat over many examples.

SFT is not reinforcement learning. The answer is supplied directly by the dataset instead of being discovered through a reward signal.

## Why use SFT first?

SFT gives a clear baseline:

- the training signal is easy to understand;
- the data has explicit reference translations;
- debugging is straightforward;
- improvements can be attributed to the examples;
- it is easier to explain in an project discussion.

That is why MediLingo does not jump immediately to GRPO. A clean SFT and RAG baseline is the foundation for any later RL experiment.

## Connection to CNN transfer learning

Your computer-vision experience transfers directly here.

In CNN transfer learning:

- a CNN has already learned general visual features such as edges and shapes;
- you keep most of the knowledge;
- you adapt the network to a new task such as classifying medical images;
- you compare the adapted model with the original model.

In language-model transfer learning:

- Qwen or Gemma has already learned general language patterns;
- you keep most of the language knowledge;
- you adapt it to English-German healthcare translation;
- you compare base, SFT, and SFT plus retrieval.

The objects are different, but the learning idea is familiar: start with a useful general representation and adapt it to a specialized task.

## What is PEFT?

PEFT means Parameter-Efficient Fine-Tuning.

Instead of changing every parameter in a four-billion-parameter model, PEFT changes only a small trainable part.

Analogy: do not rewrite the hospital's entire language handbook. Attach a thin medical-translation supplement to it.

Benefits:

- much less memory than full fine-tuning;
- faster training;
- smaller saved adapter files;
- the original base model can be reused;
- different adapters can be attached for different domains.

## What is LoRA?

LoRA stands for Low-Rank Adaptation. You do not need to explain the matrix mathematics in an project discussion unless asked.

The practical idea is:

> Keep the original model frozen and learn a small set of adjustment notes that steer it toward the new task.

Analogy: the general translator keeps the original handbook. We give the translator a stack of sticky notes saying:

- use this medical terminology;
- translate this type of sentence in this style;
- preserve dosage and warnings;
- prefer the examples from this domain.

The sticky notes are much smaller than a second complete handbook.

The Qwen and Gemma adapters use:

- LoRA rank 16;
- LoRA alpha 32;
- dropout 0.05;
- attention projections `q`, `k`, `v`, and `o`;
- MLP projections `gate`, `up`, and `down`;
- BF16 computation;
- seed 42.

## What is BF16?

BF16 means bfloat16, a compact numerical format used by modern accelerators.

Analogy: instead of writing every measurement with a very long decimal, use a shorter format that is still accurate enough for this training job. It reduces memory and speeds up computation.

BF16 is not the same thing as making the model smaller in knowledge. It is a way of storing and processing numbers during computation.

## What is a seed?

A seed starts the random-number generator at a known point.

Analogy: shuffle a deck using the same starting instruction. You do not get perfect sameness in every hardware environment, but you make experiments much easier to reproduce.

MediLingo uses seed 42 for deterministic selection and training settings.

## What is a checkpoint?

A checkpoint is a saved snapshot during training. If a long run stops, a checkpoint may allow recovery or inspection of an earlier stage.

The final adapter directory contains the learned LoRA changes and metadata. The 50k and 100k Qwen adapters are kept separate so one experiment cannot overwrite the other.

## Qwen3 first, Gemma second

Qwen3-4B was chosen first because it is small enough for local execution and provides a practical baseline.

Gemma 4 E2B was used as a controlled second-model comparison. We used the same general data, prompt style, and evaluation logic so the comparison is meaningful.

The Gemma run exposed a useful engineering lesson. A generic PEFT target search reached multimodal wrapper modules such as `Gemma4ClippableLinear`, and the first smoke test failed. The trainer was changed to select native `torch.nn.Linear` projections in the language tower. The one-step smoke test then passed, followed by the full pilot.

Interview lesson:

> A new open model is not plug-and-play. Model architecture details matter, especially when a model includes multimodal or wrapper modules.

## Why standard Transformers and PEFT instead of Unsloth?

Unsloth is an efficiency library and training toolkit. It can make some fine-tuning jobs faster or more memory efficient. It is not itself SFT, GRPO, RAG, or a model.

MediLingo uses standard Transformers, PEFT, and Accelerate because:

- the training logic is easier to inspect;
- the project does not depend on one optimization library;
- the Thor ARM/CUDA software stack should remain stable;
- the goal is a reproducible project discussion project rather than squeezing out the last speed improvement.

Unsloth could be tested later as an engineering comparison. It should not be described as the reason the model learned medical translation.

---

# Part 6 - RAG: giving the translator a librarian

## What does RAG mean?

RAG stands for Retrieval-Augmented Generation.

The phrase has two parts:

- Retrieval: search a local collection for useful evidence.
- Generation: ask the language model to write the final translation.

Analogy: a translator receives a new sentence. Before writing, the translator asks a librarian:

> Have we translated something similar before? Which approved German terms should I use?

The librarian brings a few examples. The translator uses them as reference, not as text to copy blindly.

## Why not simply fine-tune more?

Fine-tuning changes the model's general behavior. Retrieval gives it information at request time.

Analogy:

- Fine-tuning is sending the translator to a month-long course.
- Retrieval is giving the translator a searchable desk drawer during each job.

Fine-tuning is useful for style and domain behavior. Retrieval is useful for specific terminology, previous approved wording, and information that may change.

## The translation memory

The translation memory stores pairs such as:

- English source sentence;
- approved German target sentence;
- dataset name;
- split;
- row ID;
- source hash;
- provenance;
- detected entity tokens.

It is like a filing cabinet of previous approved translations.

## The medical glossary

The glossary stores:

- English source term;
- German target term;
- example context;
- source dataset;
- confidence or frequency information.

It is like a terminology card box. If the input contains a term that appears in the glossary, the UI can show the expected German term to the reviewer.

## What is an embedding?

An embedding turns text into a long list of numbers that represents its meaning and wording pattern.

You can avoid the mathematics with this analogy:

> Give every sentence a position on a giant map of meaning. Similar sentences are placed near each other. Different sentences are farther apart.

For example, two sentences about a medicine dose may be close even if the grammar differs slightly.

MediLingo uses a multilingual sentence-embedding model so English queries can search a multilingual index.

## What is cosine similarity?

Cosine similarity is a way of asking how closely two embedding directions point on the meaning map.

Everyday analogy: two people can walk in almost the same direction even if they start at different locations. A high similarity means the sentences point toward a similar topic or wording pattern.

MediLingo uses a NumPy similarity search rather than depending on FAISS. This keeps the index simple and avoids unnecessary compatibility problems on Thor.

## Retrieval steps in MediLingo

When a user enters an English healthcare sentence:

1. The system extracts conservative entity tokens and medicine-like names.
2. The query is embedded.
3. The local translation-memory vectors are searched.
4. Several more candidates than needed are collected. This is oversampling.
5. Candidates mentioning a conflicting product or entity can be removed.
6. Candidates below the similarity threshold can be removed.
7. The best surviving examples and direct glossary hits are placed into the prompt.
8. The model generates one German translation.
9. The verifier checks the output.
10. The UI shows evidence, provenance, gate decisions, and warnings.

## What is a similarity threshold?

The threshold is the minimum match quality required before an example is trusted as useful evidence.

Analogy: a librarian does not hand over a book merely because it shares one word with the question. The book must be similar enough.

The default threshold is 0.65. It is a practical control, not a universal truth. A higher threshold means fewer but stronger matches. A lower threshold means more evidence but more risk of irrelevant examples.

## What is entity filtering?

An entity is a specific thing mentioned in text, such as a medicine product, named organization, or other important identifier.

Entity filtering asks:

> Does this retrieved example talk about the same important product or does it mention a conflicting one?

For example, if the input is about Arixtra, a high-scoring example about Quixidar should not be treated as a safe translation template merely because both sentences discuss blood clots.

The gate keeps relevant matches and rejects obvious conflicts. It also avoids treating all-caps medical abbreviations such as VTE, DVT, and PE as different product names.

This is a small but meaningful safety improvement because a translation-memory system can otherwise import the wrong product name.

## What happens when no evidence is safe?

MediLingo does not force retrieval. If all candidates are too weak or conflicting, the model translates without a translation-memory example.

That is better than pretending that a bad match is good evidence.

The UI reports:

- evidence kept;
- evidence rejected;
- retrieval gate decision;
- similarity threshold;
- glossary hits;
- provenance.

## The prompt is an instruction sheet

The RAG prompt tells the model:

- what the source text is;
- that the target language is German;
- that retrieved material is reference-only;
- to output exactly one German translation;
- to preserve medicine names, numbers, dosage, units, warnings, and negation;
- not to copy the bilingual evidence section into the answer;
- not to reveal hidden reasoning or produce a chain of thought.

This matters because RAG can fail even when retrieval is correct. The model may copy the examples, mix languages, or continue writing a list of references.

In an earlier Gemma RAG run, the model sometimes translated correctly and then copied retrieved examples into the answer. The prompt and cleanup logic were corrected, and the failed run was preserved as a lesson.

## RAG is not a magic fact checker

RAG does not prove that an answer is correct. It only gives the model relevant local evidence.

A librarian can bring the wrong book. The translator can misunderstand the book. The source itself can contain an error.

That is why retrieval must be combined with:

- provenance;
- explicit source separation;
- entity filtering;
- similarity gating;
- output validation;
- human review.

---

# Part 7 - Evaluation: how we know whether it helped

## Why compare several conditions?

A single score is not enough. MediLingo compares:

- Base: original model without medical SFT;
- SFT: base model with the medical LoRA adapter;
- SFT + RAG: fine-tuned model with local evidence retrieval.

This is like testing a vehicle in three forms:

1. the original vehicle;
2. the vehicle after an engine tune-up;
3. the tuned vehicle with a navigation assistant.

If the third version is better, we can ask whether the improvement came from training, retrieval, or both.

## ChrF

ChrF is a translation metric based on character-level overlap between the model output and the reference translation.

Analogy: place the two sentences under a transparent sheet and count how many small letter sequences line up. It is often useful for morphologically rich languages because small word endings and character patterns matter.

The research group's legal-translation results use ChrF prominently, so keeping ChrF makes MediLingo easier to compare conceptually with that work.

ChrF is not a medical safety score. A translation can have a high overlap score and still change a critical number.

## BLEU

BLEU is another traditional translation metric based largely on matching word sequences with the reference.

Analogy: count how many short phrases from the reference appear in the model's translation, while penalizing awkwardly short output.

BLEU is useful as a secondary metric, but it can miss meaning changes and acceptable alternative wording.

## Medical fidelity checks

MediLingo adds direct checks for:

- number preservation;
- unit preservation;
- dosage preservation;
- medicine-name preservation;
- negation preservation;
- warning preservation.

These are conservative heuristics. If a check says `CHECK`, the result needs review. It does not mean the translation is definitely unsafe. If a check says `OK`, it does not certify safety.

## Latency

Latency is how long one translation takes.

A hospital workflow cares about both quality and response time. RAG normally adds some latency because it must embed the query and search the index.

The expanded 50k test run measured approximately:

- Base: 1.655 seconds per example;
- SFT: 2.015 seconds per example;
- gated RAG: 2.099 seconds per example.

The EMEA run measured approximately:

- Base: 1.636 seconds per example;
- SFT: 1.901 seconds per example;
- gated RAG: 2.033 seconds per example.

This is a useful result: the retrieval gate adds a relatively small amount of time in this prototype, but the actual trade-off must be measured on the deployment hardware and workload.

## The 400-example Qwen result in plain language

On the Ahazeemi test sample:

- base Qwen3 reached ChrF 55.30 and BLEU 29.90;
- SFT increased this to ChrF 62.64 and BLEU 38.58;
- gated RAG increased this further to ChrF 64.88 and BLEU 44.06.

On the separate EMEA sample:

- base Qwen3 reached ChrF 60.07 and BLEU 30.60;
- SFT increased this to ChrF 68.61 and BLEU 43.14;
- gated RAG increased this further to ChrF 73.73 and BLEU 52.10.

The EMEA result is especially useful because it is not the same training source. It suggests that the adapted model and local evidence can generalize to a different medical corpus, but it is still sentence-level evaluation and not expert clinical validation.

## Evidence usage in the 400-example run

On the Ahazeemi test sample, gated RAG used evidence for about 87.8% of inputs and filtered 6,312 conflicting candidates during retrieval.

On EMEA, gated RAG used evidence for about 98.5% of inputs and filtered 3,288 conflicting candidates.

The number of filtered candidates is not itself a quality score. It tells us that the gate was actively making decisions rather than being decorative.

## The historical 50-example pilot

The original 50-example pilot remains useful as a record of the first experiment. Its values were:

| Dataset | Model | Condition | ChrF | BLEU |
|---|---|---|---:|---:|
| Ahazeemi test | Qwen3 | Base | 55.89 | 31.61 |
| Ahazeemi test | Qwen3 | SFT | 60.09 | 33.75 |
| Ahazeemi test | Qwen3 | SFT + RAG | 60.54 | 38.43 |
| Ahazeemi test | Gemma 4 | Base | 58.88 | 34.71 |
| Ahazeemi test | Gemma 4 | SFT | 58.94 | 35.69 |
| Ahazeemi test | Gemma 4 | SFT + RAG | 56.32 | 35.80 |
| EMEA external | Qwen3 | Base | 57.10 | 34.51 |
| EMEA external | Qwen3 | SFT | 65.13 | 44.79 |
| EMEA external | Qwen3 | SFT + RAG | 66.37 | 47.41 |
| EMEA external | Gemma 4 | Base | 61.11 | 36.38 |
| EMEA external | Gemma 4 | SFT | 65.25 | 43.61 |
| EMEA external | Gemma 4 | SFT + RAG | 68.46 | 47.84 |

The Gemma in-domain RAG result is a useful counterexample. RAG did not automatically improve every slice. That is a stronger research narrative than claiming that retrieval always wins.

## What the score does not tell us

A metric does not tell us:

- whether a pharmacist approves the translation;
- whether a warning is culturally and legally appropriate;
- whether a product name was confused with another product;
- whether the source document was split at the right place;
- whether the data contained duplicates;
- whether a model leaked a memorized sentence;
- whether a real user understands the output.

Those questions require human evaluation, better annotations, and document-level tests.

---

# Part 8 - The MediLingo interface and deployment story

## The UI as a glass window into the system

The Streamlit UI makes hidden engineering decisions visible.

The user can select:

- Qwen3-4B base;
- Qwen3-4B SFT 50k;
- Qwen3-4B SFT 100k after that adapter is available;
- Gemma 4 E2B base;
- Gemma 4 E2B SFT;
- 50k or 100k translation memory;
- RAG on or off;
- entity conflict filtering on or off;
- minimum similarity threshold;
- number of retrieved examples;
- number of retrieved terminology items;
- maximum output tokens.

The interface displays:

- German translation;
- numbers and units check;
- dosage check;
- medicine-name check;
- negation check;
- warning count;
- retrieved examples;
- glossary terms;
- source dataset and provenance;
- evidence gate decision;
- human-review warnings.

A good UI for an AI research system should not hide uncertainty. It should show where the answer came from and what still needs checking.

## Why Streamlit?

Streamlit is a quick way to turn Python code into a local interactive demo. It is not the final production architecture, but it is ideal for an project discussion because the evaluator can type a sentence and see the entire pipeline.

## Why Docker?

Docker packages the application and its runtime dependencies into a reproducible container.

Analogy: instead of handing someone a kitchen recipe and hoping they have the same ingredients, pack the recipe, ingredients, and cooking tools in one labeled box.

The project prepares a local image named `medilingo:local`.

The Docker image contains:

- the Streamlit application;
- runtime scripts;
- the adapters;
- local retrieval indexes;
- the health endpoint.

It does not contain the full base model weights. A real deployment must decide where model weights live, how they are cached, and whether the serving platform has enough memory and accelerator support.

## Why Cloud Run is only prepared, not deployed

A real Google Cloud deployment needs:

- a GCP project;
- billing;
- a region;
- credentials;
- an Artifact Registry repository;
- a decision about CPU or GPU serving;
- authentication and audit policies;
- data retention and privacy rules.

Those details were not supplied, so the project prepares a Docker and Cloud Run handoff without deploying or spending money.

## Health checks

The local UI health endpoint is:

`http://127.0.0.1:8511/_stcore/health`

The local tunnel exposes it at:

`http://127.0.0.1:18511/_stcore/health`

The browser UI is available at:

`http://127.0.0.1:18511/`

Port 8501 belongs to an unrelated process and is deliberately not touched.

---

# Part 9 - What the legal-translation research group is doing conceptually

## The shared skeleton

The research group and MediLingo are not identical projects, but they share a useful skeleton:

1. choose a specialized translation domain;
2. use a relatively small open model rather than only a giant closed API;
3. adapt the model to domain data;
4. measure translation quality with metrics such as ChrF;
5. investigate methods that can improve specialized translation;
6. care about practical deployment and high-stakes errors.

The group works on legal translation, especially Swiss legal language and multilingual settings. MediLingo works on English-German healthcare-information translation.

## The important difference

The group's research question is about whether reinforcement learning, especially GRPO-style training, can outperform ordinary SFT for specialized legal machine translation.

MediLingo's first question is more controlled and easier to explain:

> What happens when a small medical translation model is trained with SFT and then given local retrieval evidence with explicit healthcare fidelity checks?

The projects can learn from each other without being copies.

- Their work can teach us how to design reward signals and compare SFT with RL.
- MediLingo can contribute ideas about terminology provenance, entity filtering, safety checks, data separation, and administrative deployment.
- A future combined study could use a healthcare corpus and compare SFT, RAG, DPO, and GRPO under the same preservation-focused evaluation.

## What DeepSeek R1 means in this story

DeepSeek R1 can play two different roles, and they should not be confused.

First, it can appear as a strong reasoning-model baseline in a comparison plot. That means people simply ask it to translate and measure the result.

Second, a strong reasoning model can act like a senior reviewer or teacher. It may help:

- judge candidate translations;
- explain why one candidate is better;
- create preference labels;
- generate demonstrations for a smaller student;
- provide a reward signal for an RL experiment.

A safe project discussion sentence is:

> A model such as DeepSeek R1 can be either a baseline translator or a teacher/judge in a training pipeline. I would check the exact paper code before claiming which role it played in a particular experiment.

Do not say that DeepSeek R1 automatically trained Qwen unless the paper or code proves that exact connection.

## Why ChrF appears in both stories

ChrF is useful for comparing specialized translation systems because it measures character-level overlap and can reflect morphology and word endings. Using ChrF in MediLingo makes the healthcare project speak the same evaluation language as the legal-translation work.

But the safety checks go beyond ChrF. A high ChrF score does not guarantee that a dose or warning is correct.

## Why this is not a copycat project

A copycat would reproduce the same legal data, model, reward, and claim with only a renamed title.

MediLingo is different because:

- the domain is healthcare information, not Swiss legal translation;
- the use case is administrative translation support;
- the initial method is SFT plus local RAG, not GRPO;
- EMEA is held out as external reference/evaluation data;
- retrieval has product/entity conflict filtering;
- evaluation includes medical fidelity checks;
- the UI exposes provenance and human-review warnings;
- the project prioritizes a reproducible baseline before advanced RL.

The right research message is not "I beat their professors." It is:

> I understood the common research pattern, chose a neighboring problem, built a controlled baseline, and identified the next experiment that could connect our work to theirs.

---

# Part 10 - GRPO explained as a story

## What reinforcement learning means here

In supervised learning, the teacher supplies the correct answer.

In reinforcement learning, the system tries an action and receives a score or reward. It gradually becomes more likely to choose actions that receive better rewards.

Analogy: an apprentice translator writes several candidate translations. A reviewer scores them. The apprentice learns to prefer the kinds of answers the reviewer likes.

## What GRPO stands for

GRPO stands for Group Relative Policy Optimization.

The name sounds intimidating, but the story is simple:

- Group: make several candidate answers for the same input.
- Relative: compare the candidates with one another.
- Policy: the model's behavior for choosing or generating answers.
- Optimization: adjust the model so better candidates become more likely.

Imagine a coach who asks a player to perform the same move five times. The coach may not know a perfect universal score, but can still say which attempts were better than the others. The player practices toward the better attempts.

## A GRPO translation round

For one English medical sentence:

1. The model generates a small group of German candidates.
2. A reward system scores each candidate.
3. The reward may consider translation quality, terminology, number preservation, warning preservation, and perhaps a judge-model score.
4. The candidates are ranked relative to the same input.
5. The better candidates receive positive pressure; weaker candidates receive less pressure.
6. The model is updated and tries again.

## What could the reward contain?

A healthcare translation reward could combine:

- ChrF or another translation score when a reference exists;
- terminology accuracy;
- medicine-name preservation;
- number and unit preservation;
- dosage preservation;
- warning and negation preservation;
- a human or LLM judge;
- a penalty for hallucinated extra content;
- a penalty for copying the retrieval evidence instead of translating.

The hard part is not writing `GRPO`. The hard part is deciding what "good" means and making sure the reward does not encourage a dangerous shortcut.

## SFT, RAG, GRPO, DPO, and distillation side by side

| Method | What it changes | Easy analogy | Main question |
|---|---|---|---|
| SFT | Model behavior from correct examples | Apprentice copies approved answer sheets | Can the model learn the domain style? |
| LoRA | Small trainable adapter rather than whole model | Sticky notes on the handbook | Can we adapt cheaply? |
| RAG | Information supplied at request time | Librarian brings relevant pages | Can the model use current approved evidence? |
| DPO | Preference between better and worse answers | Teacher circles the better answer | What answer style do experts prefer? |
| GRPO | Relative reward among several new candidates | Coach compares several attempts | Can the model discover better answers from rewards? |
| Distillation | Smaller model learns from stronger teacher | Senior interpreter coaches junior interpreter | Can we keep quality while reducing cost? |

## What can go wrong with GRPO?

A model may learn to game the reward:

- copy words that earn overlap points but damage meaning;
- preserve numbers while mistranslating the sentence;
- produce a long answer containing many expected terms;
- satisfy an LLM judge with confident wording;
- exploit a weakness in the verifier;
- overfit to the reward examples.

Analogy: if a school awards points only for using long words, students may fill answers with long words instead of answering the question.

## Why we are not doing GRPO now

GRPO would make this project much larger:

- we would need a carefully designed reward function;
- we would need multiple candidates per input;
- we would need to validate a judge or reward model;
- training would be more expensive and less predictable;
- there would be many more knobs to explain in an project discussion;
- we would need stronger human evaluation to trust the result.

The sensible sequence is:

1. get the data split right;
2. build a strong SFT baseline;
3. measure RAG and fidelity checks;
4. collect failure cases;
5. only then design a small, well-motivated GRPO experiment.

Interview answer:

> GRPO is relevant to the group's research, but I intentionally did not add it before stabilizing the healthcare SFT and RAG baseline. Otherwise I would not know whether an improvement came from better data, retrieval, prompt changes, or reinforcement learning.

---

# Part 11 - Distillation explained as a story

## What knowledge distillation means

Distillation means using a larger or stronger teacher model to train a smaller student model.

Analogy: a senior interpreter works beside a junior interpreter. The senior produces high-quality translations and explains preferences. The junior studies those examples until it can work more cheaply and locally.

Possible teachers include:

- a larger language model;
- a strong reasoning model such as a teacher or judge;
- an ensemble of several models;
- a human expert.

The student could be Qwen3-4B or Gemma 4 E2B.

## Response distillation

The teacher generates a translation for each source sentence. The student is trained on those teacher outputs.

Simple benefit: we can obtain more training labels or stylistically consistent examples.

Risk: if the teacher makes an error, the student learns it. Teacher errors can become systematic student errors.

## Preference distillation

The teacher generates several candidates and ranks them. The student learns which kind of answer is preferred.

This connects to DPO or GRPO. The student learns preferences rather than only one reference translation.

## Logit distillation

A teacher can provide its probability preferences over possible next tokens, not only the final sentence. This contains richer information but is more technically involved and usually requires compatible access to teacher outputs.

Simple analogy: instead of telling the junior only the final sentence, the senior also says, "At this word I was 80% sure this phrasing was better than the alternatives."

## Process or reasoning distillation

A teacher may produce an explanation or intermediate reasoning. The student can learn a structured procedure, but hidden reasoning should not automatically be copied into a user-facing medical translation.

For MediLingo, the output should remain one German translation plus visible validation information, not a private chain of thought.

## Why distillation is not in the current run

Distillation would add:

- teacher inference cost;
- teacher-data generation;
- quality filtering;
- possible licensing and privacy questions;
- a second training pipeline;
- another source of confounding variables.

It could be a good future project, especially if the goal is to make a local model smaller or faster. It is not necessary to answer the first research question: can a small open model learn useful specialized translation with SFT and local evidence?

Interview answer:

> Distillation is a sensible future efficiency experiment, but I would first establish that the 100k SFT and RAG baseline is reliable. Then I could use a stronger teacher to create carefully filtered preference or translation examples and test whether the smaller model preserves quality at lower cost.

---

# Part 12 - What could be improved next

## Improvement 1: finish the data-scaling experiment

The current 100k Qwen run tests whether doubling the training subset from 50k to 100k helps.

The fair comparison keeps fixed:

- base model;
- prompt structure;
- evaluation rows;
- seed policy;
- adapter method;
- RAG threshold;
- retrieval logic.

Only the training data size and matching translation-memory index change. This is an ablation: change one important thing and observe the result.

More data can help, but it can also introduce noise, duplicates, inconsistent translations, or domain imbalance. The result must be measured.

## Improvement 2: document-level EMEA reconstruction

A sentence can be translated correctly while losing context from the paragraph around it. Legal and medical documents often contain references such as "this product", "the above dose", or "it".

The retained EMEA XML has document groups. A future data-engineering step could safely reconstruct document boundaries, then split train/dev/test by document rather than by sentence.

Why this matters: if sentences from the same document appear in both training and test, the model may see the surrounding wording and obtain an unfair advantage.

## Improvement 3: human medical terminology annotation

The current checks are automated heuristics. A better study would ask qualified reviewers to annotate:

- whether the medicine name is correct;
- whether the dosage is preserved;
- whether the warning remains a warning;
- whether the German is natural and appropriate;
- whether the translation changes the meaning.

Human evaluation is slower but much closer to the real use case.

## Improvement 4: confidence intervals and statistical testing

A score from 400 examples is better than 50, but it is still a sample. A future report should show uncertainty, such as bootstrap confidence intervals, and test whether the difference between systems is likely to be meaningful.

You can explain this without mathematics:

> If I shuffled which 400 examples I selected, would the conclusion remain the same?

## Improvement 5: better retrieval ranking

The current system uses embedding similarity plus an entity gate. Future options include:

- a terminology-aware reranker;
- a cross-encoder that reads query and candidate together;
- exact medicine-name matching before semantic similarity;
- separate indexes for product names, dosage patterns, warnings, and general sentences;
- document-level retrieval instead of sentence-only retrieval.

## Improvement 6: constrained output or a post-editor

A structured post-editor could check whether every source number and unit appears in the German output. It could flag a failure or ask the model to revise.

A constrained decoder could prevent certain tokens or force required entities, but constrained generation can also make German awkward. It should be tested with human reviewers.

## Improvement 7: long-context document translation

The current pilot is sentence-oriented. A real administrative workflow may need paragraphs, tables, headings, footnotes, and references.

A document pipeline could:

1. parse the document structure;
2. preserve headings and table cells;
3. translate each segment with surrounding context;
4. run cross-segment terminology consistency checks;
5. reconstruct the document;
6. show a diff for human approval.

## Improvement 8: human preference data

Instead of asking an automatic metric to decide everything, collect pairs:

- translation A;
- translation B;
- expert preference;
- reason for preference.

This supports DPO or GRPO later. It also creates a research dataset about what experts actually value.

## Improvement 9: energy, memory, and privacy

For local healthcare use, measure:

- peak memory;
- accelerator utilization;
- time per sentence;
- energy per 1,000 translations;
- cold-start time;
- cache size;
- what logs retain.

A model that is one ChrF point better but requires ten times the resources may not be the right public-administration choice.

## Improvement 10: calibration and abstention

The system should learn when not to pretend.

A useful future behavior is:

> I found no sufficiently similar approved example and the output failed a medicine-name check. Please route this to human review.

This is often more valuable than forcing a confident answer.

---

# Part 13 - Other approaches the group or we could have used

## Train a larger model fully

This could produce more capacity, but it would require much more memory, time, and infrastructure. It is harder to reproduce and harder to deploy locally.

## Use only a general prompt

A prompt-only system is quick, but it may not consistently learn domain terminology or preserve the desired style.

## Use only a glossary

A glossary can protect terms but cannot translate grammar and sentence structure. It is a guardrail, not a complete translator.

## Use only RAG with a base model

This might be a useful baseline. It tests whether local evidence helps even without SFT. MediLingo prioritizes the SFT and SFT + RAG comparison, but a full factorial experiment could also include base + RAG.

## Use DPO

DPO means Direct Preference Optimization. It learns from preferred and rejected answers without running a full online RL loop.

Analogy: give the student pairs of answer sheets and say which one the teacher prefers. The student learns the preference directly.

DPO may be easier to stabilize than GRPO, but it still needs trustworthy preference pairs.

## Use GRPO

GRPO can let the model generate several candidates and learn from relative rewards. It matches the research group's research direction more closely, but it needs a carefully validated reward and more experiment control.

## Use a large teacher for distillation

This could create a smaller, faster model. It is attractive for local deployment but creates teacher cost and teacher-error risks.

## Use constrained decoding

This could force preservation of numbers, units, or approved names. It may improve fidelity but can produce unnatural output if the constraints are too rigid.

## Use a translation-specific encoder-decoder model

A dedicated machine-translation architecture may be more efficient for translation than a general chat language model. It could be a strong comparison, especially if low latency and predictable output matter more than open-ended instruction following.

## Use an ensemble or reranker

Several models could produce candidates, and a separate evaluator could choose one. This may improve quality but increases latency, memory, and explanation complexity.

The reason to begin with MediLingo's simpler pipeline is not that alternatives are bad. It is that a research project should first make its causal story understandable.

---

# Part 14 - What is meaningful in a real healthcare workflow?

A realistic first use case is an administrative translation assistant used by a trained staff member.

Example workflow:

1. Staff member pastes an English medicine-information paragraph.
2. MediLingo retrieves similar approved translations and terminology.
3. The model proposes German text.
4. The verifier flags numbers, units, dosage, warnings, or medicine-name problems.
5. The interface shows evidence and provenance.
6. A qualified human accepts, edits, or rejects the translation.
7. The approved result can later become part of a carefully governed translation memory.

This creates a useful feedback loop without pretending that the model is autonomous.

## Could it lead to a paper?

Yes, but the paper should make a modest, testable claim.

A possible paper question is:

> Does entity-aware, similarity-gated retrieval improve English-German medical-information translation from small locally runnable models, and does the improvement hold on an external EMEA corpus?

A credible study would need:

- clear train/dev/test and retrieval splits;
- leakage checks;
- base, SFT, and SFT + RAG ablations;
- larger evaluation sets;
- confidence intervals;
- medical expert evaluation;
- terminology and safety-oriented metrics;
- failure examples;
- latency and memory measurements;
- reproducible code and data provenance.

The paper should not claim clinical safety from automatic scores alone.

---

# Part 15 - The current group connection and research communication strategy

## What to say about your project

Start with the problem, not the tool names:

> I wanted to understand how a small open model could support reliable multilingual healthcare administration locally. I treated translation as more than fluent text generation because medicine names, doses, warnings, and negation must survive.

Then tell the story:

> I created a clean English-German medical training split, preserved EMEA as an external evaluation source, adapted Qwen3 with LoRA/SFT, added a local translation memory and glossary, and built an entity and similarity gate so unrelated evidence is not injected into the prompt. I measured ChrF, BLEU, preservation checks, retrieval usage, and latency, then exposed the whole process in a local UI.

Then connect to the group:

> I see a clear connection to your legal-translation work: both projects study domain-specific multilingual generation with open models and careful evaluation. My first method is deliberately simpler than GRPO, but the same evaluation and failure-analysis framework could support a future reward-based experiment in healthcare or legal translation.

## If they ask why you did not use GRPO

Say:

> GRPO is relevant, but I did not want to add an online reward-learning loop before stabilizing the data, SFT baseline, retrieval split, and preservation checks. Otherwise I would not know what caused an improvement. My next step would be to collect expert preferences and design a reward that penalizes medicine-name, dosage, warning, and negation errors.

## If they ask why you did not use distillation

Say:

> Distillation is attractive if the goal is a smaller or faster model. I would first use a stronger teacher to generate or rank candidate translations, filter those outputs carefully, and then train the student. I kept it out of the first experiment because teacher errors and extra generated data would add another confounding factor.

## If they ask why RAG helps

Say:

> SFT changes the model's general behavior. RAG gives the model approved local evidence at request time. It is like training a translator and also giving them a searchable terminology desk. The combination can help, but retrieval is only useful when the evidence is similar, non-conflicting, and traceable.

## If they ask what can go wrong

Say:

> A fluent model can still change a dose, lose a negation, confuse two medicines, copy the retrieved example, or look good on a metric while being wrong. That is why I preserve provenance, keep test evidence out of the index, filter entities, gate similarity, run fidelity checks, measure external data, and require human review.

## If they ask how your CNN experience helps

Say:

> The transfer-learning principle is the same. A pretrained model already contains general knowledge. I adapt a small trainable part to a specialized task, compare against the frozen or base model, and validate on data that was not used for fitting. LoRA is the language-model equivalent of adapting a small task-specific part instead of retraining everything.

## If they ask whether Gemma is better than Qwen

Say:

> The answer depends on the dataset and condition. Gemma had a stronger base score in the original 50-example pilot, while Qwen improved strongly with SFT and gated RAG in the expanded comparison. I would not declare a universal winner from a small sample. I would compare the same data, prompt, evaluation rows, latency, memory, and human judgments.

## If they ask whether you beat the research group's result

Say:

> That was not my claim. I built a neighboring healthcare experiment to understand the same class of problem. The value is the controlled baseline, the explicit error checks, and the clear next step toward preference or RL training. I would rather present a defensible experiment than claim superiority from incomparable domains and datasets.

---

# Part 16 - Your earlier healthcare NLP work and how it fits

The earlier medical-prescription administration and information project is useful in the project discussion because it shows that your interest in healthcare NLP did not begin with this one experiment.

A safe way to connect the projects is:

- the earlier project focused on applying NLP to healthcare prescription or medical-information workflows;
- MediLingo focuses on multilingual generation, supervised adaptation, retrieval, and validation;
- both are administrative or information-support systems, not autonomous clinical decision systems;
- the earlier work gives you a workflow and healthcare-user perspective;
- MediLingo gives you a language-model training and evaluation perspective.

Do not invent technical details about the earlier project if you cannot explain the exact code or dataset. Say what you personally implemented, what the input and output were, and what limitation you found.

A good project discussion bridge is:

> My earlier healthcare NLP work made me interested in extracting and organizing medically relevant information. MediLingo extends that interest toward multilingual generation: the system translates healthcare information, retrieves approved terminology, and checks whether safety-critical details survived. The common theme is using NLP to support administrative healthcare work while keeping humans in control.

This is a meaningful connection to the research position because the position asks for both strong machine-learning interest and the ability to speak with people outside computer science. You can explain the technical method while also discussing how a hospital administrator would use the result.

# Part 16 - A compact glossary in everyday language

| Term | Plain meaning | Everyday analogy |
|---|---|---|
| AI | Computer system doing a task that appears intelligent | A tool that can perform a task requiring judgment |
| ML | Learning patterns from examples | Learning from past practice sheets |
| NLP | Computing with human language | A language office inside a computer |
| Language model | Model that predicts likely text | A very well-read autocomplete |
| Token | Small text piece used internally | A puzzle piece or word card |
| Tokenizer | Breaks text into tokens | Cuts a sentence into cards |
| Context window | Text the model can see at once | The size of its desk |
| Inference | Using a trained model | Taking the exam or doing the job |
| Corpus | Large organized text collection | A library or workbook collection |
| Parallel corpus | Paired source and target translations | Two aligned columns in a workbook |
| Train split | Examples used for learning | Practice exercises |
| Dev split | Examples used while adjusting | Teacher's practice quiz |
| Test split | Held-out final evaluation | Final exam |
| Leakage | Test information accidentally enters learning | Seeing the exam answers early |
| Base model | Original pretrained model | General translator before specialization |
| SFT | Learning from correct input-output examples | Apprentice copies approved answer sheets |
| PEFT | Fine-tuning only a small part | Add a supplement instead of rewriting the handbook |
| LoRA | Small trainable steering adapter | Sticky notes on the handbook |
| Adapter | Saved task-specific changes | A detachable training supplement |
| BF16 | Compact number format for computation | Shorter measurements that save space |
| Seed | Controlled random starting point | Shuffling a deck from the same setup |
| Checkpoint | Saved training snapshot | A saved game position |
| RAG | Retrieve evidence before generating | Translator plus librarian |
| Retrieval | Searching local evidence | Finding the right card from a filing cabinet |
| Generation | Writing the final answer | Translator produces the sentence |
| Embedding | Numeric representation of meaning | Pinning a sentence on a meaning map |
| Similarity | How close two sentence representations are | How close two places are on the map |
| Threshold | Minimum acceptable similarity | Librarian's quality cutoff |
| Translation memory | Previous approved translation pairs | Filing cabinet of past work |
| Glossary | Approved term pairs | Terminology card box |
| Entity | Specific named thing | A person's name, product, or organization |
| Entity filter | Rejects conflicting named things | ID checker at the door |
| Provenance | Where evidence came from | Citation card attached to a book |
| Prompt | Instructions sent to the model | Task sheet given to the translator |
| Grounding | Tying the answer to evidence | Keeping the translator's feet on the approved documents |
| Hallucination | Confident unsupported content | Inventing a fact while sounding certain |
| ChrF | Character-overlap translation metric | Transparent sheet counting matching letter patterns |
| BLEU | Word-phrase overlap translation metric | Counting matching short phrases |
| Fidelity check | Direct safety-oriented comparison | Inspector checks every label and number |
| Latency | Time for one result | Waiting time at the service desk |
| External evaluation | Test on different source data | Exam from another school |
| Document-level split | Split whole documents, not isolated sentences | Keep every page from one book in one classroom |
| RAG gate | Decision whether evidence is safe enough | Librarian decides whether to hand over the book |
| Reward | Score used in reinforcement learning | Coach's points |
| Policy | Model behavior being changed | The player's strategy |
| Candidate | One possible model answer | One draft translation |
| GRPO | Group-relative reinforcement training | Coach compares several attempts at once |
| DPO | Learn from preferred versus rejected answers | Teacher points to the better answer sheet |
| Distillation | Strong teacher trains smaller student | Senior interpreter coaches junior interpreter |
| Teacher model | Strong model producing guidance | Experienced mentor |
| Student model | Smaller model learning the guidance | Junior worker |
| Reranker | Reorders retrieved candidates | Librarian rechecks the shortlist |
| Constrained decoding | Limit output to protect requirements | Form with fields that cannot be left blank |
| Human in the loop | Person reviews important output | Supervisor signs the document |
| Docker | Packages application and dependencies | Labeled box containing the recipe and tools |
| Streamlit | Quick Python web interface | Glass window showing the machinery |
| Cloud Run | Managed container deployment | Cloud host that runs the boxed application |
| Reproducibility | Another person can repeat the experiment | Someone else can follow the same recipe |
| Ablation | Change one component to see its effect | Remove one ingredient and taste the difference |
| Overfitting | Memorize practice instead of learning | Student memorizes the workbook but fails a new question |
| Generalization | Work well on new examples | Skill transfers to a different workbook |

---

# Part 17 - The final mental picture

Keep this story in your head:

A hospital translation desk receives an English sentence.

The trainee translator is Qwen or Gemma. The trainee already knows general language because the model was pretrained.

SFT is the medical course. It shows the trainee many correct English-German examples.

LoRA is the small folder of sticky notes attached to the general handbook. It changes the medical style without rewriting the whole handbook.

RAG is the librarian. The librarian searches the local translation memory and glossary for approved evidence.

The similarity threshold is the librarian's judgment about whether a book is close enough to the question.

Entity filtering is the ID check. It stops an Arixtra question from receiving a Quixidar example simply because both are about blood clots.

The prompt is the task sheet. It tells the translator to write exactly one German translation and preserve important details.

The verifier is the safety inspector. It checks medicine names, numbers, dosage, units, warnings, and negation.

ChrF and BLEU are the exam scores. They help compare systems, but they are not the doctor's approval.

The UI is the glass window. It lets a human see the translation, evidence, provenance, gate decision, and warnings.

GRPO is the coach who asks for several drafts and rewards the better ones relative to the group.

Distillation is the senior interpreter teaching the junior interpreter how to work faster and locally.

A responsible research project does not start by claiming the coach or senior interpreter is necessary. It first checks that the trainee, lessons, librarian, inspector, and exam are all working correctly.

That is the reason for MediLingo's order:

> clean data -> controlled SFT -> local evidence -> safety checks -> broader evaluation -> only then GRPO or distillation.

---

# Appendix A - Project map and important files

Everything for this project is inside the stable technical directory `/home/sreenath/research-space/SovereignMedTranslate`.

Important locations:

- `data/raw/ahazeemi_processed/`: raw Ahazeemi source and metadata;
- `data/raw/emea_opus/`: OPUS EMEA archive and source information;
- `data/processed/ahazeemi_train.jsonl`: 50k training selection;
- `data/processed/ahazeemi_train_100k.jsonl`: 100k training selection;
- `artifacts/data_audit.json`: initial data audit;
- `artifacts/data_audit_100k.json`: expanded data audit;
- `artifacts/rag/`: 50k translation-memory and EMEA indexes;
- `artifacts/rag_100k/`: expanded 100k translation-memory and separate EMEA indexes;
- `models/qwen3-4b-medical-lora/`: original 50k Qwen adapter;
- `models/qwen3-4b-medical-lora-100k/`: expanded Qwen adapter after the run completes;
- `models/gemma4-e2b-medical-lora/`: Gemma pilot adapter;
- `scripts/prepare_data.py`: data preparation and auditing;
- `scripts/train_sft.py`: Qwen and Gemma LoRA/SFT training;
- `scripts/build_index.py`: translation memory and glossary index building;
- `scripts/runtime.py`: model loading, retrieval, prompting, and validation;
- `scripts/evaluate.py`: reproducible base/SFT/RAG evaluation;
- `scripts/compare_scaling.py`: 50k versus 100k comparison;
- `scripts/post_100k_pipeline.sh`: continuation watcher for evaluation, UI, and Docker;
- `ui/app.py`: Streamlit MediLingo interface;
- `README.md`: technical project overview;
- `CLOUD_RUN.md`: later deployment notes;
- `reports/experiment_summary.md`: experiment report;
- `artifacts/qwen3_50k_gated400_test_comparison.json`: expanded 50k test result;
- `artifacts/qwen3_50k_gated400_emea_comparison.json`: expanded 50k EMEA result.

The 100k run writes separate artifacts and does not overwrite the 50k adapter or reports.

---

# Appendix B - Questions to ask yourself before the project discussion

1. What problem does the system solve?
2. Why is a fluent translation not enough in healthcare?
3. What is the difference between SFT and GRPO?
4. Why did you use LoRA instead of full fine-tuning?
5. Why is RAG useful if the model was already fine-tuned?
6. How do you prevent irrelevant or conflicting retrieval evidence?
7. Why is EMEA not mixed into initial training?
8. Why are document IDs important?
9. What do ChrF and BLEU tell you, and what do they miss?
10. What does your verifier check?
11. Why did Gemma require a PEFT compatibility fix?
12. Why did you avoid GRPO and distillation for the first project?
13. What would you do next with expert preferences?
14. How could this connect to legal translation research?
15. What is the failure mode you are most worried about?
16. How would you deploy this while protecting sensitive text?
17. What would make the experiment publishable?
18. How do you know the model did not simply memorize the test set?
19. What would you change if 100k performs worse than 50k?
20. What would you never claim from this prototype?

A strong answer to the final question is:

> I would never claim clinical safety, legal correctness, or replacement of a human expert from these automatic experiments. I would claim that I built a reproducible local research prototype, measured several failure-sensitive properties, and identified a clear path toward expert evaluation and reward-based training.


---

# Appendix C - References discussed while designing the project

These are the main public resources connected to the project and project preparation:

- Ahazeemi English-German medical data: https://huggingface.co/datasets/ahazeemi/opus-medical-en-de
- OPUS EMEA corpus: https://opus.nlpl.eu/datasets/EMEA
- EMEA Hugging Face fallback/repackage: https://huggingface.co/datasets/qanastek/EMEA-V3
- Swiss legal translation model/repository: https://huggingface.co/iCoSys-HEIA/Qwen3.5-4b-Swiss-Legal-Translation
- Unsloth efficiency toolkit: https://github.com/unslothai/unsloth
- Legal-translation paper discussed in the project preparation: https://arxiv.org/pdf/2607.19226v1
- Earlier healthcare NLP project: https://www.omdena.com/chapter-challenges/leveraging-nlp-in-medical-prescription-administration-and-information

Use the paper and repository links to verify exact experimental details before making a precise claim about a particular reward model, judge model, or training stage. The conceptual explanations in this guide are designed to make the ideas understandable and evidence-safe; the source paper remains the authority for exact implementation details.
