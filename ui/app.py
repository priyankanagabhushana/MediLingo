from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runtime import TranslationRuntime  # noqa: E402


APP_NAME = "MediLingo"
QWEN_MODEL_ID = (
    os.environ.get("MEDILINGO_BASE_MODEL_DIR", "Qwen/Qwen3-4B").strip()
    or "Qwen/Qwen3-4B"
)
CLOUD_MODE = os.environ.get("MEDILINGO_CLOUD", "").lower() in {"1", "true", "yes"}

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🩺",
    layout="wide",
)

MODEL_CONFIG = {
    "Qwen3-4B base": {
        "model_id": QWEN_MODEL_ID,
        "adapter": None,
    },
    "Qwen3-4B SFT · 50k": {
        "model_id": QWEN_MODEL_ID,
        "adapter": PROJECT_ROOT / "models/qwen3-4b-medical-lora",
    },
    "Qwen3-4B SFT · 100k": {
        "model_id": QWEN_MODEL_ID,
        "adapter": PROJECT_ROOT / "models/qwen3-4b-medical-lora-100k",
    },
    "Gemma 4 E2B base": {
        "model_id": "google/gemma-4-E2B-it",
        "adapter": None,
    },
    "Gemma 4 E2B SFT": {
        "model_id": "google/gemma-4-E2B-it",
        "adapter": PROJECT_ROOT / "models/gemma4-e2b-medical-lora",
    },
}

DEFAULT_EXAMPLES = [
    "At higher strengths (5, 7.5 and 10 mg), Arixtra is used to treat VTEs.",
    "Do not take more than two tablets in 24 hours.",
    "Keep this medicine out of the reach and sight of children.",
    "The medicine can only be obtained with a prescription.",
    "The usual dose is 5 mg once daily after meals.",
]


@st.cache_resource(max_entries=12)
def get_runtime(model_id: str, adapter_dir: str | None, index_dir: str) -> TranslationRuntime:
    runtime = TranslationRuntime(
        model_id=model_id,
        adapter_dir=adapter_dir,
        index_dir=index_dir,
    )
    runtime.load()
    return runtime


def available_models() -> list[str]:
    result = []
    cloud_allowed = {"Qwen3-4B base", "Qwen3-4B SFT · 100k"}
    for label, config in MODEL_CONFIG.items():
        if CLOUD_MODE and label not in cloud_allowed:
            continue
        adapter = config["adapter"]
        if label.endswith("50k") or label.endswith("100k") or label.endswith("SFT"):
            if adapter is None or not Path(adapter).exists():
                continue
        result.append(label)
    return result


def available_indexes() -> dict[str, Path]:
    indexes = {}
    index_100k = PROJECT_ROOT / "artifacts/rag_100k"
    index_50k = PROJECT_ROOT / "artifacts/rag"
    if index_100k.exists():
        indexes["100k translation memory"] = index_100k
    if index_50k.exists():
        indexes["50k translation memory"] = index_50k
    return indexes or {"default translation memory": index_50k}


def load_examples() -> list[str]:
    path = PROJECT_ROOT / "artifacts/examples.json"
    candidates = list(DEFAULT_EXAMPLES)
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            candidates.extend(row.get("source", "") for row in rows)
        except Exception:
            pass
    result = []
    for source in candidates:
        if source and source not in result:
            result.append(source)
    return result[:16]


st.title(APP_NAME)
st.caption(
    "A local English→German healthcare-information translation assistant. "
    "Administrative support only; human review is required."
)

with st.sidebar:
    st.header("MediLingo controls")
    model_label = st.selectbox("Model", available_models())
    indexes = available_indexes()
    memory_label = st.selectbox("Evidence memory", list(indexes))
    rag_enabled = st.checkbox("Use local evidence retrieval", value=True)
    entity_filter = st.checkbox(
        "Block conflicting medicine evidence",
        value=True,
        help="Reject a retrieved example when its detected product/entity conflicts with the input.",
    )
    similarity_threshold = st.slider(
        "Minimum evidence similarity",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.01,
    )
    top_k = st.slider("Retrieved examples", min_value=1, max_value=8, value=4)
    glossary_top_k = st.slider("Retrieved terminology", min_value=1, max_value=12, value=8)
    max_new_tokens = st.slider(
        "Maximum output tokens",
        min_value=64,
        max_value=512,
        value=256,
        step=32,
    )
    st.divider()
    st.info(
        "MediLingo uses only local model weights, a local translation memory, "
        "and a local terminology glossary. It does not browse the internet "
        "or provide clinical recommendations."
    )

examples = load_examples()
selected_example = st.selectbox("Load a healthcare example", ["(custom)"] + examples)
default_text = "" if selected_example == "(custom)" else selected_example
source_text = st.text_area(
    "English healthcare-information text",
    value=default_text,
    height=180,
    placeholder="Enter an English medication-information sentence or paragraph.",
)

translate_clicked = st.button("Translate with MediLingo", type="primary", use_container_width=True)

if translate_clicked:
    if not source_text.strip():
        st.warning("Enter source text first.")
    else:
        config = MODEL_CONFIG[model_label]
        adapter = config["adapter"]
        adapter_string = str(adapter) if adapter and Path(adapter).exists() else None
        index_dir = indexes[memory_label]
        try:
            with st.spinner("Loading the local model and translating..."):
                runtime = get_runtime(
                    config["model_id"],
                    adapter_string,
                    str(index_dir),
                )
                result = runtime.translate(
                    source_text.strip(),
                    use_rag=rag_enabled,
                    top_k=top_k,
                    glossary_top_k=glossary_top_k,
                    similarity_threshold=similarity_threshold,
                    entity_filter=entity_filter,
                    max_new_tokens=max_new_tokens,
                )
            st.subheader("German translation")
            st.success(result["translation"] or "The model returned an empty translation.")

            verification = result["verification"]
            st.subheader("Consistency checks")
            cols = st.columns(6)
            cols[0].metric("Numbers", "OK" if verification["number_preserved"] else "CHECK")
            cols[1].metric("Units", "OK" if verification["unit_preserved"] else "CHECK")
            cols[2].metric("Dosage", "OK" if verification["dosage_preserved"] else "CHECK")
            cols[3].metric(
                "Medicine name",
                "OK" if verification["medicine_name_preserved"] else "CHECK",
            )
            cols[4].metric(
                "Negation",
                "OK" if verification["negation_present_when_expected"] else "CHECK",
            )
            cols[5].metric("Warnings", len(verification["warnings"]))

            if verification["warnings"]:
                st.warning(
                    "Human review required: " + ", ".join(verification["warnings"])
                )

            diagnostics = result["retrieved"].get("diagnostics", {})
            with st.expander("Evidence and terminology", expanded=rag_enabled):
                if not rag_enabled:
                    st.info("Local evidence retrieval was disabled for this translation.")
                elif result["rag_used"]:
                    st.success(
                        "Evidence used: "
                        f"{len(result['retrieved']['translation_memory'])} translation examples, "
                        f"{len(result['retrieved']['glossary'])} glossary terms."
                    )
                    if diagnostics.get("entity_conflicts_filtered"):
                        st.caption(
                            "MediLingo filtered "
                            f"{diagnostics['entity_conflicts_filtered']} conflicting evidence candidate(s)."
                        )
                    for item in result["retrieved"]["translation_memory"]:
                        entity_note = " · entity match" if item.get("entity_match") else ""
                        st.markdown(
                            f"**Similarity {item.get('similarity', 0.0):.3f}{entity_note}**  \n"
                            f"EN: {item.get('source', '')}  \n"
                            f"DE: {item.get('target', '')}  \n"
                            f"Source: {item.get('provenance', item.get('dataset', 'unknown'))}"
                        )
                    if result["retrieved"]["glossary"]:
                        st.markdown("**Terminology used as reference**")
                        for item in result["retrieved"]["glossary"]:
                            st.write(
                                f"{item.get('source_term', '')} → {item.get('target_term', '')}"
                            )
                else:
                    st.info(
                        "No sufficiently similar, non-conflicting evidence was used. "
                        "The model translated without a translation-memory example."
                    )
                st.caption(
                    f"Retrieval gate: {diagnostics.get('retrieval_gate', 'unknown')} · "
                    f"threshold: {diagnostics.get('similarity_threshold', similarity_threshold):.2f}"
                )

            with st.expander("Raw verification details"):
                st.json(verification)
        except Exception as exc:
            st.error(
                "The selected model could not be loaded or run. "
                "Check the project logs and model access."
            )
            st.exception(exc)

st.divider()
st.caption(
    "MediLingo research prototype: compare Base → SFT → SFT + local evidence "
    "using ChrF, terminology, number/unit preservation, and human review."
)
