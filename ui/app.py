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
PREMIUM_CSS = """
<style>
:root {
    --ink: #eef4ff;
    --muted: #9aa9c2;
    --panel: rgba(18, 28, 48, 0.82);
    --line: rgba(146, 177, 221, 0.18);
    --cyan: #53d6d2;
    --blue: #6b8cff;
    --mint: #a5f3d0;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 78% 4%, rgba(83, 214, 210, 0.11), transparent 30rem),
        radial-gradient(circle at 8% 24%, rgba(107, 140, 255, 0.10), transparent 28rem),
        #08111f;
}
[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.78);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c31 0%, #0a1425 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}
.block-container {
    max-width: 1380px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
.hero-card {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.25rem;
    border: 1px solid rgba(139, 190, 255, 0.24);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(22, 44, 75, 0.96), rgba(15, 26, 49, 0.88));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
}
.hero-card::after {
    content: "";
    position: absolute;
    width: 18rem;
    height: 18rem;
    right: -7rem;
    top: -8rem;
    border-radius: 50%;
    background: rgba(83, 214, 210, 0.14);
    filter: blur(3px);
}
.hero-kicker {
    color: var(--cyan);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.16em;
}
.hero-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
}
.hero-title {
    margin-top: 0.35rem;
    color: var(--ink);
    font-size: clamp(2.5rem, 6vw, 4.8rem);
    font-weight: 850;
    letter-spacing: -0.065em;
    line-height: 0.98;
}
.hero-title span {
    color: var(--cyan);
}
.hero-subtitle {
    max-width: 45rem;
    margin-top: 0.8rem;
    color: var(--muted);
    font-size: 1.02rem;
    line-height: 1.55;
}
.hero-badge, .model-pill {
    position: relative;
    z-index: 1;
    padding: 0.55rem 0.8rem;
    border: 1px solid rgba(165, 243, 208, 0.30);
    border-radius: 999px;
    background: rgba(165, 243, 208, 0.09);
    color: var(--mint);
    font-size: 0.78rem;
    font-weight: 750;
    white-space: nowrap;
}
.section-label {
    margin-top: 1.6rem;
    color: var(--ink);
    font-size: 1.25rem;
    font-weight: 750;
}
.helper-copy {
    color: var(--muted);
    font-size: 0.92rem;
}
[data-testid="stTextArea"] textarea {
    border: 1px solid rgba(139, 190, 255, 0.22);
    border-radius: 16px;
    background: rgba(8, 17, 31, 0.72);
    color: var(--ink);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 1px var(--cyan);
}
div.stButton > button[kind="primary"] {
    min-height: 3.25rem;
    border: 1px solid rgba(165, 243, 208, 0.38);
    border-radius: 14px;
    background: linear-gradient(100deg, #2aa9a7, #607cf1);
    color: white;
    font-size: 1rem;
    font-weight: 800;
    box-shadow: 0 12px 28px rgba(61, 125, 212, 0.28);
}
div.stButton > button[kind="primary"]:hover {
    border-color: white;
    background: linear-gradient(100deg, #38c9c1, #7890ff);
    transform: translateY(-1px);
}
[data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"] {
    border-radius: 16px;
}
[data-testid="stMetric"] {
    border: 1px solid var(--line);
    background: var(--panel);
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    background: rgba(14, 25, 44, 0.68);
}
[data-testid="stCaptionContainer"] {
    color: var(--muted);
}
footer {
    visibility: hidden;
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

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
    cloud_allowed = {"Qwen3-4B SFT · 100k"}
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


ACTIVE_BADGE = (
    "Qwen3-4B · 100k SFT adapter"
    if CLOUD_MODE
    else "Research comparison workspace"
)
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">MEDICAL INFORMATION · ENGLISH → GERMAN</div>
        <div class="hero-row">
            <div>
                <div class="hero-title">Medi<span>Lingo</span></div>
                <div class="hero-subtitle">
                    A focused translation assistant for healthcare information.
                    Preserve terminology, numbers, dosage, warnings, and negation
                    while keeping a human reviewer in the loop.
                </div>
            </div>
            <div class="hero-badge">{ACTIVE_BADGE}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-label">Translate any healthcare text</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="helper-copy">Examples are optional. Choose a starter example '
    "or paste your own sentence, paragraph, or document excerpt below.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### MediLingo controls")
    st.caption("English → German · healthcare-information workflow")
    if CLOUD_MODE:
        model_label = "Qwen3-4B SFT · 100k"
        st.markdown(
            '<div class="model-pill">Active model · Qwen3-4B 100k SFT</div>',
            unsafe_allow_html=True,
        )
    else:
        model_label = st.selectbox("Model", available_models())
    indexes = available_indexes()
    memory_label = st.selectbox("Evidence memory", list(indexes))
    rag_enabled = st.checkbox(
        "Use local evidence (optional)",
        value=True,
        help="Turn this off to translate free-form text without retrieved examples or glossary terms.",
    )
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
        "Local model weights and local evidence only. No web browsing and no clinical recommendations. "
        "Human review is required."
    )

examples = load_examples()
example_options = ["No example — write your own"] + examples
selected_example = st.selectbox("Optional starter example", example_options)
default_text = "" if selected_example == example_options[0] else selected_example
source_text = st.text_area(
    "English text to translate",
    value=default_text,
    height=190,
    placeholder="Paste any English healthcare-information sentence or paragraph.",
)
st.caption("Free-form input is supported — you do not need to select an example.")

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
st.caption("MediLingo · Qwen3-4B 100k SFT · optional local evidence · human review required.")
