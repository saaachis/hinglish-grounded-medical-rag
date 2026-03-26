"""
Streamlit demo — Grounded RAG for Hinglish Clinical Queries.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "evidence.index"

EXAMPLE_QUERIES = [
    "Meri beti ko ek rash hai jo theek nahi ho raha hai. Hydrocortisone cream help nahi karti.",
    "Doctor meri eye me sujan hai aur laal ho gayi hai, bahut dard ho raha hai",
    "Mere gale me tonsils me dard hai aur khaana nahi kha pa rahi",
    "Mujhe pair me sujan hai aur chalne me bahut takleef hoti hai",
    "Meri skin pe ek growth aa gayi hai jo badhti ja rahi hai",
    "Mere muh me ulcers hain aur kuch bhi khaane me jalan hoti hai",
]


def check_prerequisites() -> str | None:
    """Return an error message if prerequisites are missing, else None."""
    if not INDEX_FILE.exists():
        return (
            "FAISS index not found. Run `python build_index.py` first to "
            "build the evidence index."
        )
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "GROQ_API_KEY not set. Add it to your .env file."
    return None


@st.cache_resource(show_spinner="Loading RAG pipeline (LaBSE + FAISS) ...")
def load_pipeline():
    from src.pipeline import RAGPipeline
    api_key = os.getenv("GROQ_API_KEY", "")
    return RAGPipeline(api_key=api_key, index_dir=INDEX_DIR, max_k=5)


def render_header():
    st.set_page_config(
        page_title="Hinglish Medical RAG",
        page_icon="🏥",
        layout="wide",
    )
    st.title("Grounded RAG for Hinglish Clinical Queries")
    st.markdown(
        "Type a **Hinglish patient query** and the system will retrieve "
        "relevant clinical evidence from **MultiCaRe** and generate a "
        "grounded response using **Llama-3.1-8B**."
    )


def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.markdown(
            "**Architecture**\n\n"
            "1. **Encode** query with LaBSE (cross-lingual)\n"
            "2. **Retrieve** top-k evidence via FAISS\n"
            "3. **Generate** grounded Hinglish response\n\n"
            "---\n\n"
            "**Components**\n\n"
            "- Encoder: `sentence-transformers/LaBSE`\n"
            "- Index: FAISS (inner-product)\n"
            "- Generator: Llama-3.1-8B-Instant (Groq)\n"
            "- Evidence: MultiCaRe clinical cases\n"
        )
        st.markdown("---")
        top_k = st.slider("Retrieved evidence (top-k)", 1, 10, 3)
        show_zero_shot = st.checkbox("Compare with zero-shot", value=True)
    return top_k, show_zero_shot


def render_results(result: dict, show_zero_shot: bool):
    st.markdown("---")

    st.subheader("Retrieved Evidence")
    evidence = result["retrieved_evidence"]
    for i, ev in enumerate(evidence):
        with st.expander(
            f"Evidence {i+1}: {ev['condition_group']}  |  "
            f"Similarity: {ev['score']:.3f}  |  {ev['case_id']}"
        ):
            excerpt = " ".join(ev["case_text"].split()[:250])
            st.markdown(f"```\n{excerpt}\n```")

    st.markdown("---")

    if show_zero_shot and "zero_shot_response" in result:
        col_g, col_z = st.columns(2)

        with col_g:
            st.subheader("Grounded Response")
            st.success(result["grounded_response"])
            c1, c2 = st.columns(2)
            c1.metric("Factual Support", f"{result['grounded_factual_score']:.2f}")
            c2.metric("Hallucination", f"{result['grounded_hallucination_score']:.2f}")

        with col_z:
            st.subheader("Zero-Shot Response")
            st.warning(result["zero_shot_response"])
            c1, c2 = st.columns(2)
            c1.metric("Factual Support", f"{result['zero_shot_factual_score']:.2f}")
            c2.metric("Hallucination", f"{result['zero_shot_hallucination_score']:.2f}")
    else:
        st.subheader("Grounded Response")
        st.success(result["grounded_response"])
        c1, c2 = st.columns(2)
        c1.metric("Factual Support", f"{result['grounded_factual_score']:.2f}")
        c2.metric("Hallucination", f"{result['grounded_hallucination_score']:.2f}")


def main():
    render_header()
    top_k, show_zero_shot = render_sidebar()

    err = check_prerequisites()
    if err:
        st.error(err)
        return

    pipeline = load_pipeline()

    st.markdown("#### Try an example query")
    cols = st.columns(3)
    selected_example = None
    for i, ex in enumerate(EXAMPLE_QUERIES[:6]):
        col = cols[i % 3]
        label = ex[:60] + "..." if len(ex) > 60 else ex
        if col.button(label, key=f"example_{i}", use_container_width=True):
            selected_example = ex

    st.markdown("#### Or type your own")
    user_query = st.text_area(
        "Hinglish query",
        value=selected_example or "",
        height=80,
        placeholder="Doctor, meri skin pe rash hai aur bahut khujli ho rahi hai ...",
    )

    if st.button("Get Response", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Retrieving evidence and generating response ..."):
            result = pipeline.query(
                user_query.strip(),
                top_k=top_k,
                include_zero_shot=show_zero_shot,
            )
        render_results(result, show_zero_shot)


if __name__ == "__main__":
    main()
