"""
Streamlit Demo Interface — Hinglish Grounded Medical RAG.

A clinical decision support interface that:
1. Accepts Hinglish patient queries
2. Retrieves relevant medical evidence
3. Generates grounded Hinglish explanations
4. Displays supporting evidence and confidence indicators
"""

import streamlit as st


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="Hinglish Medical RAG",
        page_icon="🏥",
        layout="wide",
    )

    st.title("Hinglish Grounded Medical RAG")
    st.markdown(
        "Evidence-first clinical decision support for "
        "Hinglish (Hindi-English) patient queries."
    )

    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Number of evidence documents", 1, 10, 5)
        use_multimodal = st.checkbox("Include X-ray evidence", value=False)
        st.divider()
        st.caption("This is a clinical decision support tool. "
                   "It does not perform diagnosis.")

    # --- Main Interface ---
    st.subheader("Enter Patient Query (Hinglish)")

    query = st.text_area(
        label="Hinglish Query",
        placeholder=(
            'e.g., "Doctor, right side chhati me pani bhar gaya hai kya?"'
        ),
        height=100,
    )

    if st.button("Generate Explanation", type="primary"):
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            st.info("Pipeline not yet connected. This is a demo stub.")

            # Placeholder for pipeline integration
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Retrieved Evidence")
                st.markdown("*Evidence documents will appear here...*")

            with col2:
                st.subheader("Grounded Explanation")
                st.markdown("*Generated Hinglish explanation will appear here...*")


if __name__ == "__main__":
    main()
