# app.py
import streamlit as st
from rag_pipeline import retrieve_and_rerank, build_context, build_prompt, call_bedrock_llama


st.title("RAG Demo (FAISS + Re-ranker + Bedrock)")

query = st.text_input("Ask a question about your benefits plan:")

if query:
    with st.spinner("Thinking..."):
        results = retrieve_and_rerank(query, k=10, top_n=5)

        context = build_context(results)
        prompt = build_prompt(query, context)

        answer = call_bedrock_llama(prompt)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved context"):
        for r in results:
            st.markdown(f"**Score:** {r['rerank_score']:.4f}")
            st.write(r["text"])
            st.divider()