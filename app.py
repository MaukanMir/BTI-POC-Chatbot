# app.py
import streamlit as st
from rag_pipeline import retrieve_and_rerank, build_context, build_prompt, call_bedrock_llama



# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Demo (FAISS + Re-ranker + Bedrock)")

# -----------------------------
# SESSION STATE (CHAT MEMORY)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
query = st.chat_input("Ask a question about your benefits plan")

# -----------------------------
# HANDLE NEW MESSAGE
# -----------------------------
if query:
    # store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = retrieve_and_rerank(query, k=10, top_n=5)
            context = build_context(results)
            prompt = build_prompt(query, context)

            answer = call_bedrock_llama(prompt)
            st.write(answer)

        with st.expander("Retrieved context"):
            for r in results:
                st.markdown(f"**Score:** {r['rerank_score']:.4f}")
                st.write(r["text"])
                st.divider()

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# -----------------------------
# CLEAR CHAT
# -----------------------------
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()