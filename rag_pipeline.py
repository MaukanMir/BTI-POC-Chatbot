# rag_pipeline.py

import pandas as pd
import numpy as np
import faiss
import json
import boto3

from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str
    results: list
    answer: str
    conversation: List[str]   # prior user turns
    last_context: Optional[str]
    
    

# -----------------------------
# LOAD DATA (runs once)
# -----------------------------
df = pd.read_csv("FEP_Blue_Focus_2026_metadata.csv")

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(
    df["text"].astype(str).tolist(),
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

# -----------------------------
# FAISS INDEX
# -----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# -----------------------------
# RE-RANKER
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# BEDROCK CLIENT
# -----------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve_and_rerank(query, k=10, top_n=5):
    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, k)

    candidates = [
        {"text": df.iloc[idx]["text"], "faiss_score": float(scores[0][i])}
        for i, idx in enumerate(indices[0])
    ]

    rerank_inputs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(rerank_inputs)

    for c, r_score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(r_score)

    return sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )[:top_n]

# -----------------------------
# PROMPT BUILDING
# -----------------------------
def build_context(results):
    return "\n\n".join(r["text"] for r in results)

def build_prompt(query, context):
    return f"""
<|begin_of_text|>

<|system|>
You are an expert healthcare benefits assistant.

Answer ONLY using the provided context.
If the answer is not present, say "Not mentioned."
Do not hallucinate or add information.
<|/system|>

<|user|>
Context:
{context}

Question:
{query}

Answer:
<|/user|>
"""

# -----------------------------
# LLM CALL
# -----------------------------
def call_bedrock_llama(prompt):
    response = bedrock.invoke_model(
        modelId="meta.llama3-70b-instruct-v1:0",
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 2000,
            "temperature": 0,
            "top_p": 0.9
        })
    )

    result = json.loads(response["body"].read())
    return result["generation"]
