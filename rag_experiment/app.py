
import streamlit as st
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Company Policy AI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Company Policy AI Assistant")
st.write(
    "Explore the policy and compare answers "
    "with and without Retrieval-Augmented Generation."
)

# Load policy
with open("company_policy.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Display the complete policy
with st.expander("📚 View Company Policy", expanded=True):
    st.markdown(document)

# Prepare policy chunks
chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]

@st.cache_resource
def prepare_rag():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    db = faiss.IndexFlatIP(embeddings.shape[1])
    db.add(embeddings)

    return model, db

embedding_model, index = prepare_rag()

question = st.text_input(
    "Ask a question about the company policy",
    placeholder="How many unused leave days can employees carry forward?"
)

if st.button("Compare Answers", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Without RAG
        with st.spinner("Generating answers..."):
            without_rag = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question directly. "
                            "You do not have access to the "
                            "company policy document.\n\n"
                            f"Question: {question}"
                        )
                    }
                ]
            )["message"]["content"]

            # With RAG
            q_embedding = embedding_model.encode(
                [question],
                normalize_embeddings=True
            )
            q_embedding = np.asarray(
                q_embedding, dtype="float32"
            )

            scores, indices = index.search(q_embedding, k=2)

            context = "\n".join(
                chunks[i]
                for i in indices[0]
                if i != -1
            )

            prompt = f"""
Answer using only the company policy context.
If the answer is not specified, say:
"The provided policy does not specify this information."
Do not invent company rules.

Policy context:
{context}

Question: {question}
"""

            with_rag = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"]

        # Display comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 Without RAG")
            st.write(without_rag)

        with col2:
            st.subheader("🔹 With RAG")
            st.write(with_rag)

            with st.expander("View retrieved policy statements"):
                st.write(context)
