
import streamlit as st
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="RAG vs LLM | Policy Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Company Policy AI Assistant")
st.caption("Compare an LLM's direct answer with retrieval-grounded answers.")

# Load the demonstration policy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "company_policy.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Display the complete policy on the webpage
with st.expander("📚 View Company Policy", expanded=True):
    st.markdown(document)

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

    vectors = model.encode(
        chunks,
        normalize_embeddings=True
    )
    vectors = np.asarray(vectors, dtype="float32")

    db = faiss.IndexFlatIP(vectors.shape[1])
    db.add(vectors)

    return model, db

@st.cache_resource
def get_llm_client(api_key):
    return Groq(api_key=api_key)

def ask_llm(client, prompt):
    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return result.choices[0].message.content

try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "Groq API key is missing. Add GROQ_API_KEY "
        "to Streamlit Secrets before asking questions."
    )
    st.stop()

client = get_llm_client(api_key)

with st.spinner("Loading the embedding model..."):
    embedding_model, index = prepare_rag()

question = st.text_input(
    "Ask a question about the company policy",
    placeholder="How many unused leave days can employees carry forward?"
)

if st.button("Compare Answers", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Generating both answers..."):

                # WITHOUT RAG: no company policy context
                direct_answer = ask_llm(
                    client,
                    f"""
Answer the question directly.
You have not been given the company's policy document.
If you do not know the company-specific answer,
say that you do not have enough information.

Question: {question}
"""
                )

                # WITH RAG: retrieve relevant policy statements
                question_vector = embedding_model.encode(
                    [question],
                    normalize_embeddings=True
                )
                question_vector = np.asarray(
                    question_vector,
                    dtype="float32"
                )

                scores, indices = index.search(
                    question_vector, k=min(2, len(chunks))
                )

                retrieved = [
                    chunks[i]
                    for i in indices[0]
                    if i != -1
                ]

                context = "\n".join(retrieved)

                rag_answer = ask_llm(
                    client,
                    f"""
Answer using only the company policy context below.

If the context does not contain the answer, say:
"The provided policy does not specify this information."

Do not invent company rules.

Policy context:
{context}

Question: {question}
"""
                )

            left, right = st.columns(2)

            with left:
                st.subheader("🔹 Without RAG")
                st.write(direct_answer)

            with right:
                st.subheader("🔹 With RAG")
                st.write(rag_answer)

                with st.expander("View retrieved policy statements"):
                    st.write(context if context else "No statements retrieved.")

        except Exception as error:
            st.error(
                "The request failed. Check your API key, "
                "model availability, network connection, and API limits."
            )
            st.caption(f"Error details: {error}")
