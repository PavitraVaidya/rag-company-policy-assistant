
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the company policy
with open("company_policy.txt", "r", encoding="utf-8") as file:
    document = file.read()

# 2. Split the document into individual policy statements
chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]

# 3. Load the embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Convert each policy statement into a vector
embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

# 5. Create a FAISS similarity-search index
embeddings = np.array(embeddings, dtype="float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 6. Ask the user a question
question = input("Ask a question about the company policy: ")

# 7. Convert the question into a vector
question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)
question_embedding = np.array(
    question_embedding, dtype="float32"
)

# 8. Retrieve the two most relevant policy statements
scores, indices = index.search(question_embedding, k=2)

context = "\n".join(
    chunks[i] for i in indices[0] if i != -1
)

# 9. Give the retrieved context and question to the LLM
prompt = f"""
Answer the question using only the company policy below.

If the policy does not contain the answer, clearly say:
"The provided policy does not specify this information."

Do not use outside knowledge or invent company rules.

Company policy:
{context}

Question: {question}
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# 10. Display the retrieved context and final answer
print("\nRetrieved policy statements:")
print(context)

print("\nRAG Answer:")
print(response["message"]["content"])
