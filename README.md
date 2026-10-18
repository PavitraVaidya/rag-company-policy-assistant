📄 RAG vs. LLM — Company Policy Assistant

A hands-on project exploring the difference between answering questions with a Large Language Model (LLM) alone and answering questions using Retrieval-Augmented Generation (RAG).

🌐 Live Demo: "Try the Company Policy Assistant" (https://pavitravaidya-rag-company-policy-assis-rag-experimentapp-mdimem.streamlit.app/)

🤔 What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances an LLM's responses by retrieving relevant information from an external knowledge source and providing it as context to the model.

A traditional LLM generates responses using the information available through its training and the context supplied in the prompt. It may not know the latest or specific information contained in a private company document.

RAG addresses this limitation by retrieving relevant information from a knowledge source before generating a response.

In simple terms:

- Without RAG: The LLM answers without access to the external document.
- With RAG: The system retrieves relevant information from the document and supplies it to the LLM to generate a context-aware answer.

🎯 Project Objective

The objective of this project is to understand RAG practically by comparing two approaches to answering questions about a company policy document.

The application allows users to ask a question and compare the answers generated:

1. Without providing the company policy as context.
2. With relevant policy statements retrieved and provided as context.

⚙️ How Does RAG Work?

The application follows these steps:

1. Document loading: Read the company policy document.
2. Text preparation: Split the document into individual text statements.
3. Embedding generation: Convert the statements into numerical vectors using Sentence Transformers.
4. Vector indexing: Store the vectors in a FAISS index.
5. Query embedding: Convert the user's question into an embedding.
6. Similarity search: Retrieve the two most relevant policy statements using FAISS.
7. Context augmentation: Add the retrieved statements to the LLM prompt.
8. Answer generation: Use the Groq API to generate an answer grounded in the retrieved context.
9. Comparison: Display the answers with and without RAG side by side.

🏗️ Architecture

             Company Policy Document
                       |
                 Text Preparation
                       |
              Sentence Transformers
                 (Embeddings)
                       |
                  FAISS Index
                       |
User Question → Query Embedding
                       |
                 Similarity Search
                       |
             Relevant Policy Statements
                       |
                Context + Question
                       |
                   Groq LLM
                       |
                RAG-Based Answer

The application also sends the question directly to the LLM without policy context, allowing users to compare both approaches.

🛠️ Tech Stack

- Python — Application logic
- Streamlit — Interactive web interface
- Groq API — Hosted LLM inference
- Sentence Transformers — Text embeddings using "all-MiniLM-L6-v2"
- FAISS — Vector similarity search
- NumPy — Numerical operations and embedding arrays

✨ Features

- Compare LLM responses with and without RAG.
- Ask questions about a company policy document.
- Retrieve relevant policy statements using semantic similarity.
- View the retrieved context used to generate the RAG answer.
- Use a hosted LLM API instead of requiring a locally running LLM.
- Access the application through a deployed web interface.

🚀 Run Locally

1. Clone the repository

git clone https://github.com/PavitraVaidya/rag-company-policy-assistant.git
cd rag-company-policy-assistant/rag_experiment

2. Create and activate a virtual environment

python -m venv venv

Windows PowerShell:

.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Configure your Groq API key

Create a file named ".streamlit/secrets.toml" inside the "rag_experiment" directory:

GROQ_API_KEY = "your_groq_api_key"

Replace the placeholder with your own API key. Never commit this file or expose your key publicly.

5. Run the application

streamlit run app.py

The application will be available at the local URL shown in your terminal, typically "http://localhost:8501".

🧪 Example Experiment

Question: How many unused leave days can employees carry forward?

Policy information: Employees can carry forward a maximum of five unused leave days to the next year.

- Without RAG: The model may not know the company-specific policy and should acknowledge when it lacks sufficient information.
- With RAG: The application retrieves the relevant policy statement and uses it to answer the question.

Note: Actual responses may vary. RAG can improve grounding, but its reliability depends on retrieval quality, the provided context, and the LLM.

🔐 Security

- Store API keys in Streamlit Secrets or environment-specific secret storage.
- Never hardcode API keys into source files.
- Do not commit ".streamlit/secrets.toml" or other files containing credentials.

📚 Key Learnings

Through this project, I explored:

- The fundamentals of Retrieval-Augmented Generation.
- How text embeddings represent semantic meaning.
- How vector similarity search retrieves relevant information.
- How retrieved context can guide LLM responses.
- How to integrate a hosted LLM API into a Python application.
- How to deploy an AI application using Streamlit Community Cloud.

🔮 Future Improvements

- Support PDF and multiple-document uploads.
- Add source citations and document references to answers.
- Experiment with chunking strategies and retrieval parameters.
- Compare different embedding models.
- Add conversation history and multi-turn question answering.
- Evaluate retrieval accuracy and answer faithfulness.

👩‍💻 Author

Pavitra Vaidya

Interested in Python, Generative AI, LLM applications, and building practical AI projects.

- Live Demo: "Company Policy Assistant" (https://pavitravaidya-rag-company-policy-assis-rag-experimentapp-mdimem.streamlit.app/)
- GitHub: "rag-company-policy-assistant" (https://github.com/PavitraVaidya/rag-company-policy-assistant)
