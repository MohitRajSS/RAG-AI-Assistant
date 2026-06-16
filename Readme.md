# RAG AI Teaching Assistant for Liver Medical Knowledge Base

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** based AI assistant designed to answer **liver-related medical questions** using a company-specific knowledge base.

Instead of relying solely on the Large Language Model (LLM), the application first retrieves the most relevant information from the internal medical documents and then generates an answer using only that retrieved context. This approach improves accuracy and minimizes hallucinations.

The project uses:

* **BGE-M3** for generating embeddings
* **Llama 3.2** for response generation
* **Ollama** for local model serving
* **Cosine Similarity** for semantic search
* **Joblib** for storing embeddings

---

# Project Structure

```
Project/
│
├── input.txt
├── preprocess_chunks.py
├── read_chunks_modified.py
├── embeddings.joblib
├── prompt.txt
├── response.txt
└── README.md
```

---

# Project Workflow

```
                input.txt
                     │
                     ▼
      Split into Paragraph Chunks
                     │
                     ▼
      Generate BGE-M3 Embeddings
                     │
                     ▼
      Store Embeddings in DataFrame
                     │
                     ▼
          Save as embeddings.joblib
                     │
                     ▼
             User Enters Query
                     │
                     ▼
      Generate Query Embedding
                     │
                     ▼
         Cosine Similarity Search
                     │
                     ▼
         Retrieve Top Matching Chunks
                     │
                     ▼
         Create Prompt with Context
                     │
                     ▼
          Llama 3.2 Generates Answer
                     │
                     ▼
            Display and Save Response
```

---

# File Description

## 1. input.txt

This file contains the company's liver-related medical knowledge.

Each paragraph is treated as an individual chunk during preprocessing.

Example:

```
Liver anatomy paragraph...

(blank line)

Liver function paragraph...

(blank line)

Liver disease paragraph...
```

The quality of the responses depends on the quality of the information stored in this file.

---

## 2. preprocess_chunks.py

This script converts the medical knowledge into vector embeddings.

### Steps Performed

1. Read the `input.txt` file.
2. Split the content into paragraph-based chunks.
3. Generate embeddings using the **BGE-M3** model.
4. Store each chunk along with its embedding.
5. Create a Pandas DataFrame.
6. Save the DataFrame as `embeddings.joblib`.

The generated DataFrame contains:

| Column    | Description             |
| --------- | ----------------------- |
| chunk_id  | Unique chunk identifier |
| text      | Original paragraph      |
| embedding | Vector representation   |

---

## 3. embeddings.joblib

This file acts as the project's vector database.

It stores:

* Chunk ID
* Original text
* Embedding vector

The embeddings are generated only once and reused for every query, making retrieval significantly faster.

---

## 4. read_chunks_modified.py

This script performs the Retrieval-Augmented Generation (RAG) process.

### Workflow

**Step 1**

Load the stored embeddings.

```
embeddings.joblib
```

↓

**Step 2**

Accept a user question.

Example:

```
What are the major functions of the liver?
```

↓

**Step 3**

Generate an embedding for the user query using **BGE-M3**.

↓

**Step 4**

Compute cosine similarity between the query embedding and stored embeddings.

↓

**Step 5**

Retrieve the Top-3 most relevant chunks.

↓

**Step 6**

Create a prompt consisting of:

* Retrieved context
* User question
* System instructions

↓

**Step 7**

Send the prompt to **Llama 3.2**.

↓

**Step 8**

Display the final answer.

↓

**Step 9**

Save:

```
prompt.txt
response.txt
```

---

# Retrieval Pipeline

```
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Retrieve Top 3 Chunks
      │
      ▼
Build Prompt
      │
      ▼
Llama 3.2
      │
      ▼
Final Answer
```

---

# Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core Programming Language |
| Pandas       | Data Handling             |
| NumPy        | Numerical Operations      |
| Scikit-Learn | Cosine Similarity         |
| Requests     | API Calls                 |
| Joblib       | Saving Embeddings         |
| Ollama       | Local Model Serving       |
| BGE-M3       | Embedding Model           |
| Llama 3.2    | Language Model            |

---

# Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn joblib requests
```

Install the required Ollama models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Start the Ollama server:

```bash
ollama serve
```

---

# Running the Project

## Step 1

Populate `input.txt` with liver-related medical information.

---

## Step 2

Generate embeddings.

```bash
python preprocess_chunks.py
```

This creates:

```
embeddings.joblib
```

---

## Step 3

Run the RAG assistant.

```bash
python read_chunks_modified.py
```

---

## Step 4

Enter a liver-related question.

Example:

```
Ask a Question:

What are the functions of the liver?
```

---

## Step 5

The application will display:

* Top matching chunks
* Similarity scores
* Retrieved medical context
* Final answer generated by Llama 3.2

---

# Prompt Strategy

The model receives a prompt in the following format:

```
Retrieved Liver Context

----------------------------------

User Question

Answer ONLY from the retrieved content.

If the question is unrelated to liver topics,
politely ask the user to stay within liver-related discussions.
```

This ensures that the generated responses remain grounded in the company's internal medical knowledge.

---

# Output Files

## prompt.txt

Stores the exact prompt sent to the LLM.

Useful for:

* Prompt debugging
* Prompt engineering
* Evaluation

---

## response.txt

Stores the final generated answer.

Useful for:

* Logging
* Testing
* Model comparison

---

# Features

* Retrieval-Augmented Generation (RAG)
* Company-specific liver knowledge base
* Local inference using Ollama
* BGE-M3 semantic embeddings
* Cosine similarity retrieval
* Top-K document search
* Context-aware answer generation
* Prompt and response logging
* Lightweight and modular architecture

---

# Current Scope

The knowledge base is intentionally limited to **liver-related medical information** contained in `input.txt`.

Questions outside this domain are not answered directly. Instead, the assistant politely requests that the user ask liver-related questions.

---

# Future Improvements

* FAISS/ChromaDB vector database integration
* PDF and DOCX document ingestion
* Metadata filtering
* Hybrid keyword + semantic search
* Cross-encoder reranking
* Multi-document indexing
* Conversation memory
* Streamlit/Gradio web interface
* Source citation for retrieved chunks
