import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Create Query Embedding
# -----------------------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()
    return r.json()["embeddings"]


# -----------------------------
# LLaMA Inference
# -----------------------------
def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    r.raise_for_status()

    return r.json()["response"]


# -----------------------------
# Load database
# -----------------------------
df = joblib.load("embeddings.joblib")

# -----------------------------
# User Query
# -----------------------------
query = input("\nAsk a Question: ")

# -----------------------------
# Query Embedding
# -----------------------------
query_embedding = create_embedding([query])[0]

# -----------------------------
# Similarity Search
# -----------------------------
embedding_matrix = np.vstack(df["embedding"].values)

similarities = cosine_similarity(
    embedding_matrix,
    [query_embedding]
).flatten()

top_k = 3

top_indices = similarities.argsort()[-top_k:][::-1]

retrieved_chunks = df.iloc[top_indices]

# -----------------------------
# Print Retrieved Chunks
# -----------------------------
print("\n==============================")
print("Retrieved Chunks")
print("==============================")

context = ""

for _, row in retrieved_chunks.iterrows():

    print("\n------------------------------")
    print("Chunk ID :", row["chunk_id"])
    print("Similarity :", similarities[row.name])

    print("\nText:\n")
    print(row["text"])

    context += f"""
Chunk {row['chunk_id']}:
{row['text']}

"""

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are an AI assistant for a medical company.

Answer ONLY using the information provided in the Context.

If the answer cannot be found in the Context, reply:

"The provided documents do not contain enough information to answer this question."

If the question is unrelated to liver topics, politely ask the user to ask a liver-related question.

--------------------
Context
--------------------

{context}

--------------------
Question
--------------------

{query}

--------------------
Answer
--------------------
"""

# Save prompt for debugging
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# -----------------------------
# LLM Answer
# -----------------------------
response = inference(prompt)

# Save response
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# -----------------------------
# Final Answer
# -----------------------------
print("\n==============================")
print("FINAL ANSWER")
print("==============================\n")

print(response)
