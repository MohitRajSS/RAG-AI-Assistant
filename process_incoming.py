import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

# -----------------------------
# Create Embedding
# -----------------------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
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

    data = r.json()
    return data.get("response", "No response from model")


# -----------------------------
# Load stored embeddings
# -----------------------------
df = joblib.load('embeddings.joblib')


# -----------------------------
# Ask user query
# -----------------------------
query = input("\nAsk a Question: ")


# -----------------------------
# Create query embedding
# -----------------------------
query_embedding = create_embedding([query])[0]



# -----------------------------
# Convert embeddings into matrix
# -----------------------------
embedding_matrix = np.vstack(df["embedding"].values)


# -----------------------------
# Compute cosine similarity
# -----------------------------
similarities = cosine_similarity(
    embedding_matrix,
    [query_embedding]
).flatten()


# -----------------------------
# Get top-k matches
# -----------------------------
top_k = 3
top_indices = similarities.argsort()[-top_k:][::-1]
new_df = df.iloc[top_indices]


# -----------------------------
# Create prompt
# -----------------------------
prompt = f'''I am working in a medical company and need the answers according to the content below:

{new_df.to_json()}

----------------------------------------

User Question:
{query}

User asked this question related to the liver. Answer ONLY from the above content.
If the question is unrelated, ask the user to stay within liver-related topics.
'''


# -----------------------------
# Print retrieved chunks
# -----------------------------
print("\n==============================")
print("Top Matching Chunks")
print("==============================")

for i in top_indices:
    print("\n------------------------------")
    print("Chunk ID :", df.iloc[i]["chunk_id"])
    print("Similarity Score :", similarities[i])
    print("\nText:\n")
    print(df.iloc[i]["text"])


# -----------------------------
# Get LLaMA Answer
# -----------------------------
response = inference(prompt)


# -----------------------------
# Print final answer
# -----------------------------
print("\n==============================")
print("FINAL ANSWER (LLaMA)")
print("==============================\n")

print(response.strip())


# -----------------------------
# Save prompt + response
# -----------------------------
with open("prompt.txt", "w") as f:
    f.write(prompt)

with open("response.txt", "w") as f:
    f.write(response)