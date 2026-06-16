import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
# -----------------------------
# Create Embeddings
# -----------------------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    embeddings = r.json()["embeddings"]
    return embeddings


# -----------------------------
# Read txt file and create chunks
# -----------------------------
file_path = "input.txt"      # Your txt file

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split on blank lines (each paragraph is one chunk)
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Total Chunks: {len(paragraphs)}")

# -----------------------------
# Create embeddings
# -----------------------------
embeddings = create_embedding(paragraphs)

# -----------------------------
# Store in list of dictionaries
# -----------------------------
my_dicts = []

for idx, (chunk, embedding) in enumerate(zip(paragraphs, embeddings)):
    my_dicts.append({
        "chunk_id": idx,
        "text": chunk,
        "embedding": embedding
    })

# -----------------------------
# Convert to DataFrame
# -----------------------------
df = pd.DataFrame.from_records(my_dicts)
#Save this dataframe
joblib.dump(df,'embeddings.joblib')

print(df.head())

