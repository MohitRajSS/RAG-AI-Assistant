import requests
import pandas as pd
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

    r.raise_for_status()
    return r.json()["embeddings"]


# -----------------------------
# Read txt file
# -----------------------------
file_path = "input.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split paragraphs
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Total Chunks : {len(paragraphs)}")

# -----------------------------
# Create embeddings
# -----------------------------
embeddings = create_embedding(paragraphs)

# -----------------------------
# Store data
# -----------------------------
records = []

for idx, (chunk, embedding) in enumerate(zip(paragraphs, embeddings)):
    records.append({
        "chunk_id": idx,
        "text": chunk,
        "embedding": embedding
    })

df = pd.DataFrame(records)

joblib.dump(df, "embeddings.joblib")

print("\nEmbeddings saved successfully!")
print(df.head())
