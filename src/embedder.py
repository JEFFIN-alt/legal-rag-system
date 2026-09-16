from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))

    if not chunk_files:
        print("No chunk files found.")
        return

    model = SentenceTransformer(MODEL_NAME)

    for chunk_file in chunk_files:

        print(f"\nProcessing: {chunk_file.name}")

        chunks = load_chunks(chunk_file)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(f"Chunks loaded: {len(texts)}")

        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
