from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "cognivault_documents"


def load_chunks(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    chunk_files = sorted(
        PROCESSED_DIR.glob("*_chunks.json")
    )

    if not chunk_files:

        print("No chunk files found.")

        return

    print("Documents to index:")

    for file in chunk_files:
        print(f"  - {file.name}")

    print(
        f"\nTotal documents: "
        f"{len(chunk_files)}"
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Existing documents: "
        f"{collection.count()}"
    )

    total_indexed = 0

    for chunk_file in chunk_files:

        print("\n" + "=" * 70)

        print(
            f"Processing: "
            f"{chunk_file.name}"
        )

        print("=" * 70)

        chunks = load_chunks(
            chunk_file
        )

        print(
            f"Chunks: "
            f"{len(chunks)}"
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(
            "Generating embeddings..."
        )

        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        ids = [
            f"{chunk['source']}_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        metadatas = []

        for chunk in chunks:

            metadata = {
                "source": chunk["source"],
                "page": chunk.get(
                    "page",
                    chunk.get("page_start", 0)
                ),
                "page_start": chunk.get(
                    "page_start",
                    chunk.get("page", 0)
                ),
                "page_end": chunk.get(
                    "page_end",
                    chunk.get("page", 0)
                ),
                "chunk_id": chunk["chunk_id"]
            }

            metadatas.append(
                metadata
            )

        print(
            "Adding chunks to ChromaDB..."
        )

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        total_indexed += len(chunks)

        print(
            "Indexing complete."
        )

    print("\n" + "=" * 70)

    print("CHROMA INDEX SUMMARY")

    print("=" * 70)

    print(
        f"Files indexed : "
        f"{len(chunk_files)}"
    )

    print(
        f"Chunks added  : "
        f"{total_indexed}"
    )

    print(
        f"Documents     : "
        f"{collection.count()}"
    )

    print(
        f"Database      : "
        f"{CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()
