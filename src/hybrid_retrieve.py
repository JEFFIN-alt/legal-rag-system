from pathlib import Path
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cognivault_documents"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_K = 10


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))

    if not chunk_files:
        raise FileNotFoundError("No chunk files found.")

    with open(chunk_files[0], "r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    """
    Combine two ranked result lists using Reciprocal Rank Fusion.
    """

    scores = {}

    for rank, chunk_id in enumerate(semantic_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, chunk_id in enumerate(keyword_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [chunk_id for chunk_id, _ in ranked]


def main():

    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks)}")

    print("Building BM25 index...")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query = "What are the different types of electric current?"

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL")
    print("=" * 70)

    print(f"\nQuery: {query}")

    # ---------------------------------------------------------
    # 1. Semantic retrieval
    # ---------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=CANDIDATE_K
    )

    semantic_ids = semantic_results["ids"][0]

    # ---------------------------------------------------------
    # 2. Keyword retrieval using BM25
    # ---------------------------------------------------------

    tokenized_query = tokenize(query)

    keyword_scores = bm25.get_scores(tokenized_query)

    keyword_indices = sorted(
        range(len(keyword_scores)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )[:CANDIDATE_K]

    keyword_ids = [
        f"{chunks[i]['source']}_{chunks[i]['chunk_id']}"
        for i in keyword_indices
    ]

    # ---------------------------------------------------------
    # 3. Fuse both rankings
    # ---------------------------------------------------------

    fused_ids = reciprocal_rank_fusion(
        semantic_ids,
        keyword_ids
    )

    # ---------------------------------------------------------
    # 4. Show results
    # ---------------------------------------------------------

    chunk_lookup = {
        f"{chunk['source']}_{chunk['chunk_id']}": chunk
        for chunk in chunks
    }

    print("\n" + "-" * 70)
    print("SEMANTIC TOP RESULTS")
    print("-" * 70)

    for rank, chunk_id in enumerate(semantic_ids[:TOP_K], start=1):

        chunk = chunk_lookup[chunk_id]

        print(
            f"\n#{rank} | Page {chunk['page']} | "
            f"Chunk {chunk['chunk_id']}"
        )

        print(chunk["text"][:500])

    print("\n" + "-" * 70)
    print("KEYWORD TOP RESULTS (BM25)")
    print("-" * 70)

    for rank, chunk_id in enumerate(keyword_ids[:TOP_K], start=1):

        chunk = chunk_lookup[chunk_id]

        print(
            f"\n#{rank} | Page {chunk['page']} | "
            f"Chunk {chunk['chunk_id']}"
        )

        print(chunk["text"][:500])

    print("\n" + "-" * 70)
    print("HYBRID TOP RESULTS")
    print("-" * 70)

    for rank, chunk_id in enumerate(fused_ids[:TOP_K], start=1):

        chunk = chunk_lookup[chunk_id]

        print(
            f"\n#{rank} | Page {chunk['page']} | "
            f"Chunk {chunk['chunk_id']}"
        )

        print(chunk["text"][:500])


if __name__ == "__main__":
    main()
