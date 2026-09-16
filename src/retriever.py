from pathlib import Path
import json

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "cognivault_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


class HybridRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Loading all legal chunks...")

        self.chunks = []

        chunk_files = sorted(
            PROCESSED_DIR.glob("*_chunks.json")
        )

        for file in chunk_files:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                file_chunks = json.load(f)

            self.chunks.extend(file_chunks)

            print(
                f"  {file.name}: "
                f"{len(file_chunks)} chunks"
            )

        print(
            f"Total BM25 chunks: "
            f"{len(self.chunks)}"
        )

        # BM25 corpus
        self.tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("Connecting to ChromaDB...")

        client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        self.collection = client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"ChromaDB chunks: "
            f"{self.collection.count()}"
        )


    def semantic_search(
        self,
        query,
        top_k=10
    ):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k
        )

        output = []

        for i in range(
            len(results["documents"][0])
        ):

            metadata = results["metadatas"][0][i]

            output.append({
                "source": metadata["source"],
                "page": metadata.get("page"),
                "page_start": metadata.get(
                    "page_start",
                    metadata.get("page")
                ),
                "page_end": metadata.get(
                    "page_end",
                    metadata.get("page")
                ),
                "chunk_id": metadata["chunk_id"],
                "text": results["documents"][0][i],
                "semantic_rank": i + 1,
                "semantic_distance":
                    results["distances"][0][i]
            })

        return output


    def bm25_search(
        self,
        query,
        top_k=10
    ):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(
            tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        output = []

        for rank, index in enumerate(
            ranked_indices,
            start=1
        ):

            chunk = self.chunks[index]

            output.append({
                "source": chunk["source"],
                "page": chunk.get("page"),
                "page_start": chunk.get(
                    "page_start",
                    chunk.get("page")
                ),
                "page_end": chunk.get(
                    "page_end",
                    chunk.get("page")
                ),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "bm25_rank": rank,
                "bm25_score": float(
                    scores[index]
                )
            })

        return output


    def rrf_fusion(
        self,
        semantic_results,
        bm25_results,
        top_k=10,
        rrf_k=60
    ):

        fused = {}

        # Semantic contribution
        for result in semantic_results:

            key = (
                result["source"],
                result["chunk_id"]
            )

            if key not in fused:
                fused[key] = {
                    **result,
                    "rrf_score": 0.0
                }

            fused[key]["rrf_score"] += (
                1 / (
                    rrf_k
                    + result["semantic_rank"]
                )
            )

        # BM25 contribution
        for result in bm25_results:

            key = (
                result["source"],
                result["chunk_id"]
            )

            if key not in fused:
                fused[key] = {
                    **result,
                    "rrf_score": 0.0
                }

            fused[key]["rrf_score"] += (
                1 / (
                    rrf_k
                    + result["bm25_rank"]
                )
            )

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return ranked[:top_k]


    def search(
        self,
        query,
        top_k=10
    ):

        semantic_results = self.semantic_search(
            query,
            top_k
        )

        bm25_results = self.bm25_search(
            query,
            top_k
        )

        fused_results = self.rrf_fusion(
            semantic_results,
            bm25_results,
            top_k
        )

        return fused_results
