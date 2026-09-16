from retriever import HybridRetriever
from reranker import Reranker


def main():

    # Initialize retrieval and reranking systems.
    retriever = HybridRetriever()
    reranker = Reranker()

    # Get user query.
    query = input("\nAsk Cognivault: ")

    print("\nRetrieving candidates...")

    # Hybrid retrieval gives us the best 10 candidates.
    candidates = retriever.retrieve(
        query,
        top_k=10
    )

    print(f"Retrieved {len(candidates)} candidates.")

    print("\nReranking candidates...")

    # Cross-encoder reranks the candidates.
    results = reranker.rerank(
        query,
        candidates,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n#{rank} "
            f"| Page {result['page']} "
            f"| Chunk {result['chunk_id']} "
            f"| Rerank score "
            f"{result['rerank_score']:.4f}"
        )

        print("-" * 70)
        print(result["text"])


if __name__ == "__main__":
    main()
