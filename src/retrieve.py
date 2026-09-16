import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cognivault_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Indexed chunks: {collection.count()}"
    )

    while True:

        query = input(
            "\nAsk Cognivault (type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        print("\nSearching...")

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )

        print("\n" + "=" * 70)
        print("TOP RETRIEVED CHUNKS")
        print("=" * 70)

        for i in range(len(results["documents"][0])):

            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]

            page_start = metadata.get(
                "page_start",
                metadata.get("page", "?")
            )

            page_end = metadata.get(
                "page_end",
                page_start
            )

            if page_start == page_end:
                page_display = str(page_start)
            else:
                page_display = (
                    f"{page_start}-{page_end}"
                )

            print(f"\nResult #{i + 1}")
            print("-" * 70)

            print(
                f"Source   : "
                f"{metadata['source']}"
            )

            print(
                f"Pages    : "
                f"{page_display}"
            )

            print(
                f"Chunk ID : "
                f"{metadata['chunk_id']}"
            )

            print(
                f"Distance : "
                f"{distance:.4f}"
            )

            print("-" * 70)

            print(document)


if __name__ == "__main__":
    main()
