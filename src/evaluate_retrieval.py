import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cognivault_documents"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


QUESTIONS = [
    {
        "question": "What is electric current?",
        "expected_terms": ["electric current", "flow of electrons"]
    },
    {
        "question": "What is Ohm's law?",
        "expected_terms": ["ohm", "voltage", "current", "resistance"]
    },
    {
        "question": "What factors affect the resistance of a conductor?",
        "expected_terms": ["length", "area", "resistance"]
    },
    {
        "question": "What is electrical power?",
        "expected_terms": ["electric power", "work", "time"]
    },
    {
        "question": "What is a DC circuit?",
        "expected_terms": ["d.c. circuit", "direct current"]
    },
    {
        "question": "What is electrical energy?",
        "expected_terms": ["electrical energy", "power", "time"]
    },
    {
        "question": "What is resistivity?",
        "expected_terms": ["resistivity", "resistance"]
    },
    {
        "question": "What are the different types of electric current?",
        "expected_terms": ["steady current", "varying current", "alternating current"]
    },
    {
        "question": "What is electric potential difference?",
        "expected_terms": ["potential difference", "voltage"]
    },
    {
        "question": "What is a series circuit?",
        "expected_terms": ["series circuit", "current"]
    }
]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    total_passed = 0

    print("\n" + "=" * 70)
    print("COGNIVAULT RETRIEVAL BENCHMARK")
    print("=" * 70)

    for index, item in enumerate(QUESTIONS, start=1):

        question = item["question"]
        expected_terms = item["expected_terms"]

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        combined_text = " ".join(documents).lower()

        matched_terms = [
            term
            for term in expected_terms
            if term.lower() in combined_text
        ]

        passed = len(matched_terms) > 0

        if passed:
            total_passed += 1

        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] Question {index}")
        print(f"Query: {question}")
        print(f"Matched terms: {matched_terms}")

        print("Retrieved pages:", [
            metadata["page"]
            for metadata in metadatas
        ])

    accuracy = total_passed / len(QUESTIONS)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULT")
    print("=" * 70)
    print(f"Questions : {len(QUESTIONS)}")
    print(f"Passed    : {total_passed}")
    print(f"Accuracy  : {accuracy:.2%}")


if __name__ == "__main__":
    main()
