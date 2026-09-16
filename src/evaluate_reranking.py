from retriever import HybridRetriever
from reranker import Reranker


TOP_K = 5
CANDIDATE_K = 10


QUESTIONS = [
    {
        "question": "What is electric current?",
        "expected_terms": [
            "electric current",
            "flow of electrons"
        ]
    },
    {
        "question": "What is Ohm's law?",
        "expected_terms": [
            "ohm",
            "voltage",
            "current",
            "resistance"
        ]
    },
    {
        "question": "What factors affect the resistance of a conductor?",
        "expected_terms": [
            "length",
            "area",
            "resistance"
        ]
    },
    {
        "question": "What is electrical power?",
        "expected_terms": [
            "electric power",
            "work",
            "time"
        ]
    },
    {
        "question": "What is a DC circuit?",
        "expected_terms": [
            "d.c. circuit",
            "direct current"
        ]
    },
    {
        "question": "What is electrical energy?",
        "expected_terms": [
            "electrical energy",
            "power",
            "time"
        ]
    },
    {
        "question": "What is resistivity?",
        "expected_terms": [
            "resistivity",
            "resistance"
        ]
    },
    {
        "question": "What are the different types of electric current?",
        "expected_terms": [
            "steady current",
            "varying current",
            "alternating current"
        ]
    },
    {
        "question": "What is electric potential difference?",
        "expected_terms": [
            "potential difference",
            "voltage"
        ]
    },
    {
        "question": "What is a series circuit?",
        "expected_terms": [
            "series circuit",
            "current"
        ]
    }
]


def check_result(results, expected_terms):

    combined_text = " ".join(
        result["text"]
        for result in results
    ).lower()

    matched_terms = [
        term
        for term in expected_terms
        if term.lower() in combined_text
    ]

    return matched_terms


def main():

    print("=" * 70)
    print("COGNIVAULT RERANKING BENCHMARK")
    print("=" * 70)

    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever()

    print("\nInitializing reranker...")
    reranker = Reranker()

    passed = 0

    for index, item in enumerate(
        QUESTIONS,
        start=1
    ):

        question = item["question"]

        print("\n" + "-" * 70)
        print(f"Question {index}: {question}")
        print("-" * 70)

        # Hybrid retrieval.
        candidates = retriever.retrieve(
            question,
            top_k=CANDIDATE_K
        )

        # Cross-encoder reranking.
        results = reranker.rerank(
            question,
            candidates,
            top_k=TOP_K
        )

        matched_terms = check_result(
            results,
            item["expected_terms"]
        )

        if matched_terms:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        pages = [
            result["page"]
            for result in results
        ]

        print(f"Status        : {status}")
        print(f"Matched terms : {matched_terms}")
        print(f"Retrieved pages: {pages}")

        if results:
            print(
                f"Top result    : Page {results[0]['page']}"
            )

    accuracy = passed / len(QUESTIONS)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Questions : {len(QUESTIONS)}")
    print(f"Passed    : {passed}")
    print(f"Accuracy  : {accuracy:.2%}")

    print("\nPipeline:")
    print("Semantic/BM25 → RRF → Cross-Encoder Reranker")


if __name__ == "__main__":
    main()
