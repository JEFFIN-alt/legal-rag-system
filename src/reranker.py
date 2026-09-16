from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self):

        print("Loading cross-encoder...")

        self.model = CrossEncoder(
            MODEL_NAME
        )


    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        if not results:
            return []

        pairs = [
            (
                query,
                result["text"]
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            result = result.copy()

            result["rerank_score"] = float(
                score
            )

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
