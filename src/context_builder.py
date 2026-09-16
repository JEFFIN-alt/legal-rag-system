class ContextBuilder:
    def __init__(self, chunks):
        self.chunks = chunks

        self.lookup = {
            (chunk["source"], chunk["chunk_id"]): chunk
            for chunk in chunks
        }

    def build(self, results, window=1):
        if not results:
            return []

        expanded = []

        # Primary retrieved results
        for result in results:
            expanded.append(result)

        # Expand around the highest-ranked result
        primary = results[0]

        source = primary["source"]
        chunk_id = primary["chunk_id"]

        for offset in range(-window, window + 1):
            if offset == 0:
                continue

            nearby_id = chunk_id + offset
            key = (source, nearby_id)

            if key in self.lookup:
                nearby = self.lookup[key]

                expanded.append({
                    "chunk_id": nearby["chunk_id"],
                    "source": nearby["source"],
                    "page": nearby.get("page"),
                    "page_start": nearby.get(
                        "page_start",
                        nearby.get("page")
                    ),
                    "page_end": nearby.get(
                        "page_end",
                        nearby.get("page")
                    ),
                    "text": nearby["text"],
                    "retrieval_score": primary.get(
                        "retrieval_score", 0
                    ),
                    "rerank_score": primary.get(
                        "rerank_score", 0
                    )
                })

        # Remove duplicates
        unique = {}

        for chunk in expanded:
            key = (
                chunk["source"],
                chunk["chunk_id"]
            )
            unique[key] = chunk

        return list(unique.values())
