from pathlib import Path
import re
import json
import pymupdf

from cleaner import clean_pages


DOCUMENTS_DIR = Path("data/documents")
PROCESSED_DIR = Path("data/processed")

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300
MIN_CHUNK_LENGTH = 100


def extract_pages(pdf_path: Path):

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if text:

            pages.append({
                "source": pdf_path.name,
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


def split_into_sentences(text):

    # Normalize whitespace while preserving readable text.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Split on normal sentence endings.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_document_stream(pages):

    """
    Convert all cleaned pages into one continuous
    sequence of sentences.

    Page metadata is preserved for every sentence.
    """

    stream = []

    for page in pages:

        sentences = split_into_sentences(
            page["text"]
        )

        for sentence in sentences:

            stream.append({
                "text": sentence,
                "source": page["source"],
                "page": page["page"]
            })

    return stream


def create_chunks(stream):

    """
    Create chunks across page boundaries.

    Unlike the previous version, a page ending does
    NOT force a chunk boundary.
    """

    chunks = []
    chunk_id = 0

    current = []
    current_length = 0

    for item in stream:

        sentence = item["text"]
        sentence_length = len(sentence)

        # If this sentence would make the chunk too large,
        # save the current chunk first.
        if (
            current
            and current_length + sentence_length > CHUNK_SIZE
        ):

            chunk_text = " ".join(
                item["text"]
                for item in current
            ).strip()

            if len(chunk_text) >= MIN_CHUNK_LENGTH:

                pages = [
                    item["page"]
                    for item in current
                ]

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": current[0]["source"],
                    "page": pages[0],
                    "page_start": min(pages),
                    "page_end": max(pages),
                    "text": chunk_text
                })

                chunk_id += 1

            # Build overlap from the end of the
            # previous chunk.
            overlap = []
            overlap_length = 0

            for previous in reversed(current):

                if (
                    overlap_length
                    + len(previous["text"])
                    > CHUNK_OVERLAP
                ):
                    break

                overlap.insert(
                    0,
                    previous
                )

                overlap_length += len(
                    previous["text"]
                )

            current = overlap
            current_length = overlap_length

        current.append(item)
        current_length += sentence_length

    # Save final chunk.
    if current:

        chunk_text = " ".join(
            item["text"]
            for item in current
        ).strip()

        if len(chunk_text) >= MIN_CHUNK_LENGTH:

            pages = [
                item["page"]
                for item in current
            ]

            chunks.append({
                "chunk_id": chunk_id,
                "source": current[0]["source"],
                "page": pages[0],
                "page_start": min(pages),
                "page_end": max(pages),
                "text": chunk_text
            })

    return chunks


def save_chunks(chunks, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("No PDF files found.")

        return

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: "
            f"{pdf_path.name}"
        )

        # --------------------------------------------------
        # 1. Extract pages
        # --------------------------------------------------

        pages = extract_pages(
            pdf_path
        )

        # --------------------------------------------------
        # 2. Clean pages
        # --------------------------------------------------

        cleaned_pages = clean_pages(
            pages
        )

        # --------------------------------------------------
        # 3. Convert pages into continuous stream
        # --------------------------------------------------

        stream = create_document_stream(
            cleaned_pages
        )

        # --------------------------------------------------
        # 4. Create cross-page chunks
        # --------------------------------------------------

        chunks = create_chunks(
            stream
        )

        # --------------------------------------------------
        # 5. Save
        # --------------------------------------------------

        output_path = (
            PROCESSED_DIR
            / f"{pdf_path.stem}_chunks.json"
        )

        save_chunks(
            chunks,
            output_path
        )

        print(
            f"Pages extracted      : "
            f"{len(pages)}"
        )

        print(
            f"Pages after cleaning : "
            f"{len(cleaned_pages)}"
        )

        print(
            f"Sentences            : "
            f"{len(stream)}"
        )

        print(
            f"Chunks               : "
            f"{len(chunks)}"
        )

        print(
            f"Saved to             : "
            f"{output_path}"
        )

        # --------------------------------------------------
        # Show cross-page chunks
        # --------------------------------------------------

        cross_page = [
            chunk
            for chunk in chunks
            if chunk["page_start"]
            != chunk["page_end"]
        ]

        print(
            f"Cross-page chunks    : "
            f"{len(cross_page)}"
        )


if __name__ == "__main__":
    main()
