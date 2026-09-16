from pathlib import Path
import pymupdf


DOCUMENTS_DIR = Path("data/documents")


def extract_text(pdf_path: Path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "source": pdf_path.name,
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return pages


def main():
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s)\n")

    for pdf_path in pdf_files:

        pages = extract_text(pdf_path)

        print("=" * 70)
        print(f"DOCUMENT : {pdf_path.name}")
        print(f"PAGES    : {len(pages)}")
        print("=" * 70)

        for page in pages[:2]:
            print(f"\n--- Page {page['page']} ---")
            print(page["text"][:1000])

        print()


if __name__ == "__main__":
    main()

