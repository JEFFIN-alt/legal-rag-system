import re
from collections import Counter


NOISE_PATTERNS = [
    r"^www\.",
    r"^https?://",
    r"all rights reserved",
    r"copyright",
]


def clean_text(text: str) -> str:
    """
    Remove obvious line-level noise while preserving useful content.
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Remove obvious noise lines.
        if any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in NOISE_PATTERNS
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def noise_score(text: str) -> int:
    """
    Estimate how likely a page is to be non-content noise.

    Higher score = more likely to be noise.
    """

    text_lower = text.lower()

    score = 0

    # Publisher/contact indicators.
    noise_keywords = [
        "s. chand",
        "head office",
        "branch office",
        "marketing office",
        "phone:",
        "fax:",
        "e-mail:",
        "isbn",
        "printed at",
        "published by",
    ]

    for keyword in noise_keywords:
        if keyword in text_lower:
            score += 1

    # Excessive contact information.
    phone_numbers = re.findall(r"\b\d{6,}\b", text)
    email_addresses = re.findall(r"\S+@\S+", text)

    if len(phone_numbers) >= 3:
        score += 2

    if len(email_addresses) >= 2:
        score += 2

    # Excessive address-like content.
    address_keywords = [
        "road",
        "street",
        "nagar",
        "floor",
        "p.o.",
        "pin",
    ]

    address_hits = sum(
        text_lower.count(keyword)
        for keyword in address_keywords
    )

    if address_hits >= 5:
        score += 2

    return score


def clean_pages(pages):
    cleaned_pages = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        if len(cleaned_text.strip()) < 50:
            continue

        score = noise_score(cleaned_text)

        # Skip pages that strongly resemble publisher/front-matter noise.
        if score >= 5:
            continue

        cleaned_pages.append({
            "source": page["source"],
            "page": page["page"],
            "text": cleaned_text
        })

    return cleaned_pages
