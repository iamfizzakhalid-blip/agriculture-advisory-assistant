"""
extract_text.py — Generic PDF text extraction for all crop folders.

Traverses every crop folder inside data/raw/, extracts text from PDFs
using PyMuPDF (fitz), cleans the output, and saves UTF-8 .txt files
into the matching data/clean/<crop>/ directory.
"""

import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR: Path = PROJECT_ROOT / "data" / "clean"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "extract_text.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _detect_repeated_lines(pages_text: list[str], threshold: int = 3) -> set[str]:
    """Return lines that appear on *threshold* or more pages (headers/footers).

    Args:
        pages_text: Raw text extracted per page.
        threshold:  Minimum number of pages a line must appear on to be
                    considered a repeated header or footer.

    Returns:
        A set of normalised line strings to strip.
    """
    line_counter: Counter[str] = Counter()
    for page in pages_text:
        # Deduplicate within a single page so a line printed twice on one
        # page doesn't inflate the count.
        unique_lines = {line.strip() for line in page.splitlines() if line.strip()}
        line_counter.update(unique_lines)

    return {line for line, count in line_counter.items() if count >= threshold}


def _remove_page_numbers(text: str) -> str:
    """Strip common page-number patterns.

    Handles patterns such as:
        - "Page 3", "page 3 of 10", "– 3 –", "| 3 |"
        - Standalone numbers on their own line
    """
    # "Page X" / "Page X of Y" (case-insensitive, whole line)
    text = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text)
    # "– 3 –" or "- 3 -" style centered page numbers
    text = re.sub(r"(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", text)
    # "| 3 |" style
    text = re.sub(r"(?m)^\s*\|\s*\d+\s*\|\s*$", "", text)
    # Standalone number on its own line (1-5 digits)
    text = re.sub(r"(?m)^\s*\d{1,5}\s*$", "", text)
    return text


def clean_text(pages_text: list[str]) -> str:
    """Apply all cleaning steps and return the final text.

    Steps:
        1. Remove lines that are repeated headers / footers.
        2. Remove page-number patterns.
        3. Collapse excessive blank lines (3+ → 2).
        4. Collapse multiple spaces within lines.
        5. Strip leading/trailing whitespace.

    Args:
        pages_text: List of raw text strings, one per PDF page.

    Returns:
        Cleaned text as a single string.
    """
    repeated = _detect_repeated_lines(pages_text)

    cleaned_pages: list[str] = []
    for page in pages_text:
        lines = page.splitlines()
        lines = [line for line in lines if line.strip() not in repeated]
        cleaned_pages.append("\n".join(lines))

    text = "\n".join(cleaned_pages)

    # Page numbers
    text = _remove_page_numbers(text)

    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces → single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract and clean text from a single PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Cleaned text, or ``None`` if extraction failed entirely.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        logger.exception("Failed to open PDF: %s", pdf_path)
        return None

    pages_text: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text()
            if text:
                pages_text.append(text)
        except Exception:
            logger.warning(
                "Could not extract page %d from %s — skipping page.",
                page_num,
                pdf_path,
            )

    doc.close()

    if not pages_text:
        logger.warning("No text extracted from %s", pdf_path)
        return None

    return clean_text(pages_text)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_crop_folder(crop_dir: Path) -> tuple[int, int]:
    """Process all PDFs in a single crop folder.

    Args:
        crop_dir: Path to a crop folder inside ``data/raw/``.

    Returns:
        ``(success_count, failure_count)``
    """
    crop_name = crop_dir.name
    output_dir = CLEAN_DIR / crop_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for pdf_path in sorted(crop_dir.rglob("*.pdf")):
        # Build a relative label for progress messages
        relative_label = f"{crop_name}/{pdf_path.relative_to(crop_dir)}"
        print(f"Processing {relative_label}")
        logger.info("Processing %s", relative_label)

        text = extract_pdf_text(pdf_path)
        if text is None:
            logger.error("FAILED — %s", relative_label)
            failed += 1
            continue

        # Mirror any subdirectory structure inside the crop folder
        relative_sub = pdf_path.relative_to(crop_dir).parent
        dest_dir = output_dir / relative_sub
        dest_dir.mkdir(parents=True, exist_ok=True)

        out_file = dest_dir / pdf_path.with_suffix(".txt").name
        out_file.write_text(text, encoding="utf-8")
        print(f"Saved {out_file.relative_to(PROJECT_ROOT)}")
        logger.info("Saved %s", out_file.relative_to(PROJECT_ROOT))
        success += 1

    return success, failed


def main() -> None:
    """Entry point — iterate over every crop folder in ``data/raw/``."""
    if not RAW_DIR.exists():
        logger.error("Raw data directory does not exist: %s", RAW_DIR)
        sys.exit(1)

    crop_dirs = sorted(
        [d for d in RAW_DIR.iterdir() if d.is_dir()]
    )

    if not crop_dirs:
        logger.warning("No crop folders found in %s", RAW_DIR)
        return

    total_success = 0
    total_failed = 0

    for crop_dir in crop_dirs:
        logger.info("=== Crop: %s ===", crop_dir.name)
        s, f = process_crop_folder(crop_dir)
        total_success += s
        total_failed += f

    logger.info(
        "Done — %d files processed successfully, %d failed.",
        total_success,
        total_failed,
    )


if __name__ == "__main__":
    main()
