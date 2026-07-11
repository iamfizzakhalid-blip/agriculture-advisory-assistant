import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF

# ══════════════════════════════════════════════════════════════════════════════
# Constants & paths
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR: Path = PROJECT_ROOT / "data" / "clean"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# [Step 7] OCR correction dictionary  —  easy to extend, just add entries
# ══════════════════════════════════════════════════════════════════════════════
OCR_CORRECTIONS: Dict[str, str] = {
    # ── Agricultural / scientific terms ──
    "hervibores":      "herbivores",
    "hervibore":       "herbivore",
    "analyis":         "analysis",
    "enviroment":      "environment",
    "enviromental":    "environmental",
    "agroecosytem":    "agroecosystem",
    "agroecosytems":   "agroecosystems",
    "photosythesis":   "photosynthesis",
    "photosythetic":   "photosynthetic",
    "transpiraton":    "transpiration",
    "nutirent":        "nutrient",
    "nutirents":       "nutrients",
    "fertlizer":       "fertilizer",
    "fertlizers":      "fertilizers",
    "fertilzer":       "fertilizer",
    "fertilzers":      "fertilizers",
    "pesticde":        "pesticide",
    "pesticdes":       "pesticides",
    "insecticde":      "insecticide",
    "insecticdes":     "insecticides",
    "germiantion":     "germination",
    "germinaion":      "germination",
    "vegitative":      "vegetative",
    "vegetaive":       "vegetative",
    "polliation":      "pollination",
    "polination":      "pollination",
    "irigation":       "irrigation",
    "irrigaton":       "irrigation",
    "drougth":         "drought",
    "harvsting":       "harvesting",
    "harvestng":       "harvesting",
    "pathoen":         "pathogen",
    "pathoens":        "pathogens",
    "funigcide":       "fungicide",
    "funigcides":      "fungicides",
    "funguicide":      "fungicide",
    "neamtode":        "nematode",
    "neamtodes":       "nematodes",
    "bollwrom":        "bollworm",
    "bollwroms":       "bollworms",
    "heribcide":       "herbicide",
    "heribcides":      "herbicides",
    "chlrophyll":      "chlorophyll",
    "nitrgen":         "nitrogen",
    "phosporus":       "phosphorus",
    "potasium":        "potassium",
    "monocultre":      "monoculture",
    "intercropping":   "intercropping",        # correct — keep
    "bioligical":      "biological",
    "agiculture":      "agriculture",
    "agicultural":     "agricultural",
    # ── General English ──
    "teh":             "the",
    "hte":             "the",
    "adn":             "and",
    "nad":             "and",
    "wiht":            "with",
    "htis":            "this",
    "taht":            "that",
    "thier":           "their",
    "recieve":         "receive",
    "recieves":        "receives",
    "occurence":       "occurrence",
    "occurences":      "occurrences",
    "managment":       "management",
    "managemnt":       "management",
    "developement":    "development",
    "goverment":       "government",
    "govenment":       "government",
    "reserach":        "research",
    "reserch":         "research",
    "yeild":           "yield",
    "yeilds":          "yields",
    "seperate":        "separate",
    "seperately":      "separately",
    "temperture":      "temperature",
    "temperatue":      "temperature",
    "measurment":      "measurement",
    "measurments":     "measurements",
    "significnt":      "significant",
    "signficant":      "significant",
    "diffrence":       "difference",
    "diffrences":      "differences",
    "comparision":     "comparison",
    "recomendation":   "recommendation",
    "recomendations":  "recommendations",
    "recommandation":  "recommendation",
    "recommandations": "recommendations",
    "necessory":       "necessary",
    "neccesary":       "necessary",
    "neccessary":      "necessary",
    "availble":        "available",
    "availabel":       "available",
    "aproximately":    "approximately",
    "aproximate":      "approximate",
}

# Pre-compile a single regex for OCR replacement (longest keys first to
# prevent partial matches).
_ocr_keys_sorted = sorted(OCR_CORRECTIONS.keys(), key=len, reverse=True)
_OCR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ocr_keys_sorted) + r")\b",
    re.IGNORECASE,
)

# ══════════════════════════════════════════════════════════════════════════════
# Pre-compiled regex patterns for page artifacts  [Step 3 / 10]
# ══════════════════════════════════════════════════════════════════════════════

# Standalone date on its own line  (03/05/09, 2024-01-15, …)
_DATE_LINE = re.compile(r"(?m)^\s*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*$")

# "Page 15" / "Page 15 of 42"
_PAGE_LABEL = re.compile(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$")

# "15 of 42"
_X_OF_Y = re.compile(r"(?m)^\s*\d+\s+of\s+\d+\s*$")

# "– 3 –" / "- 3 -"
_DASH_PAGE = re.compile(r"(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$")

# "| 3 |"
_PIPE_PAGE = re.compile(r"(?m)^\s*\|\s*\d+\s*\|\s*$")

# Standalone number (1-5 digits)
_STANDALONE_NUM = re.compile(r"(?m)^\s*\d{1,5}\s*$")

# "Confidential" / "DRAFT" / "RESTRICTED"
_CONFIDENTIAL = re.compile(r"(?im)^\s*(confidential|draft|restricted)\s*$")

# Header / footer line that ends with a date + optional page number
# e.g.  "field guide exercises for ipm in cotton: research methods.03/05/09 3"
_HEADER_WITH_DATE = re.compile(
    r"(?im)^\s*.{5,}\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*\d*\s*$"
)

# Footer line that starts with a page number + content + date
# e.g.  "4 field guide exercises for ipm in cotton: research methods. 03/05/09"
_FOOTER_WITH_DATE = re.compile(
    r"(?im)^\s*\d{1,5}\s+.{5,}\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*$"
)


# ══════════════════════════════════════════════════════════════════════════════
# Heading / list detection helpers  [used by Step 5 to avoid bad merges]
# ══════════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(
    r"^(?:"
    r"\s*#{1,6}\s"                # Markdown heading
    r"|\s*\d+(\.\d+)*\.?\s"      # Numbered heading  1.  1.2  1.2.3.
    r"|\s*[A-Z][A-Z\s]{3,}$"     # ALL-CAPS line ≥ 4 chars
    r")",
    re.MULTILINE,
)

_LIST_RE = re.compile(
    r"^\s*(?:[-•●◦▪*]|\d+[.)]\s|\([a-z]\)|\([ivxlc]+\))\s",
    re.MULTILINE | re.IGNORECASE,
)


def _is_heading_or_list(line: str) -> bool:
    """Return True if *line* looks like a heading, bullet, or numbered item."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.isupper() and len(stripped) >= 4:
        return True
    if _HEADING_RE.match(line):
        return True
    if _LIST_RE.match(line):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING STEPS  —  each is a small, testable function
# ══════════════════════════════════════════════════════════════════════════════

# ── Steps 1 & 2: Remove repeated headers / footers ──────────────────────────

def _detect_repeated_lines(
    pages_text: List[str],
    threshold: int = 3,
) -> Set[str]:
    """
    Return normalised lines that appear on ≥ *threshold* pages.

    These are almost certainly repeated headers or footers injected by
    the PDF layout engine.
    """
    line_counter: Counter = Counter()
    for page in pages_text:
        # De-duplicate within a single page so a line printed twice on
        # one page does not inflate the count.
        unique = {
            ln.strip()
            for ln in page.splitlines()
            if ln.strip() and len(ln.strip()) < 200
        }
        line_counter.update(unique)

    num_pages = len(pages_text)
    repeated: Set[str] = set()
    for line, count in line_counter.items():
        # Lines that repeat across many pages (absolute count)
        if count >= threshold and len(line) < 120:
            repeated.add(line)
        # Lines that appear on > 30 % of pages (ratio-based)
        if num_pages > 4 and count / num_pages >= 0.3 and len(line) < 120:
            repeated.add(line)

    return repeated


def remove_repeated_headers_footers(
    text: str,
    pages_text: List[str],
) -> str:
    """Step 1 & 2 — strip lines detected as repeated headers / footers."""
    repeated = _detect_repeated_lines(pages_text)
    if not repeated:
        return text

    lines = text.splitlines()
    cleaned = [ln for ln in lines if ln.strip() not in repeated]
    return "\n".join(cleaned)


# ── Step 3: Remove standalone page numbers ───────────────────────────────────

def remove_standalone_page_numbers(text: str) -> str:
    """Step 3 — remove lines that contain only page-number patterns."""
    text = _PAGE_LABEL.sub("", text)
    text = _X_OF_Y.sub("", text)
    text = _DASH_PAGE.sub("", text)
    text = _PIPE_PAGE.sub("", text)
    text = _STANDALONE_NUM.sub("", text)
    return text


# ── Step 4: Remove excessive blank lines ─────────────────────────────────────

def collapse_blank_lines(text: str) -> str:
    """Step 4 — never allow more than one consecutive blank line."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ── Step 5: Merge wrapped lines ─────────────────────────────────────────────

def merge_wrapped_lines(text: str) -> str:
    """
    Step 5 — join lines that were split by the PDF extractor.

    Heuristic:
      • current line does NOT end with sentence-ending punctuation
      • current line is NOT a heading or list item
      • next line starts with a lowercase letter  (strong continuation signal)
      • current line is shorter than 120 chars (not already full-width)

    Paragraph breaks and headings are preserved.
    """
    lines = text.split("\n")
    merged: List[str] = []
    i = 0

    while i < len(lines):
        current = lines[i].rstrip()

        # Blank line — keep as-is
        if not current.strip():
            merged.append(current)
            i += 1
            continue

        # Try merging with following continuation lines
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_stripped = nxt.strip()

            # Stop if next line is blank
            if not nxt_stripped:
                break
            # Stop if next line is a heading / list item
            if _is_heading_or_list(nxt):
                break
            # Stop if current line ends with terminal punctuation
            if current.strip() and current.strip()[-1] in ".!?:;":
                break
            # Stop if current line is a heading / list item
            if _is_heading_or_list(current):
                break
            # Stop if current line is already long
            if len(current.strip()) > 120:
                break
            # Only merge when next line starts with a lowercase letter
            if nxt_stripped and nxt_stripped[0].islower():
                current = current.rstrip() + " " + nxt_stripped
                i += 1
            else:
                break

        merged.append(current)
        i += 1

    return "\n".join(merged)


# ── Step 6: Normalize whitespace ─────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """
    Step 6 — tabs → spaces, collapse multi-spaces, strip trailing spaces.
    Preserves line breaks and UTF-8 encoding.
    """
    text = text.replace("\t", " ")
    # Collapse horizontal whitespace (not newlines) to a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove trailing spaces per line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    return text


# ── Step 7: Fix common OCR mistakes ─────────────────────────────────────────

def fix_ocr_mistakes(text: str) -> str:
    """
    Step 7 — replace known OCR errors using OCR_CORRECTIONS.

    Matching is case-insensitive.  The original casing style
    (lower / UPPER / Title) is preserved in the replacement.
    """
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        replacement = OCR_CORRECTIONS.get(word.lower(), word)
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement

    return _OCR_PATTERN.sub(_replace, text)


# ── Step 8: Remove unnecessary tables ────────────────────────────────────────

def _is_table_line(line: str) -> bool:
    """Return True if *line* looks like a row of a data table."""
    stripped = line.strip()
    if not stripped:
        return False
    # Pure separator rows  (----  ====  |---|  ____)
    if re.match(r"^[\-=_|+\s]+$", stripped):
        return True
    # Multi-column data: split on ≥ 2 spaces or tabs
    cells = re.split(r"\s{2,}|\t", stripped)
    if len(cells) >= 2:
        numeric = sum(
            1 for c in cells if re.match(r"^[\d.,/%$()+-]+$", c.strip())
        )
        if numeric / len(cells) >= 0.4:
            return True
    return False


def _is_numeric_table(block: List[str]) -> bool:
    """Return True if most cells in *block* are numeric / very short labels."""
    total = 0
    numeric = 0
    for line in block:
        for cell in re.split(r"\s{2,}|\t", line.strip()):
            cell = cell.strip()
            if not cell:
                continue
            total += 1
            if re.match(r"^[\d.,/%$()+-]+$", cell) or len(cell) <= 3:
                numeric += 1
    return total > 0 and numeric / total >= 0.5


def remove_unnecessary_tables(text: str) -> str:
    """
    Step 8 — remove table blocks that are mostly numeric / separator data.

    If a contiguous block of ≥ 3 "table-like" lines is predominantly
    numeric, it is removed.  Blocks with explanatory text are kept.
    """
    lines = text.split("\n")
    result: List[str] = []
    i = 0

    while i < len(lines):
        if _is_table_line(lines[i]):
            block: List[str] = []
            while i < len(lines) and _is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            if len(block) >= 3 and _is_numeric_table(block):
                result.append("")          # keep a blank line for spacing
            else:
                result.extend(block)
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result)


# ── Step 9: Remove repeated document titles ──────────────────────────────────

def remove_repeated_document_titles(text: str) -> str:
    """
    Step 9 — if the first non-empty lines repeat ≥ 3 times in the
    document, remove all duplicates after the first occurrence.
    """
    lines = text.split("\n")

    # Gather candidate titles from the top of the file
    candidates: List[str] = []
    for ln in lines:
        if ln.strip():
            candidates.append(ln.strip())
            if len(candidates) >= 3:
                break

    if not candidates:
        return text

    # Keep only candidates that repeat ≥ 3 times
    title_set: Set[str] = set()
    for cand in candidates:
        if sum(1 for ln in lines if ln.strip() == cand) >= 3:
            title_set.add(cand)

    if not title_set:
        return text

    # Remove all but the first occurrence of each repeated title
    seen: Dict[str, bool] = {t: False for t in title_set}
    cleaned: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s in seen:
            if not seen[s]:
                seen[s] = True          # keep first occurrence
                cleaned.append(ln)
            # else: skip duplicate
        else:
            cleaned.append(ln)

    return "\n".join(cleaned)


# ── Step 10: Remove page artifacts ───────────────────────────────────────────

def remove_page_artifacts(text: str) -> str:
    """
    Step 10 — remove standalone dates, "Page X", "X of Y",
    "Confidential", and header/footer lines with embedded dates.
    """
    text = _DATE_LINE.sub("", text)
    text = _CONFIDENTIAL.sub("", text)
    text = _HEADER_WITH_DATE.sub("", text)
    text = _FOOTER_WITH_DATE.sub("", text)
    return text


# ── Step 11: Normalize punctuation ───────────────────────────────────────────

def normalize_punctuation(text: str) -> str:
    """
    Step 11 — fix repeated / broken punctuation from OCR.

      • 4+ dots  → ellipsis (…)
      • exactly 2 dots  → single period
      • repeated commas  → single comma
      • 3+ hyphens  → single hyphen
      • 3+ underscores  → removed
      • space before . , ; : ! ?  → removed
      • missing space after sentence-ending punctuation  → added
    """
    text = re.sub(r"\.{4,}", "...", text)              # 4+ dots → ellipsis
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)      # 2 dots → 1
    text = re.sub(r",{2,}", ",", text)                  # repeated commas
    text = re.sub(r"-{3,}", "-", text)                  # 3+ hyphens
    text = re.sub(r"_{3,}", "", text)                   # 3+ underscores
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)        # space before punct
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)    # missing space after
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Master cleaning pipeline
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(pages_text: List[str]) -> str:
    """
    Apply all 13 cleaning steps to the per-page text list and return
    a single RAG-ready string.

    Step 12 (preserve useful content) is enforced implicitly by the
    conservative heuristics used in every step above.

    The ordering is deliberate — page-level artefacts are removed first
    so that later steps (merging, whitespace) operate on cleaner input.
    """
    # Join pages into one text block
    text = "\n".join(pages_text)

    # Normalise line endings  (CRLF → LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Steps 1 & 2 — repeated headers / footers  (needs pages_text)
    text = remove_repeated_headers_footers(text, pages_text)

    # Step 3 — standalone page numbers
    text = remove_standalone_page_numbers(text)

    # Step 10 — page artifacts  (dates, "Page X", "Confidential", …)
    text = remove_page_artifacts(text)

    # Step 9 — repeated document titles
    text = remove_repeated_document_titles(text)

    # Step 8 — unnecessary numeric tables
    text = remove_unnecessary_tables(text)

    # Step 7 — OCR mistake corrections
    text = fix_ocr_mistakes(text)

    # Step 11 — punctuation normalization
    text = normalize_punctuation(text)

    # Step 6 — whitespace normalization
    text = normalize_whitespace(text)

    # Step 5 — merge wrapped lines
    text = merge_wrapped_lines(text)

    # Step 4 — collapse excessive blank lines
    text = collapse_blank_lines(text)

    # Final trim
    text = text.strip() + "\n"

    return text


# ══════════════════════════════════════════════════════════════════════════════
# PDF extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: Path) -> Optional[List[str]]:
    """
    Extract raw text from every page of *pdf_path*.

    Returns a list of strings (one per page), or None on failure.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        logger.exception("Failed to open PDF: %s", pdf_path)
        return None

    pages_text: List[str] = []
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

    return pages_text


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline — process folders
# ══════════════════════════════════════════════════════════════════════════════

def process_crop_folder(crop_dir: Path) -> Tuple[int, int]:
    """
    Process all PDFs in a single crop folder.

    For each PDF:
      1. Extract per-page text  (Phase 1).
      2. Run the full cleaning pipeline  (Phase 2).
      3. Save UTF-8 .txt to  data/clean/<crop>/<filename>.txt.

    Returns (success_count, failure_count).
    """
    crop_name = crop_dir.name
    output_dir = CLEAN_DIR / crop_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    pdf_files = sorted(crop_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"  (no PDF files found in {crop_name}/)")
        return 0, 0

    for pdf_path in pdf_files:
        relative_label = f"{crop_name}/{pdf_path.relative_to(crop_dir)}"
        print(f"  -> Processing {relative_label} ... ", end="", flush=True)
        logger.info("Processing %s", relative_label)

        # Phase 1 - extraction
        pages_text = extract_pdf_text(pdf_path)
        if pages_text is None:
            print("FAILED (extraction)")
            logger.error("FAILED - %s", relative_label)
            failed += 1
            continue

        raw_chars = sum(len(p) for p in pages_text)

        # Phase 2 - cleaning
        try:
            cleaned = clean_text(pages_text)
        except Exception:
            logger.exception("Cleaning failed for %s", relative_label)
            print("FAILED (cleaning)")
            failed += 1
            continue

        # Mirror any sub-directory structure inside the crop folder
        relative_sub = pdf_path.relative_to(crop_dir).parent
        dest_dir = output_dir / relative_sub
        dest_dir.mkdir(parents=True, exist_ok=True)

        out_file = dest_dir / pdf_path.with_suffix(".txt").name
        out_file.write_text(cleaned, encoding="utf-8")

        clean_chars = len(cleaned)
        reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars else 0
        print(
            f"OK  ({raw_chars:,} -> {clean_chars:,} chars, "
            f"{reduction:.1f}% reduction)"
        )
        logger.info(
            "Saved %s  (%d -> %d chars, %.1f%% reduction)",
            out_file.relative_to(PROJECT_ROOT),
            raw_chars,
            clean_chars,
            reduction,
        )
        success += 1

    return success, failed


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    """Iterate over every crop folder in data/raw/ and process all PDFs."""

    print("=" * 70)
    print("  Agricultural PDF -> Clean Text Pipeline")
    print(f"  Source : {RAW_DIR}")
    print(f"  Output : {CLEAN_DIR}")
    print("=" * 70)

    if not RAW_DIR.exists():
        logger.error("Raw data directory does not exist: %s", RAW_DIR)
        sys.exit(1)

    crop_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

    if not crop_dirs:
        logger.warning("No crop folders found in %s", RAW_DIR)
        print(f"\n[!] No crop folders found in {RAW_DIR}")
        return

    total_success = 0
    total_failed = 0
    start = time.time()

    for crop_dir in crop_dirs:
        print(f"\n-- Crop: {crop_dir.name} --")
        logger.info("=== Crop: %s ===", crop_dir.name)
        s, f = process_crop_folder(crop_dir)
        total_success += s
        total_failed += f

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Total files : {total_success + total_failed}")
    print(f"  Success     : {total_success}")
    print(f"  Failed      : {total_failed}")
    print("=" * 70)

    logger.info(
        "Done - %d files processed successfully, %d failed.",
        total_success,
        total_failed,
    )


if __name__ == "__main__":
    main()
