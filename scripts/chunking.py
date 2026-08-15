from pathlib import Path


CHUNK_SIZE = 300
STEP_SIZE = 250
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "clean"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "chunks"


def split_into_chunks(text, chunk_size=CHUNK_SIZE, step_size=STEP_SIZE):
    words = text.split()
    chunks = []

    for start in range(0, len(words), step_size):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


def get_crop_directories(input_root):
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def has_existing_chunks(file_path, output_parent):
    """
    Return True if this source .txt file has already been chunked
    (i.e. at least one matching '<stem>_chunk_*.txt' file exists in the
    output folder). Used to skip files that were chunked previously.
    """
    if not output_parent.exists():
        return False
    return any(output_parent.glob(f"{file_path.stem}_chunk_*.txt"))


def write_chunks_for_file(file_path, crop_output_dir, crop_dir):
    text = file_path.read_text(encoding="utf-8")
    chunks = split_into_chunks(text)
    output_parent = crop_output_dir / file_path.relative_to(crop_dir).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    created_chunks = 0
    for index, chunk in enumerate(chunks, start=1):
        output_file = output_parent / f"{file_path.stem}_chunk_{index:03d}.txt"
        output_file.write_text(chunk, encoding="utf-8")
        created_chunks += 1

    return created_chunks


def process_crop_directory(crop_dir, output_root):
    files_processed = 0
    files_skipped = 0
    chunks_created = 0
    crop_output_dir = output_root / crop_dir.name

    for file_path in sorted(crop_dir.rglob("*.txt")):
        if not file_path.is_file():
            continue

        output_parent = crop_output_dir / file_path.relative_to(crop_dir).parent

        # ── Skip files that already have chunks ──────────────────────────
        # Reruns only chunk NEW/changed .txt files, so previously created
        # chunk files are left untouched.
        if has_existing_chunks(file_path, output_parent):
            print(f"  -> Skipping {file_path.relative_to(crop_dir)} (already chunked)")
            files_skipped += 1
            continue

        files_processed += 1
        chunks_created += write_chunks_for_file(file_path, crop_output_dir, crop_dir)

    return files_processed, chunks_created, files_skipped


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    crops_processed = 0
    files_processed = 0
    files_skipped = 0
    chunks_created = 0

    for crop_dir in get_crop_directories(INPUT_ROOT):
        crop_files, crop_chunks, crop_skipped = process_crop_directory(crop_dir, OUTPUT_ROOT)
        files_skipped += crop_skipped
        if crop_files:
            crops_processed += 1
            files_processed += crop_files
            chunks_created += crop_chunks

    print("====================================")
    print("Chunking Complete")
    print("====================================")
    print(f"Crops processed : {crops_processed}")
    print(f"Files processed : {files_processed}  (new)")
    print(f"Files skipped   : {files_skipped}  (already chunked)")
    print(f"Chunks created  : {chunks_created}")
    print(f"Output folder   : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("====================================")


if __name__ == "__main__":
    main()