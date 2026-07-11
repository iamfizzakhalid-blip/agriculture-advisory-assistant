from pathlib import Path
import csv
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = PROJECT_ROOT / "data" / "inventory.csv"

def detect_source(filename: str) -> str:
    """
    Detects source organization from filename.
    """
    name = filename.lower()

    if "parc" in name:
        return "PARC"

    elif "fao" in name:
        return "FAO"

    elif "uaf" in name:
        return "UAF"

    elif "punjab" in name or "extension" in name or "agriculture" in name:
        return "Punjab Agriculture Department"

    elif "pmd" in name or "meteorological" in name:
        return "Pakistan Meteorological Department"

    else:
        return "Unknown"


def count_words(text: str) -> int:
    """
    Counts words using regex.
    """

    words = re.findall(r"\b\w+\b", text)
    return len(words)

def main():

    if not CLEAN_DIR.exists():
        print(f"\nERROR: Folder not found:\n{CLEAN_DIR}")
        return

    inventory = []

    crop_folders = sorted(
        [
            folder
            for folder in CLEAN_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    for crop_folder in crop_folders:

        crop = crop_folder.name.capitalize()

        txt_files = sorted(crop_folder.glob("*.txt"))

        for file in txt_files:

            try:

                text = file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                word_count = count_words(text)

                source = detect_source(file.name)

                inventory.append(
                    {
                        "filename": file.name,
                        "crop": crop,
                        "word_count": word_count,
                        "source": source,
                    }
                )

            except Exception as e:
                print(f"Skipping {file.name}: {e}")

    with open(
        OUTPUT_FILE,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filename",
                "crop",
                "word_count",
                "source",
            ],
        )

        writer.writeheader()

        writer.writerows(inventory)

    print("=" * 50)
    print("Inventory Created Successfully")
    print("=" * 50)
    print(f"Output File : {OUTPUT_FILE}")
    print(f"Total Files : {len(inventory)}")

    crops = {}

    for item in inventory:
        crops[item["crop"]] = crops.get(item["crop"], 0) + 1

    print("\nFiles Per Crop:")

    for crop, count in sorted(crops.items()):
        print(f"  {crop:<12} : {count}")

    print("=" * 50)


if __name__ == "__main__":
    main()