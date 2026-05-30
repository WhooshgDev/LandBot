from datasets import load_dataset
from pathlib import Path
import pandas as pd


def main():
    # =========================================================
    # Paths
    # =========================================================
    project_root = Path(__file__).resolve().parent.parent.parent

    output_path = project_root / "data" / "LandLawDocument.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Load streaming datasets
    # =========================================================
    print("Loading metadata dataset...")

    metadata_ds = load_dataset(
        "minhdoan17/vietnamese-legal-documents",
        "metadata",
        split="data",
        streaming=True,
    )

    print("Loading content dataset...")

    content_ds = load_dataset(
        "minhdoan17/vietnamese-legal-documents",
        "content",
        split="data",
        streaming=True,
    )

    # =========================================================
    # Filter metadata
    # =========================================================
    print("Filtering metadata...")

    metadata_map = {}

    for row in metadata_ds:

        title = str(row.get("title", "")).lower()
        legal_sectors = str(row.get("legal_sectors", "")).lower()
        issuance_date = str(row.get("issuance_date", ""))

        # Filter conditions
        if (
            "bất động sản" in legal_sectors
            and "đất" in title
        ):

            try:
                date = pd.to_datetime(issuance_date)

                if date >= pd.Timestamp("2010-01-01"):
                    metadata_map[row["id"]] = row

            except Exception:
                continue

    print(f"Filtered metadata rows: {len(metadata_map)}")

    # =========================================================
    # Match content by id
    # =========================================================
    print("Matching content dataset...")

    merged_rows = []

    matched = 0

    for row in content_ds:

        doc_id = row.get("id")

        if doc_id in metadata_map:

            merged_row = {
                **metadata_map[doc_id],
                **row,
            }

            merged_rows.append(merged_row)

            matched += 1

            if matched % 1000 == 0:
                print(f"Matched {matched} documents...")

    print(f"Total merged rows: {len(merged_rows)}")

    # =========================================================
    # Export
    # =========================================================
    print("Converting to DataFrame...")

    merged_df = pd.DataFrame(merged_rows)

    print(f"Saving parquet to: {output_path}")

    merged_df.to_parquet(
        output_path,
        index=False,
    )

    print("Done!")


if __name__ == "__main__":
    main()