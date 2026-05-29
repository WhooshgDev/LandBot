import random
import polars as pl

random.seed(42)

df = pl.read_parquet("data/LandLawDocumentCleaned.parquet")

df = df.with_columns(
    pl.col("content").str.len_chars().alias("content_length")
)

large_df = df.filter(pl.col("content_length") >= 500)

print(f"Total 500+ char chunks: {large_df.height:,}")

rows = large_df.select([
    "chunk_id",
    "document_id",
    "title",
    "article",
    "clause",
    "point",
    "content_length",
    "content",
]).to_dicts()

sample_size = min(30, len(rows))
sample_rows = random.sample(rows, sample_size)

for i, row in enumerate(sample_rows):
    print("=" * 120)
    print(f"SAMPLE {i}")
    print("chunk_id:", row["chunk_id"])
    print("document_id:", row["document_id"])
    print("title:", row["title"])
    print("article:", row["article"])
    print("clause:", row["clause"])
    print("point:", row["point"])
    print("length:", row["content_length"])
    print()
    print(row["content"])