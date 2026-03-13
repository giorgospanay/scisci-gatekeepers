"""
author_publication_years.py

Reads a large OpenAlex works TSV file and produces a TSV with three columns:
    author_id  |  first_year  |  last_year

Supports checkpointing: saves progress every CHECKPOINT_EVERY rows so the job
can be resubmitted and resume from where it left off if it gets killed.
"""

import csv
import ast
import sys
import pickle
from pathlib import Path

# Some fields (e.g. referenced_works) can be very large
csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────────
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path   = "/N/scratch/gpanayio"

metadata_path   = (
    f"{raw_workspace_path}/"
    "works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
)
output_path      = f"{out_workspace_path}/author_publication_years.tsv"
checkpoint_path  = f"{out_scratch_path}/author_years_checkpoint.pkl"

# ── Settings ──────────────────────────────────────────────────────────────────
CHECKPOINT_EVERY = 1_000_000   # save state every 1M rows
LOG_EVERY        =   500_000   # print progress every 500k rows

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_list_field(raw: str) -> list:
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def load_checkpoint(path: str) -> tuple[dict, int]:
    """Return (author_years dict, last completed row_num) or fresh state."""
    p = Path(path)
    if p.exists():
        print(f"Checkpoint found: {path}", flush=True)
        with open(p, "rb") as f:
            state = pickle.load(f)
        author_years = state["author_years"]
        last_row     = state["last_row"]
        print(f"  Resuming from row {last_row:,} with {len(author_years):,} authors so far.", flush=True)
        return author_years, last_row
    print("No checkpoint found — starting from scratch.", flush=True)
    return {}, 0


def save_checkpoint(path: str, author_years: dict, last_row: int):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"author_years": author_years, "last_row": last_row}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    Path(tmp).replace(path)   # atomic rename so a crash never corrupts the checkpoint
    print(f"  [checkpoint] saved at row {last_row:,}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    author_years, resume_from = load_checkpoint(checkpoint_path)

    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        sys.exit(f"ERROR: input file not found:\n  {metadata_path}")

    print(f"Reading: {metadata_path}", flush=True)

    with open(metadata_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")

        for row_num, row in enumerate(reader, start=1):

            # ── Skip already-processed rows ───────────────────────────────────
            if row_num <= resume_from:
                if row_num % 5_000_000 == 0:
                    print(f"  … fast-forwarding, skipped {row_num:,} rows", flush=True)
                continue

            # ── Progress log ──────────────────────────────────────────────────
            if row_num % LOG_EVERY == 0:
                print(f"  … processed {row_num:,} rows | unique authors: {len(author_years):,}", flush=True)

            # ── Checkpoint save ───────────────────────────────────────────────
            if row_num % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_path, author_years, row_num)

            # ── Parse year ────────────────────────────────────────────────────
            year_raw = row.get("publication_year", "").strip()
            if not year_raw:
                continue
            try:
                year = int(year_raw)
            except ValueError:
                continue

            # ── Parse author IDs ──────────────────────────────────────────────
            author_ids = parse_list_field(row.get("authorships:author:id", ""))

            for entry in author_ids:
                ids = entry if isinstance(entry, list) else [entry]
                for aid in ids:
                    if not aid:
                        continue
                    aid = str(aid)
                    if aid not in author_years:
                        author_years[aid] = [year, year]
                    else:
                        if year < author_years[aid][0]:
                            author_years[aid][0] = year
                        if year > author_years[aid][1]:
                            author_years[aid][1] = year

    print(f"Finished. Total rows: {row_num:,} | Unique authors: {len(author_years):,}", flush=True)

    # ── Write final output ────────────────────────────────────────────────────
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing output: {output_path}", flush=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["author_id", "first_year", "last_year"])
        for aid, (first, last) in sorted(author_years.items()):
            writer.writerow([aid, first, last])

    # ── Clean up checkpoint once done ─────────────────────────────────────────
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        ckpt.unlink()
        print("Checkpoint deleted (job complete).", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()