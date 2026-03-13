import csv
import ast
import sys
from pathlib import Path
 
# Some fields (e.g. referenced_works) can be very large
csv.field_size_limit(sys.maxsize)
 
# ── Paths ────────────────────────────────────────────────────────────────────
raw_workspace_path  = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path  = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path    = "/N/scratch/gpanayio"
 
metadata_path = (
	f"{raw_workspace_path}/"
	"works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
)
output_path = f"{out_workspace_path}/author_publication_years.tsv"
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_list_field(raw: str) -> list:
	"""
	Safely parse a field that may be a JSON-like list string, e.g.
		[5041754959, 5060344265]
	or the escaped variant produced by the TSV writer:
		[\"5041754959\",\"5060344265\"]
	Returns an empty list on any parse failure.
	"""
	raw = raw.strip()
	if not raw or raw in ("[]", ""):
		return []
	try:
		return ast.literal_eval(raw)
	except Exception:
		return []
 
 
def main():
	# author_id -> [min_year, max_year]
	author_years: dict[str | int, list[int]] = {}
 
	metadata_file = Path(metadata_path)
	if not metadata_file.exists():
		sys.exit(f"ERROR: input file not found:\n  {metadata_path}")
 
	print(f"Reading: {metadata_path}", flush=True)
 
	with open(metadata_path, "r", encoding="utf-8") as fh:
		reader = csv.DictReader(fh, delimiter="\t")
 
		for row_num, row in enumerate(reader, start=1):
			# Progress indicator every 500 000 rows
			if row_num % 500_000 == 0:
				print(f"  … processed {row_num:,} rows", flush=True)
 
			# Publication year – skip rows with no usable year
			year_raw = row.get("publication_year", "").strip()
			if not year_raw:
				continue
			try:
				year = int(year_raw)
			except ValueError:
				continue
 
			# Author IDs – stored as a list in the authorships:author:id column
			author_ids_raw = row.get("authorships:author:id", "")
			author_ids = parse_list_field(author_ids_raw)
 
			# Each element may itself be a list (one entry per authorship row
			# when the file is "exploded"), or a plain scalar id.
			for entry in author_ids:
				if isinstance(entry, list):
					ids = entry
				else:
					ids = [entry]
 
				for aid in ids:
					if aid is None or aid == "":
						continue
					# Normalise to string for consistent dict keys
					aid = str(aid)
 
					if aid not in author_years:
						author_years[aid] = [year, year]
					else:
						if year < author_years[aid][0]:
							author_years[aid][0] = year
						if year > author_years[aid][1]:
							author_years[aid][1] = year
 
	print(f"Total rows processed: {row_num:,}", flush=True)
	print(f"Unique authors found: {len(author_years):,}", flush=True)
 
	# ── Write output ──────────────────────────────────────────────────────────
	out_file = Path(output_path)
	out_file.parent.mkdir(parents=True, exist_ok=True)
 
	print(f"Writing: {output_path}", flush=True)
	with open(output_path, "w", encoding="utf-8", newline="") as fh:
		writer = csv.writer(fh, delimiter="\t")
		writer.writerow(["author_id", "first_year", "last_year"])
		for aid, (first, last) in sorted(author_years.items()):
			writer.writerow([aid, first, last])
 
	print("Done.", flush=True)
 
 
if __name__ == "__main__":
	main()