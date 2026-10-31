from __future__ import annotations

import argparse
import gc
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import BTS_BASE_URL, FLIGHTS_PATH, RAW_DIR, FEATURES, TARGET

ARCHIVE_RE = re.compile(r"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_(\d{4})_(\d{1,2})\.zip", re.I)
# BTS has used more than one header convention over time. Map both the legacy
# uppercase/underscore names and the current title-case names to our canonical schema.
COLUMN_ALIASES = {
    "OPUNIQUECARRIER": "OP_UNIQUE_CARRIER",
    "OPCARRIER": "OP_CARRIER",
    "REPORTINGAIRLINE": "OP_CARRIER",
    "MARKETINGAIRLINENETWORK": "OP_CARRIER",
    "ORIGIN": "ORIGIN",
    "DEST": "DEST",
    "CRSDEPTIME": "CRS_DEP_TIME",
    "DEPDELAY": "DEP_DELAY",
    "DEPDELAYMINUTES": "DEP_DELAY",
    "DISTANCE": "DISTANCE",
    "MONTH": "MONTH",
    "DAYOFWEEK": "DAY_OF_WEEK",
    "FLDATE": "FL_DATE",
    "FLIGHTDATE": "FL_DATE",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "DIVERTED": "DIVERTED",
}


def canonical_bts_column(column: str) -> str | None:
    """Return this project's canonical name for a recognized BTS column."""
    key = re.sub(r"[^A-Z0-9]", "", str(column).strip().lstrip("\ufeff").upper())
    return COLUMN_ALIASES.get(key)


def find_latest_archives(months: int = 3) -> list[tuple[int, int, str]]:
    """Read BTS's official PREZIP directory and return latest available monthly archives."""
    if months < 1:
        raise ValueError("months harus >= 1")
    response = requests.get(BTS_BASE_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    found = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        match = ARCHIVE_RE.search(href) or ARCHIVE_RE.search(anchor.get_text(" ", strip=True))
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            filename = match.group(0)
            if 1 <= month <= 12:
                found[(year, month)] = filename
    if not found:
        raise RuntimeError("Tidak menemukan arsip BTS yang cocok pada direktori resmi. Periksa koneksi atau format sumber.")
    return [(year, month, found[(year, month)]) for year, month in sorted(found)[-months:]]


def download_archive(filename: str, force: bool = False) -> Path:
    """Download a BTS ZIP, resuming a partial download when the server supports Range."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / filename
    temp = destination.with_suffix(destination.suffix + ".part")

    if force:
        destination.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
    elif destination.exists() and destination.stat().st_size > 1000 and zipfile.is_zipfile(destination):
        print(f"Arsip sudah tersedia, unduhan dilewati: {filename}", flush=True)
        return destination

    # A previous run can be killed after finishing the bytes but before renaming.
    if temp.exists() and temp.stat().st_size > 1000 and zipfile.is_zipfile(temp):
        temp.replace(destination)
        print(f"Arsip sementara sudah lengkap; dipulihkan: {filename}", flush=True)
        return destination

    url = urljoin(BTS_BASE_URL, filename)
    offset = temp.stat().st_size if temp.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    if offset:
        print(f"Melanjutkan unduhan {filename} dari {offset / (1024 * 1024):.1f} MB", flush=True)
    else:
        print(f"Mengunduh arsip BTS: {filename}", flush=True)

    with requests.get(url, stream=True, headers=headers, timeout=(30, 180)) as response:
        # If the server supports byte ranges it should return 206 for a resume.
        # If it ignores Range and returns 200, overwrite the partial file safely.
        if offset and response.status_code == 416:
            temp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Server menolak resume untuk {filename} (HTTP 416). "
                "Jalankan bootstrap sekali lagi untuk memulai unduhan ulang yang bersih."
            )
        response.raise_for_status()
        if offset and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            range_match = re.match(r"bytes\s+(\d+)-", content_range, re.I)
            if not range_match or int(range_match.group(1)) != offset:
                temp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Respons resume tidak cocok untuk {filename} (Content-Range: {content_range!r}). "
                    "Ulangi bootstrap untuk memulai unduhan baru."
                )
            mode = "ab"
            total_bytes = None
            total_match = re.match(r"bytes\s+\d+-\d+/(\d+)", content_range, re.I)
            if total_match:
                total_bytes = int(total_match.group(1))
            downloaded = offset
        else:
            mode = "wb"
            downloaded = 0
            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length and content_length.isdigit() else None

        next_report = ((downloaded // (5 * 1024 * 1024)) + 1) * (5 * 1024 * 1024)
        with temp.open(mode) as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total_bytes:
                        print(
                            f"Unduhan {filename}: {downloaded / (1024 * 1024):.1f} / "
                            f"{total_bytes / (1024 * 1024):.1f} MB",
                            flush=True,
                        )
                    else:
                        print(f"Unduhan {filename}: {downloaded / (1024 * 1024):.1f} MB", flush=True)
                    next_report += 5 * 1024 * 1024

    if not zipfile.is_zipfile(temp):
        raise RuntimeError(
            f"Unduhan {filename} belum menjadi ZIP valid. File .part dipertahankan "
            "agar proses berikutnya dapat mencoba melanjutkan unduhan."
        )
    temp.replace(destination)
    print(f"Unduhan selesai: {filename} ({destination.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
    return destination


def _read_bts_zip(path: Path) -> pd.DataFrame:
    """Read only required fields and normalize legacy/current BTS header names."""
    with zipfile.ZipFile(path) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise RuntimeError(f"Tidak menemukan CSV di {path.name}")

        # Read only the header first so large monthly archives do not need to be
        # loaded with every BTS field (there can be over 100 columns).
        with archive.open(csv_files[0]) as stream:
            header = pd.read_csv(stream, nrows=0, low_memory=False)
        original_columns = list(header.columns)

        selected_columns: list[str] = []
        rename_columns: dict[str, str] = {}
        canonical_seen: set[str] = set()
        for original in original_columns:
            canonical = canonical_bts_column(original)
            if canonical is None or canonical in canonical_seen:
                continue
            selected_columns.append(original)
            rename_columns[original] = canonical
            canonical_seen.add(canonical)

        if not selected_columns:
            preview = ", ".join(map(str, original_columns[:25]))
            raise ValueError(
                "Tidak ada kolom BTS yang dikenali. "
                f"Contoh header CSV: {preview}"
            )

        with archive.open(csv_files[0]) as stream:
            frame = pd.read_csv(stream, low_memory=False, usecols=selected_columns)

    frame = frame.rename(columns=rename_columns)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def clean_flights(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize BTS columns and generate a leak-free delay>=15-minute target."""
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    if "DEP_DELAY" not in data.columns:
        raise ValueError("Kolom DEP_DELAY tidak ditemukan; pastikan memakai dataset BTS On-Time Performance.")

    carrier_col = "OP_UNIQUE_CARRIER" if "OP_UNIQUE_CARRIER" in data.columns else "OP_CARRIER"
    required = [carrier_col, "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY", "DISTANCE"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Kolom penting tidak tersedia pada dataset: {missing}")

    if "CANCELLED" in data.columns:
        cancelled = pd.to_numeric(data["CANCELLED"], errors="coerce").fillna(0)
        data = data.loc[cancelled == 0].copy()
    if "DIVERTED" in data.columns:
        diverted = pd.to_numeric(data["DIVERTED"], errors="coerce").fillna(0)
        data = data.loc[diverted == 0].copy()

    if "MONTH" not in data.columns and "FL_DATE" in data.columns:
        data["MONTH"] = pd.to_datetime(data["FL_DATE"], errors="coerce").dt.month
    if "DAY_OF_WEEK" not in data.columns and "FL_DATE" in data.columns:
        # BTS convention in this project: Sunday=1 ... Saturday=7.
        dates = pd.to_datetime(data["FL_DATE"], errors="coerce")
        data["DAY_OF_WEEK"] = ((dates.dt.weekday + 1) % 7) + 1
    for column in ["CRS_DEP_TIME", "DEP_DELAY", "DISTANCE", "MONTH", "DAY_OF_WEEK"]:
        if column not in data.columns:
            raise ValueError(f"Kolom {column} dibutuhkan dan tidak bisa diturunkan dari sumber.")
        data[column] = pd.to_numeric(data[column], errors="coerce")

    standardized = pd.DataFrame({
        "OP_CARRIER": data[carrier_col].astype("string").str.strip().str.upper(),
        "ORIGIN": data["ORIGIN"].astype("string").str.strip().str.upper(),
        "DEST": data["DEST"].astype("string").str.strip().str.upper(),
        "CRS_DEP_TIME": data["CRS_DEP_TIME"],
        "MONTH": data["MONTH"],
        "DAY_OF_WEEK": data["DAY_OF_WEEK"],
        "DISTANCE": data["DISTANCE"],
        TARGET: (data["DEP_DELAY"] >= 15).astype("int8"),
    })
    standardized = standardized.replace([float("inf"), float("-inf")], pd.NA)
    standardized = standardized.dropna(subset=FEATURES + [TARGET])
    standardized = standardized.loc[
        standardized["CRS_DEP_TIME"].between(0, 2359)
        & standardized["MONTH"].between(1, 12)
        & standardized["DAY_OF_WEEK"].between(1, 7)
        & standardized["DISTANCE"].gt(0)
    ].copy()
    for column in ["OP_CARRIER", "ORIGIN", "DEST"]:
        standardized[column] = standardized[column].astype(str)
    return standardized[FEATURES + [TARGET]].reset_index(drop=True)


def run_pipeline(months: int = 3, force: bool = False) -> Path:
    """Download and clean one month at a time to limit peak memory use."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    archive_list = find_latest_archives(months)
    FLIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_output = FLIGHTS_PATH.with_suffix(".parquet.part")
    temp_output.unlink(missing_ok=True)

    writer = None
    total_rows = 0
    delay_rows = 0
    observed_classes: set[int] = set()
    print("Arsip BTS yang diproses:", flush=True)
    try:
        for year, month, filename in archive_list:
            print(f" - {year}-{month:02d}: {filename}", flush=True)
            path = download_archive(filename, force=force)
            raw = _read_bts_zip(path)
            cleaned = clean_flights(raw).drop_duplicates().reset_index(drop=True)
            del raw
            gc.collect()

            if cleaned.empty:
                print(f"Tidak ada baris valid setelah pembersihan: {filename}", flush=True)
                del cleaned
                gc.collect()
                continue

            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(temp_output), table.schema, compression="snappy")
            else:
                # Keep a stable Arrow schema if pandas infers a different integer width.
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            total_rows += len(cleaned)
            delay_rows += int(cleaned[TARGET].sum())
            observed_classes.update(int(value) for value in cleaned[TARGET].unique())
            print(f"Selesai membersihkan {filename}: {len(cleaned):,} baris valid", flush=True)
            del table, cleaned
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0 or not temp_output.exists():
        temp_output.unlink(missing_ok=True)
        raise RuntimeError("Tidak ada dataset yang berhasil diproses.")
    if len(observed_classes) < 2:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError("Data hanya memiliki satu kelas target; pilih jumlah bulan yang lebih besar.")

    temp_output.replace(FLIGHTS_PATH)
    delay_rate = delay_rows / total_rows
    print(
        f"Data bersih tersimpan: {FLIGHTS_PATH} | {total_rows:,} baris | "
        f"{delay_rate:.1%} delay >=15 menit",
        flush=True,
    )
    return FLIGHTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh dan bersihkan dataset BTS On-Time Performance.")
    parser.add_argument("--months", type=int, default=3, help="Jumlah arsip bulanan terbaru yang diambil (default: 3).")
    parser.add_argument("--force", action="store_true", help="Unduh ulang arsip walau file sudah ada.")
    args = parser.parse_args()
    run_pipeline(months=args.months, force=args.force)

if __name__ == "__main__":
    main()
