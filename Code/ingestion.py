from pathlib import Path
import pandas as pd
from pypdf import PdfReader
"""
Validates user-uploaded files: exactly 1 CSV and up to 2 PDFs.
"""
def ingest_files(paths:list[str]):
    """Validates file paths for required CSV and optional PDFs."""
    csvs = 0
    pdfs = 0

    results = {"csvs":[],"pdfs":[]}

    for path in paths:
        extn = Path(path).suffix.lower()

        if extn not in {".csv", ".pdf"}:
            raise ValueError("Unsupported file")

        if extn == ".csv":
            csvs += 1
            results["csvs"].append(path)
        elif extn == ".pdf":
            pdfs += 1
            results["pdfs"].append(path)
    if csvs != 1:
        raise ValueError("Exactly 1 CSV required")

    if pdfs > 2:
        raise ValueError("Maximum 2 PDFs allowed")
    
    # validating csv file
    path_csv= valid_csv(results["csvs"])

    # validating pdf files
    paths_pdf,errors,is_pdf = valid_pdf(results["pdfs"])
    print(f"{paths_pdf} | {errors} |{is_pdf} ")

    # return only valid csv and pdfs
    return path_csv,paths_pdf 

# Validates CSV file for content and structure.
def valid_csv(paths:list[str]):
    """Checks CSV for valid headers, rows, and minimum data."""
    try:
        df = pd.read_csv(paths[0])
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is completely empty (0 bytes)")
    # non empty
    if df.empty:
        raise ValueError("CSV is empty")
    
    #  has header and rows
    if df.columns.isnull().any() or len(df.columns) == 0:
        raise ValueError("CSV has no valid header row")
    #  has at least 10 rows
    if len(df) < 10:
        raise ValueError("CSV must have at least 10 rows")
    return paths[0]



# Validates PDF files for readability and content.
def valid_pdf(pdf_paths, min_characters=20):
    """
    Validates PDFs for extractable text and usability.
    """
    valid_pdfs = []
    errors = []

    for item in pdf_paths:
        pdf_path = item[0] if isinstance(item, list) else item
        pdf_path = Path(pdf_path)

        try:
            # File size
            if pdf_path.stat().st_size == 0:
                raise ValueError("empty file (0 bytes)")

            #Page count
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                raise ValueError("no pages")

            # Extractable text
            total_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    total_text += text

            if len(total_text.strip()) < min_characters:
                raise ValueError("no readable text (likely scanned PDF)")

            # Passed all checks
            valid_pdfs.append(pdf_path)

        except Exception as e:
            errors.append({
                "file": str(pdf_path),
                "error": str(e)
            })

    # Final usability gate
    is_usable = len(valid_pdfs) > 0

    return valid_pdfs, errors,is_usable